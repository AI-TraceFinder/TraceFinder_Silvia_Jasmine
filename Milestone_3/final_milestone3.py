# ============================================================
# MILESTONE 3: CNN for Scanner Identification (FINAL VERSION)
# TraceFinder - Forensic Scanner Identification Project
# ============================================================
# This script trains a CNN on 200 raw images per scanner
# with proper image-wise split to prevent data leakage
# ============================================================

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------
# REPRODUCIBILITY
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CONFIGURATION
# -------------------------
class Config:
    # CHANGE THIS PATH TO YOUR DATASET LOCATION
    DATA_PATH = r"C:\Forensic_Project\dataset"
    
    # Training parameters
    BATCH_SIZE = 32  # Reduced for better generalization
    EPOCHS = 80
    LEARNING_RATE = 3e-4
    IMAGE_SIZE = 224
    NUM_WORKERS = 0  # Keep 0 for Windows
    
    # Early stopping
    PATIENCE = 15
    
    # Augmentation - DISABLED for scanner identification
    # Scanner artifacts are subtle and augmentation destroys them
    USE_AUGMENTATION = False

config = Config()

print("="*80)
print("TRACEFINDER - MILESTONE 3: CNN TRAINING")
print("="*80)
print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Dataset Path: {config.DATA_PATH}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Epochs: {config.EPOCHS}")
print(f"  Learning Rate: {config.LEARNING_RATE}")
print(f"  Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
print("="*80)

# ============================================================
# DATASET CLASS
# ============================================================
class ForensicDataset(Dataset):
    """Dataset for forensic scanner images"""
    
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Read image in grayscale
        img = cv2.imread(self.paths[idx], 0)
        
        if img is None:
            print(f"Warning: Could not read {self.paths[idx]}")
            img = np.zeros((224, 224), dtype=np.uint8)

        # Apply transforms
        img = self.transform(img)
        
        # Get filename for evaluation
        fname = os.path.basename(self.paths[idx])
        
        return img, self.labels[idx], fname

# ============================================================
# CNN MODEL ARCHITECTURE
# ============================================================
class TraceFinderCNN(nn.Module):
    """
    CNN for scanner identification
    Architecture: 4 conv blocks + GAP + 2 FC layers
    """
    
    def __init__(self, num_classes):
        super(TraceFinderCNN, self).__init__()

        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features
        fmap = self.features(x)
        
        # Retain gradients for Grad-CAM (only during training)
        if self.training and fmap.requires_grad:
            fmap.retain_grad()

        # Pool and classify
        pooled = self.gap(fmap).view(x.size(0), -1)
        out = self.classifier(pooled)
        
        return out, fmap

# ============================================================
# DATA LOADING WITH IMAGE-WISE SPLIT
# ============================================================
def load_data(root):
    """
    Load data and split by unique images (not patches)
    This prevents data leakage between train/val/test sets
    """
    
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print(f"{'='*80}")
    
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset path not found: {root}")
    
    paths, labels = [], []
    class_names = sorted(os.listdir(root))
    
    print(f"\nFound {len(class_names)} scanner classes:")
    
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(root, cls)
        
        if not os.path.isdir(cls_path):
            continue
            
        files = [f for f in os.listdir(cls_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"  {idx}. {cls}: {len(files)} images")
        
        for f in files:
            paths.append(os.path.join(cls_path, f))
            labels.append(idx)
    
    print(f"\nTotal images loaded: {len(paths)}")
    
    # Extract unique image IDs
    # If files are named like "scan_001.jpg", the ID is "scan_001"
    image_ids = []
    for p in paths:
        basename = os.path.basename(p)
        # Remove extension
        img_id = os.path.splitext(basename)[0]
        image_ids.append(img_id)
    
    unique_ids = list(set(image_ids))
    print(f"Unique source images: {len(unique_ids)}")
    
    # Split by image IDs (70% train, 15% val, 15% test)
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, 
                                           random_state=42, shuffle=True)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, 
                                         random_state=42, shuffle=True)
    
    print(f"\nSplit:")
    print(f"  Train images: {len(train_ids)} ({len(train_ids)/len(unique_ids)*100:.1f}%)")
    print(f"  Val images: {len(val_ids)} ({len(val_ids)/len(unique_ids)*100:.1f}%)")
    print(f"  Test images: {len(test_ids)} ({len(test_ids)/len(unique_ids)*100:.1f}%)")
    
    # Filter paths and labels by split
    def filter_by_ids(ids):
        filtered = [(p, l) for p, l, i in zip(paths, labels, image_ids) if i in ids]
        return filtered
    
    train_data = filter_by_ids(train_ids)
    val_data = filter_by_ids(val_ids)
    test_data = filter_by_ids(test_ids)
    
    return train_data, val_data, test_data, class_names

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    correct, total = 0, 0
    loss_sum = 0.0
    
    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        loss_sum += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        # Progress update every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100*correct/total:.2f}%")
    
    accuracy = 100 * correct / total
    avg_loss = loss_sum / len(loader)
    
    return accuracy, avg_loss

def validate(model, loader, criterion):
    """Validate the model"""
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(imgs)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = loss_sum / len(loader)
    
    return accuracy, avg_loss

# ============================================================
# IMAGE-WISE EVALUATION (VOTING)
# ============================================================
def evaluate_imagewise(model, loader):
    """
    Evaluate using image-wise voting
    Multiple predictions per image are aggregated by majority vote
    """
    model.eval()
    votes = defaultdict(list)
    ground_truth = {}
    
    with torch.no_grad():
        for imgs, labels, filenames in loader:
            imgs = imgs.to(DEVICE)
            outputs, _ = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            
            for pred, label, filename in zip(preds, labels, filenames):
                # Extract image ID (remove extension)
                img_id = os.path.splitext(filename)[0]
                
                votes[img_id].append(pred)
                ground_truth[img_id] = label.item()
    
    # Vote for each image (majority wins)
    y_true, y_pred = [], []
    for img_id in votes:
        # Most common prediction
        voted_pred = Counter(votes[img_id]).most_common(1)[0][0]
        y_pred.append(voted_pred)
        y_true.append(ground_truth[img_id])
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, f1, cm, y_true, y_pred

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_training_curves(train_acc, val_acc, train_loss, val_loss):
    """Plot training curves"""
    epochs = range(1, len(train_acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: training_curves.png")
    plt.close()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.xlabel('Predicted Scanner', fontsize=12, fontweight='bold')
    plt.ylabel('True Scanner', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Image-wise Evaluation', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrix.png")
    plt.close()

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    """Main training pipeline"""
    
    start_time = datetime.now()
    
    # ------------------
    # 1. LOAD DATA
    # ------------------
    train_data, val_data, test_data, class_names = load_data(config.DATA_PATH)
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Check class distribution
    train_labels = [l for _, l in train_data]
    class_distribution = Counter(train_labels)
    
    print(f"\nClass Distribution in Training Set:")
    for idx, cls in enumerate(class_names):
        count = class_distribution[idx]
        percentage = 100 * count / len(train_labels)
        print(f"  {cls}: {count} samples ({percentage:.1f}%)")
    
    # ------------------
    # 2. CREATE TRANSFORMS
    # ------------------
    if config.USE_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # ------------------
    # 3. CREATE DATASETS
    # ------------------
    train_dataset = ForensicDataset(*zip(*train_data), train_transform)
    val_dataset = ForensicDataset(*zip(*val_data), val_transform)
    test_dataset = ForensicDataset(*zip(*test_data), val_transform)
    
    # ------------------
    # 4. CREATE WEIGHTED SAMPLER (for class imbalance)
    # ------------------
    class_weights_list = [1.0 / class_distribution[l] for l in train_labels]
    sampler = WeightedRandomSampler(class_weights_list, len(class_weights_list), 
                                   replacement=True)
    
    # ------------------
    # 5. CREATE DATALOADERS
    # ------------------
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             sampler=sampler, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    
    # ------------------
    # 6. CREATE MODEL
    # ------------------
    print(f"\n{'='*80}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*80}")
    
    model = TraceFinderCNN(num_classes=len(class_names)).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ------------------
    # 7. LOSS AND OPTIMIZER
    # ------------------
    # Class weights for imbalanced dataset
    class_weights = torch.tensor([1.0 / class_distribution[i] 
                                 for i in range(len(class_names))]).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # ------------------
    # 8. TRAINING LOOP
    # ------------------
    print(f"\n{'='*80}")
    print(f"TRAINING (Target: {config.EPOCHS} epochs)")
    print(f"{'='*80}\n")
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{config.EPOCHS}]")
        print(f"{'='*80}")
        
        # Train
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_acc, val_loss = validate(model, val_loader, criterion)
        
        # Store history
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f}")
        print(f"    Val Acc:   {val_acc:.2f}% | Val Loss:   {val_loss:.4f}")
        print(f"    Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'config': {
                    'image_size': config.IMAGE_SIZE,
                    'num_classes': len(class_names)
                }
            }, 'best_forensic_cnn.pth')
            print(f"\n  ✨ NEW BEST MODEL! Validation Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\n  No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # ------------------
    # 9. PLOT TRAINING CURVES
    # ------------------
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_training_curves(train_acc_history, val_acc_history,
                        train_loss_history, val_loss_history)
    
    # ------------------
    # 10. FINAL EVALUATION
    # ------------------
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load('best_forensic_cnn.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best validation accuracy was: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    print("\nEvaluating on test set (image-wise voting)...")
    test_acc, test_f1, test_cm, y_true, y_pred = evaluate_imagewise(model, test_loader)
    
    # Print results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nTest Accuracy (Image-wise): {test_acc:.2f}%")
    print(f"F1-Score (Weighted): {test_f1:.4f}")
    
    # Per-class accuracy
    print(f"\nPer-Class Performance:")
    print(f"{'Scanner':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)
    for idx, cls in enumerate(class_names):
        class_correct = test_cm[idx, idx]
        class_total = test_cm[idx, :].sum()
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        print(f"{cls:<20} {class_correct:<10} {class_total:<10} {class_acc:.2f}%")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_cm, class_names)
    
    # ------------------
    # 11. SUMMARY
    # ------------------
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTotal training time: {duration}")
    print(f"\nGenerated files:")
    print(f"  ✅ best_forensic_cnn.pth (trained model)")
    print(f"  ✅ training_curves.png (training visualization)")
    print(f"  ✅ confusion_matrix.png (evaluation results)")
    print(f"\n{'='*80}")
    print("Ready for Milestone 4 (Streamlit Deployment)!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()