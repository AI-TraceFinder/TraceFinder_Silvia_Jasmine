import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# MODEL (EXACT COPY FROM TRAINING)
# -----------------------------
class TraceFinderCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        fmap = self.features(x)
        pooled = self.gap(fmap).view(x.size(0), -1)
        out = self.classifier(pooled)
        return out, fmap

# -----------------------------
# LOAD MODEL
# -----------------------------
num_classes = 5
model = TraceFinderCNN(num_classes).to(DEVICE)

checkpoint = torch.load("best_forensic_cnn.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
img_path = "sample_test_image2.png"  # change if needed

img = cv2.imread(img_path, 0)
img = cv2.resize(img, (224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -----------------------------
# GRAD-CAM
# -----------------------------
feature_maps = []
gradients = []

def forward_hook(module, inp, out):
    feature_maps.append(out)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# last conv layer BEFORE pooling
model.features[24].register_forward_hook(forward_hook)
model.features[24].register_backward_hook(backward_hook)

# Forward + Backward
output, _ = model(img_tensor)
pred_class = output.argmax(dim=1).item()

model.zero_grad()
output[0, pred_class].backward()

# Grad-CAM computation
fmap = feature_maps[0].squeeze().cpu().detach().numpy()
grads = gradients[0].mean(dim=(1, 2)).cpu().detach().numpy()

cam = np.zeros(fmap.shape[1:], dtype=np.float32)
for i, w in enumerate(grads):
    cam += w * fmap[i]

cam = np.maximum(cam, 0)
cam /= cam.max() + 1e-8
cam = cv2.resize(cam, (224, 224))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

# -----------------------------
# SAVE RESULT
# -----------------------------
plt.figure(figsize=(5,5))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM Visualization")
plt.savefig("gradcam_output.png", bbox_inches="tight")
plt.show()

print("✅ Grad-CAM saved as gradcam_output.png")
