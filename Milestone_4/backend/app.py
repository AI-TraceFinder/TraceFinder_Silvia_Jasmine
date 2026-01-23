from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import io
import base64
from datetime import datetime

# ============================================================
# MODEL ARCHITECTURE - MUST MATCH YOUR TRAINING
# ============================================================

class TraceFinderCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TraceFinderCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        fmap = self.features(x)
        pooled = self.gap(fmap).view(x.size(0), -1)
        out = self.classifier(pooled)
        return out, fmap

# ============================================================
# CONFIGURATION - UPDATE THESE TO MATCH YOUR TRAINING
# ============================================================

SCANNER_NAMES = [
    "Cannon120-1",
    "Cannon200",
    "EpsonV500",
    "Epsonv39-1",
    "Hp"
]

MODEL_PATH = "C:\\Users\\mamil\\OneDrive\\Desktop\\milestone4\\backend\\best_forensic_cnn.pth"

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TraceFinderCNN(num_classes=len(SCANNER_NAMES))

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"⚠️  Make sure {MODEL_PATH} is in the backend folder!")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="TraceFinder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history = []

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        gray = img_array
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    tensor = transform(gray).unsqueeze(0)
    return tensor, gray

# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, img_tensor, original_img, class_idx, device):
    try:
        model.eval()
        img_tensor = img_tensor.to(device)
        img_tensor.requires_grad = True
        
        outputs, _ = model(img_tensor)
        class_score = outputs[0, class_idx]
        
        model.zero_grad()
        class_score.backward(retain_graph=True)
        
        gradients = img_tensor.grad.data
        heatmap = torch.mean(gradients, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img
        
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except:
        return None

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "TraceFinder API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": True,
        "scanners": len(SCANNER_NAMES)
    }

@app.get("/scanners")
async def get_scanners():
    return {"scanners": SCANNER_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        img_tensor, original = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs, _ = model(img_tensor)
        
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = int(np.argmax(probs))
        predicted_scanner = SCANNER_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Top 3
        top_3_idx = np.argsort(probs)[::-1][:3]
        top_3 = [
            {
                "rank": i+1,
                "scanner": SCANNER_NAMES[idx],
                "probability": float(probs[idx]),
                "confidence": f"{probs[idx]*100:.2f}%"
            }
            for i, idx in enumerate(top_3_idx)
        ]
        
        # Grad-CAM
        gradcam = generate_gradcam(model, img_tensor, original, predicted_idx, device)
        
        # Log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({
            "timestamp": timestamp,
            "filename": file.filename,
            "scanner": predicted_scanner,
            "confidence": confidence
        })
        
        return {
            "success": True,
            "timestamp": timestamp,
            "filename": file.filename,
            "prediction": {
                "scanner": predicted_scanner,
                "confidence": confidence,
                "confidence_pct": f"{confidence*100:.2f}%"
            },
            "top_3": top_3,
            "all_probabilities": {
                SCANNER_NAMES[i]: float(probs[i]) for i in range(len(SCANNER_NAMES))
            },
            "gradcam": gradcam
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    return {"history": history[-10:]}

@app.delete("/history")
async def clear_history():
    global history
    history = []
    return {"success": True}

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🔍 TraceFinder Backend Starting...")
    print("="*60)
    print(f"Device: {device}")
    print(f"Scanners: {len(SCANNER_NAMES)}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
