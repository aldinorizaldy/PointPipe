import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from .models.pointnet import PointNetSeg
from .preprocess import prepare_pc

# --- INITIALIZATION ---
app = FastAPI(title="PointPipe 3D Segmentation API")

NUM_CLASSES = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = PointNetSeg(num_classes=NUM_CLASSES).to(device)

# Load the weights
try:
    # map_location ensures we can load GPU-trained weights onto a CPU server
    checkpoint = torch.load("weights/pointnet.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print("Warning: weights/pointnet.pth not found. API will run but predictions will be random.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- ROUTES ---

@app.get("/health")
def health():
    return {"status": "online", "model": "PointNet", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read file bytes
    content = await file.read()
    
    try:
        # 2. Parse (assuming space-separated XYZ text file)
        raw_data = np.fromstring(content.decode('utf-8'), sep=' ').reshape(-1, 3)
        
        # 3. Preprocess (sampling and normalizing)
        processed_points = prepare_pc(raw_data, num_points=4096)
        
        # 4. Prepare for PyTorch (B, C, N)
        input_tensor = torch.from_numpy(processed_points).transpose(1, 0).unsqueeze(0).to(device)
        
        # 5. Inference
        with torch.no_grad():
            output = model(input_tensor) # Shape: (1, 4096, 9)
            preds = torch.argmax(output, dim=2).cpu().numpy().flatten()
            
        return {
            "status": "success",
            "points_count": len(preds),
            "labels": preds.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")