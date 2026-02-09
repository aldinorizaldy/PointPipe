from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import io
from models.pointnet import PointNetSeg
from src.preprocess import PointCloudProcessor

app = FastAPI(title="PointPipe 3D API")

# Setup
processor = PointCloudProcessor(num_points=4096)
DEVICE = torch.device("cpu") # Mac deployment usually uses CPU
model = PointNetSeg(num_classes=9)

# Load the brain
@app.on_event("startup")
def load_model():
    # map_location='cpu' ensures cluster-trained weights work on your Mac
    model.load_state_dict(torch.load("weights/pointnet_final.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("🚀 PointNet model loaded successfully on CPU")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read uploaded .txt or .npy file
    contents = await file.read()
    data = np.loadtxt(io.BytesIO(contents)) # Assuming XYZ + others
    
    # 2. Preprocess (Center, Scale, Sample to 4096)
    xyz = data[:, :3]
    if xyz.shape[0] == 4096:
        input_points = processor.normalize(xyz)
    else:
        input_points = processor.process_for_inference(xyz)
    
    # 3. Convert to Tensor (B, C, N)
    input_tensor = torch.from_numpy(input_points.T).float().unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        preds = model(input_tensor) # Shape: [1, 4096, 9]
        labels = torch.argmax(preds, dim=2).squeeze().tolist()

    # 5. Map indices to names
    class_names = [processor.label_map.get(l, "Unknown") for l in labels]
    
    return {
        "filename": file.filename,
        "points_processed": len(labels),
        "predictions": class_names
    }
