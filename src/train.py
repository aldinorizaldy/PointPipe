import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from models.pointnet import PointNetSeg  # Assumes your model is here
from src.preprocess import PointCloudProcessor

# 1. Dataset Class for the tiled .npy files
class TiledVaihingenDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, num_points=4096):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.processor = PointCloudProcessor(num_points=num_points)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load tiled data (X, Y, Z, Intensity, ..., Label)
        data = np.load(self.files[idx])
        xyz = data[:, :3]
        labels = data[:, -1].astype(int) 
        
        # Normalize and Resample
        xyz, labels = self.processor.fix_num_points(xyz, labels)
        xyz = self.processor.normalize(xyz)
        
        # PointNet expects (Batch, Channels, Num_Points) -> (B, 3, 4096)
        return torch.from_numpy(xyz.T), torch.from_numpy(labels).long()

# 2. Training Function
def train():
    # Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Setup
    train_ds = TiledVaihingenDataset("data/processed/train_tiles")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model Setup (9 classes for Vaihingen)
    model = PointNetSeg(num_classes=9).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (points, labels) in enumerate(train_loader):
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()

            pred = model(points) # PointNet usually returns (pred, trans_matrix)

            pred_flat = pred.contiguous().view(-1, 9)
            labels_flat = labels.view(-1)

            loss = criterion(pred_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Save Best Weights
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"weights/pointnet_epoch_{epoch+1}.pth")

    # Final Save
    torch.save(model.state_dict(), "weights/pointnet_final.pth")
    print("Training Complete. Weights saved to weights/pointnet_final.pth")

if __name__ == "__main__":
    train()
    