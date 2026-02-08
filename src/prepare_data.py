import numpy as np
import os
from tqdm import tqdm

def tile_dataset(input_path, output_dir, tile_size=30.0, stride=5.0):
    """Splits giant UTM files into smaller overlapping tiles."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data (X, Y, Z, Intensity, ..., Label)
    print(f"Loading {input_path}...")
    data = np.loadtxt(input_path)
    xyz = data[:, :3]
    labels = data[:, -1]
    
    x_min, y_min = np.min(xyz[:, :2], axis=0)
    x_max, y_max = np.max(xyz[:, :2], axis=0)
    
    tile_id = 0
    for x in tqdm(np.arange(x_min, x_max, stride)):
        for y in np.arange(y_min, y_max, stride):
            # Mask points within the current 10m x 10m tile
            mask = (xyz[:, 0] >= x) & (xyz[:, 0] < x + tile_size) & \
                   (xyz[:, 1] >= y) & (xyz[:, 1] < y + tile_size)
            
            tile_points = data[mask]
            if len(tile_points) > 512:  # Ignore tiny/empty tiles
                np.save(f"{output_dir}/tile_{tile_id}.npy", tile_points.astype(np.float32))
                tile_id += 1

if __name__ == "__main__":
    tile_dataset("data/Vaihingen3D/train.txt", "data/processed/train_tiles")
    tile_dataset("data/Vaihingen3D/test.txt", "data/processed/test_tiles")