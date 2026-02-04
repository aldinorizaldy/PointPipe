import numpy as np

def prepare_pc(points: np.ndarray, num_points: int = 4096):
    """
    Standard Production Preprocessing for PointNet
    points: Nx3 array (X, Y, Z)
    """
    # 1. Subsampling (PointNet needs a fixed N)
    if len(points) >= num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
    points = points[idx, :]

    # 2. Centering (Crucial for 3D Deep Learning)
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # 3. Scaling
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / m
    
    return points.astype(np.float32)