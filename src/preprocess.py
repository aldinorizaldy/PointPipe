import numpy as np

class PointCloudProcessor:
    def __init__(self, num_points=4096):
        self.num_points = num_points
        self.label_map = {
            1: "Powerline", 2: "Low Veg", 3: "Impervious", 
            4: "Car", 5: "Fence", 6: "Roof", 
            7: "Facade", 8: "Shrub", 9: "Tree"
        }

    def normalize(self, points):
        """Standard PointNet normalization: Center to 0 and scale to unit sphere."""
        # 1. Center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 2. Scale to unit sphere
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if dist > 0:
            points = points / dist
        return points

    def fix_num_points(self, points, labels=None):
        """Ensures the batch size is exactly num_points (4096)."""
        curr_num = points.shape[0]
        if curr_num >= self.num_points:
            # Random downsample
            idx = np.random.choice(curr_num, self.num_points, replace=False)
        else:
            # Random upsample (duplicate points)
            idx = np.random.choice(curr_num, self.num_points, replace=True)
            
        points = points[idx]
        if labels is not None:
            return points, labels[idx]
        return points

    def process_for_inference(self, raw_xyz):
        """Pipeline for the FastAPI endpoint."""
        # We only need first 3 columns for basic PointNet
        xyz = raw_xyz[:, :3]
        xyz = self.fix_num_points(xyz)
        xyz = self.normalize(xyz)
        return xyz