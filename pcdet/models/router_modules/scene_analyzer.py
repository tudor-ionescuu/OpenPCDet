"""
Scene Complexity Estimator for Adaptive Routing
Analyzes point cloud characteristics to determine scene complexity
"""

import torch
import numpy as np
from collections import deque


class SceneComplexityEstimator:
    """
    Fast scene complexity estimation based on point cloud statistics.
    Target: <1ms computation time for real-time operation.
    """
    
    def __init__(self, point_cloud_range, history_size=10):
        """
        Args:
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            history_size: Number of frames to keep for temporal analysis
        """
        self.pc_range = point_cloud_range
        self.volume = (
            (point_cloud_range[3] - point_cloud_range[0]) *
            (point_cloud_range[4] - point_cloud_range[1]) *
            (point_cloud_range[5] - point_cloud_range[2])
        )
        
        # Temporal history for motion estimation
        self.history = deque(maxlen=history_size)
        
        # Complexity weights (tunable)
        self.weights = {
            'density': 0.3,
            'occupancy': 0.3,
            'z_variance': 0.2,
            'front_density': 0.2
        }
    
    def compute_complexity(self, points, prev_detections=None):
        """
        Compute scene complexity score
        
        Args:
            points: (N, 3+C) tensor or numpy array of point cloud
            prev_detections: Optional previous frame detections for temporal analysis
            
        Returns:
            dict with complexity metrics and overall score (0.0 to 1.0)
        """
        # Handle both tensor and numpy inputs
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        
        if points.shape[0] == 0:
            return self._empty_complexity()
        
        # Extract xyz coordinates (handle batch_idx if present)
        if points.shape[1] > 3 and points[:, 0].max() < 10:
            # Likely has batch_idx in first column
            xyz = points[:, 1:4]
        else:
            xyz = points[:, :3]
        
        xyz = xyz.float()
        
        # Compute metrics (all designed to be fast)
        metrics = {}
        
        # 1. Point Density (instant)
        metrics['density'] = self._compute_density(xyz)
        
        # 2. Voxel Occupancy (fast with sparse operations)
        metrics['occupancy'] = self._compute_occupancy(xyz)
        
        # 3. Height Variance (instant)
        metrics['z_variance'] = self._compute_z_variance(xyz)
        
        # 4. Front Region Density (instant, critical for AD)
        metrics['front_density'] = self._compute_front_density(xyz)
        
        # 5. Temporal Complexity (if available)
        if prev_detections is not None:
            metrics['temporal'] = self._compute_temporal_complexity(prev_detections)
        else:
            metrics['temporal'] = 0.5  # neutral
        
        # Compute weighted complexity score
        complexity_score = (
            self.weights['density'] * metrics['density'] +
            self.weights['occupancy'] * metrics['occupancy'] +
            self.weights['z_variance'] * metrics['z_variance'] +
            self.weights['front_density'] * metrics['front_density']
        )
        
        # Clamp to [0, 1]
        complexity_score = float(torch.clamp(torch.tensor(complexity_score), 0.0, 1.0))
        
        metrics['complexity_score'] = complexity_score
        return metrics
    
    def _compute_density(self, xyz):
        """Normalized point density"""
        num_points = xyz.shape[0]
        density = num_points / self.volume
        
        # Normalize to typical range (0-50k points in ~20000 m³)
        normalized_density = min(density / 2.5, 1.0)  # Cap at 1.0
        return normalized_density
    
    def _compute_occupancy(self, xyz):
        """Voxel occupancy ratio (fast sparse computation)"""
        voxel_size = torch.tensor([0.5, 0.5, 0.5])  # 50cm voxels
        
        # Compute voxel indices
        voxel_coords = torch.floor(xyz / voxel_size).long()
        
        # Unique voxels
        unique_voxels = torch.unique(voxel_coords, dim=0).shape[0]
        
        # Expected voxels in range
        x_range = (self.pc_range[3] - self.pc_range[0]) / voxel_size[0]
        y_range = (self.pc_range[4] - self.pc_range[1]) / voxel_size[1]
        z_range = (self.pc_range[5] - self.pc_range[2]) / voxel_size[2]
        expected_voxels = x_range * y_range * z_range
        
        occupancy = unique_voxels / expected_voxels
        return float(min(occupancy, 1.0))
    
    def _compute_z_variance(self, xyz):
        """Height variance (indicates terrain complexity)"""
        z_std = torch.std(xyz[:, 2])
        
        # Normalize (typical std: 0-3 meters)
        normalized_std = min(z_std / 3.0, 1.0)
        return float(normalized_std)
    
    def _compute_front_density(self, xyz):
        """Density in critical front region (0-30m ahead)"""
        # Front region mask
        front_mask = (xyz[:, 0] > 0) & (xyz[:, 0] < 30)
        front_points = front_mask.sum().float()
        total_points = xyz.shape[0]
        
        if total_points == 0:
            return 0.0
        
        front_ratio = front_points / total_points
        
        # Normalize (typical: 20-60% of points in front region)
        normalized_ratio = (front_ratio - 0.2) / 0.4
        return float(torch.clamp(normalized_ratio, 0.0, 1.0))
    
    def _compute_temporal_complexity(self, prev_detections):
        """
        Estimate temporal complexity based on previous detections
        
        Args:
            prev_detections: List of detection dicts from previous frames
        """
        if prev_detections is None or len(prev_detections) == 0:
            return 0.5
        
        # Simple heuristic: more objects = higher complexity
        num_objects = len(prev_detections)
        
        # Normalize (typical: 0-20 objects)
        temporal_complexity = min(num_objects / 20.0, 1.0)
        
        return temporal_complexity
    
    def _empty_complexity(self):
        """Return default metrics for empty point cloud"""
        return {
            'density': 0.0,
            'occupancy': 0.0,
            'z_variance': 0.0,
            'front_density': 0.0,
            'temporal': 0.0,
            'complexity_score': 0.0
        }
    
    def update_history(self, frame_info):
        """Update temporal history"""
        self.history.append(frame_info)


class FastComplexityEstimator:
    """
    Ultra-fast complexity estimator using only basic statistics
    Target: <0.5ms for extreme real-time requirements
    """
    
    def __init__(self):
        pass
    
    def estimate(self, points):
        """
        Minimal complexity estimation
        
        Args:
            points: (N, 3+) point cloud
            
        Returns:
            float: complexity score 0.0-1.0
        """
        if isinstance(points, np.ndarray):
            num_points = points.shape[0]
        else:
            num_points = points.shape[0]
        
        # Single metric: point count
        # Typical range: 10k (simple) to 100k (complex)
        complexity = min((num_points - 10000) / 90000, 1.0)
        complexity = max(complexity, 0.0)
        
        return float(complexity)
