"""
Voxel-FPN VFE (Voxel Feature Encoding) with Multi-Scale Fusion
Based on: Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection from LIDAR Point Clouds
Paper: Sensors 2020, 20, 704

This implements the VFE-FPN encoder with:
- Multi-scale voxelization (0.16m, 0.32m, 0.64m)
- PointNet-style VFE layers (2 layers with 64 channels)
- Early fusion through bottom-up upsampling and concatenation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VFELayer(nn.Module):
    """
    Single VFE Layer following PointNet architecture
    - Fully connected layer for pointwise features
    - Max pooling for global feature aggregation
    - Concatenation of pointwise and global features
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Fully connected layer
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        
    def forward(self, inputs):
        """
        Args:
            inputs: (num_voxels, max_points, in_channels)
        Returns:
            (num_voxels, max_points, out_channels)
        """
        # Point-wise feature transformation
        num_voxels, max_points, _ = inputs.shape
        x = inputs.view(-1, self.in_channels)  # (num_voxels * max_points, in_channels)
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        x = x.view(num_voxels, max_points, self.out_channels)
        
        # Global max pooling
        global_feat = torch.max(x, dim=1, keepdim=True)[0]  # (num_voxels, 1, out_channels)
        
        # Concatenate pointwise and global features
        global_feat_expanded = global_feat.expand(-1, max_points, -1)
        output = torch.cat([x, global_feat_expanded], dim=-1)  # (num_voxels, max_points, out_channels * 2)
        
        return output


class StackedVFELayers(nn.Module):
    """
    Stacked VFE Layers with final max pooling
    Paper configuration: 2 VFE layers with 64 output channels each
    """
    def __init__(self, in_channels, vfe_channels=[64, 64]):
        super().__init__()
        self.vfe_channels = vfe_channels
        
        # Build VFE layers
        self.vfe_layers = nn.ModuleList()
        c_in = in_channels
        for c_out in vfe_channels:
            self.vfe_layers.append(VFELayer(c_in, c_out))
            c_in = c_out * 2  # Due to concatenation in VFELayer
        
        self.out_channels = vfe_channels[-1]
        
    def forward(self, voxel_features, voxel_num_points):
        """
        Args:
            voxel_features: (num_voxels, max_points, in_channels)
            voxel_num_points: (num_voxels,)
        Returns:
            (num_voxels, out_channels)
        """
        # Create mask for valid points
        mask = self.get_paddings_indicator(voxel_num_points, voxel_features.shape[1])
        mask = mask.unsqueeze(-1).type_as(voxel_features)
        
        # Apply VFE layers
        x = voxel_features
        for vfe in self.vfe_layers:
            x = vfe(x)
            x = x * mask  # Apply mask to zero out padded points
        
        # Final max pooling across points dimension
        x = torch.max(x, dim=1)[0]  # (num_voxels, out_channels * 2)
        
        # Take only the first half (the concatenated part from last layer)
        # This matches the paper's final voxel feature dimension
        x = x[:, :self.out_channels]
        
        return x
    
    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        """
        Create boolean mask for valid points
        Args:
            actual_num: (batch_size,) actual number of points per voxel
            max_num: int, maximum number of points
        Returns:
            (batch_size, max_num) boolean mask
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator


class VoxelFPNVFE(nn.Module):
    """
    Voxel-FPN VFE: Multi-scale voxel feature encoding with early fusion
    
    Architecture:
    1. Process three scales (0.16m, 0.32m, 0.64m) independently through VFE layers
    2. Upsample coarser scales and concatenate with finer scales (bottom-up fusion)
    3. Output three fused feature maps at different scales
    
    Paper specifications:
    - Input: 9-feature points (x, y, z, r, dx, dy, dz, dx_plane, dy_plane)
    - VFE: 2 layers with 64 output channels
    - Output: 64-channel voxel features per scale
    """
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features
        
        # Multi-scale voxel sizes (from paper)
        self.voxel_sizes = model_cfg.get('VOXEL_SIZES', [[0.16, 0.16, 4.0], 
                                                           [0.32, 0.32, 4.0], 
                                                           [0.64, 0.64, 4.0]])
        self.num_scales = len(self.voxel_sizes)
        
        # Build VFE for each scale
        # Input features: 9 (x, y, z, r, dx, dy, dz, dx_plane, dy_plane)
        self.vfe_layers = nn.ModuleList()
        for i in range(self.num_scales):
            # Each VFE processes 9-feature points and outputs 64-channel voxel features
            self.vfe_layers.append(StackedVFELayers(in_channels=9, vfe_channels=[64, 64]))
        
        # Early fusion: upsampling layers for bottom-up fusion
        # We'll upsample with bilinear interpolation + conv to match dimensions
        self.fusion_convs = nn.ModuleList()
        for i in range(self.num_scales - 1):
            # After upsampling, we concatenate, so input is 64 + 64 = 128
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
        
        # Output feature dimension
        self.num_point_features_out = 64
        
    def get_output_feature_dim(self):
        return self.num_point_features_out
    
    def forward(self, batch_dict):
        """
        Process multi-scale voxels through VFE and perform early fusion
        
        Expected inputs in batch_dict:
            'voxels_scale_0': (N0, T0, 4) - finest scale voxels
            'voxel_num_points_scale_0': (N0,)
            'voxel_coords_scale_0': (N0, 4) - [batch_idx, z, y, x]
            (similar for scale_1 and scale_2)
            
        Outputs:
            'spatial_features_scale_0': (B, 64, H0, W0) - fused features
            (similar for scale_1 and scale_2)
            'spatial_features_2d': (B, 64, H0, W0) - finest scale for backbone_2d
            'num_bev_features': 64 - for compatibility with detector template
        """
        # Process each scale through VFE
        voxel_features_list = []
        spatial_features_list = []
        
        for scale_idx in range(self.num_scales):
            voxels = batch_dict[f'voxels_scale_{scale_idx}']
            voxel_num_points = batch_dict[f'voxel_num_points_scale_{scale_idx}']
            voxel_coords = batch_dict[f'voxel_coords_scale_{scale_idx}']
            
            # Compute augmented features (9 features total)
            # Original: x, y, z, r (4 features)
            # Add: dx, dy, dz (offset from voxel centroid)
            # Add: dx_plane, dy_plane (offset from xy-plane centroid)
            points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
                         voxel_num_points.type_as(voxels).view(-1, 1, 1)
            
            f_cluster = voxels[:, :, :3] - points_mean
            
            # XY plane centroid
            points_mean_xy = voxels[:, :, :2].sum(dim=1, keepdim=True) / \
                            voxel_num_points.type_as(voxels).view(-1, 1, 1)
            f_cluster_xy = voxels[:, :, :2] - points_mean_xy
            
            # Concatenate all features: [x, y, z, r, dx, dy, dz, dx_plane, dy_plane]
            voxel_features_augmented = torch.cat([
                voxels[:, :, :4],  # x, y, z, r
                f_cluster,          # dx, dy, dz
                f_cluster_xy        # dx_plane, dy_plane
            ], dim=-1)
            
            # Apply VFE
            voxel_features = self.vfe_layers[scale_idx](voxel_features_augmented, voxel_num_points)
            voxel_features_list.append(voxel_features)
            
            # Convert to spatial feature map (pseudo-image)
            batch_size = batch_dict['batch_size']
            spatial_shape = batch_dict[f'spatial_shape_scale_{scale_idx}']
            
            # Create dense feature map
            # spatial_shape is [H, W] - convert to plain integers
            if isinstance(spatial_shape, torch.Tensor):
                # Flatten and convert: handles both 1D [H,W] and 2D [[H,W]] tensors
                flat = spatial_shape.flatten().tolist()
                h, w = int(flat[0]), int(flat[1])
            elif isinstance(spatial_shape, np.ndarray):
                #Numpy: flatten and convert
                flat = spatial_shape.flatten()
                h, w = int(flat[0]), int(flat[1])
            else:
                # List/tuple: handle nested or flat
                if isinstance(spatial_shape[0], (list, tuple)):
                    h, w = int(spatial_shape[0][0]), int(spatial_shape[0][1])
                else:
                    h, w = int(spatial_shape[0]), int(spatial_shape[1])
            
            spatial_features = torch.zeros(
                batch_size, 64, h, w,
                dtype=voxel_features.dtype,
                device=voxel_features.device
            )
            
            # Fill in non-empty voxels
            batch_idx = voxel_coords[:, 0].long()
            spatial_idx_y = voxel_coords[:, 2].long()  # Y coordinate  
            spatial_idx_x = voxel_coords[:, 3].long()  # X coordinate
            
            # Clamp indices to be within valid range
            spatial_idx_y = torch.clamp(spatial_idx_y, 0, h - 1)
            spatial_idx_x = torch.clamp(spatial_idx_x, 0, w - 1)
            
            spatial_features[batch_idx, :, spatial_idx_y, spatial_idx_x] = voxel_features
            spatial_features_list.append(spatial_features)
        
        # Early fusion: bottom-up upsampling and concatenation
        # Start from coarsest scale (scale_2) and fuse upward
        fused_features = []
        
        # Scale 2 (coarsest) - no fusion, just use as is
        fused_features.append(spatial_features_list[2])
        
        # Scale 2 -> Scale 1 fusion
        upsampled = F.interpolate(spatial_features_list[2], 
                                 size=(spatial_features_list[1].shape[2], 
                                      spatial_features_list[1].shape[3]),
                                 mode='bilinear', align_corners=False)
        concatenated = torch.cat([spatial_features_list[1], upsampled], dim=1)
        fused_1 = self.fusion_convs[1](concatenated)
        fused_features.append(fused_1)
        
        # Scale 1 -> Scale 0 fusion
        upsampled = F.interpolate(fused_1,
                                 size=(spatial_features_list[0].shape[2],
                                      spatial_features_list[0].shape[3]),
                                 mode='bilinear', align_corners=False)
        concatenated = torch.cat([spatial_features_list[0], upsampled], dim=1)
        fused_0 = self.fusion_convs[0](concatenated)
        fused_features.append(fused_0)
        
        # Store outputs in batch_dict (reverse order: finest to coarsest)
        batch_dict['spatial_features_scale_0'] = fused_features[2]  # Finest (0.16m)
        batch_dict['spatial_features_scale_1'] = fused_features[1]  # Medium (0.32m)
        batch_dict['spatial_features_scale_2'] = fused_features[0]  # Coarsest (0.64m)
        
        # For compatibility, set the finest scale as spatial_features_2d for backbone_2d
        batch_dict['spatial_features_2d'] = fused_features[2]
        batch_dict['spatial_features'] = fused_features[2]
        
        return batch_dict
