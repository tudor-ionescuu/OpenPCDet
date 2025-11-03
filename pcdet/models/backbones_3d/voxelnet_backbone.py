"""
VoxelNet 3D Convolutional Middle Layers (CML)
Matches the architecture from the original paper and reference implementation
Based on: https://arxiv.org/abs/1711.06396
Reference: https://github.com/ModelBunker/VoxelNet-PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import spconv


class Conv3d(nn.Module):
    """3D Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class VoxelNetBackbone(nn.Module):
    """
    VoxelNet Convolutional Middle Layer (CML)
    
    Architecture from paper:
    - Conv3D(128, 64, k=3, s=(2,1,1), p=(1,1,1))
    - Conv3D(64, 64, k=3, s=(1,1,1), p=(0,1,1))
    - Conv3D(64, 64, k=3, s=(2,1,1), p=(1,1,1))
    
    This progressively downsamples in Z direction while maintaining X,Y resolution
    """
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(VoxelNetBackbone, self).__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        
        # Three 3D convolutional blocks
        self.conv3d_1 = Conv3d(input_channels, 64, k=3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, k=3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, k=3, s=(2, 1, 1), p=(1, 1, 1))
        
        # Output will be 64 channels
        self.num_point_features = 64
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxel_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4) [batch_idx, z, y, x]
                batch_size: int
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                encoded_spconv_tensor_stride: int
        """
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        # Convert sparse voxel features to dense 4D tensor for regular Conv3D
        # grid_size is [X, Y, Z], we need [Z, Y, X] for indexing
        dense_shape = (batch_size, voxel_features.shape[1], 
                      self.grid_size[2], self.grid_size[1], self.grid_size[0])
        
        # Initialize dense feature tensor
        voxel_feature_dense = torch.zeros(
            dense_shape,
            dtype=voxel_features.dtype,
            device=voxel_features.device
        )
        
        # Fill in the non-empty voxels
        # voxel_coords: [batch_idx, z, y, x]
        batch_idx = voxel_coords[:, 0].long()
        z_idx = voxel_coords[:, 1].long()
        y_idx = voxel_coords[:, 2].long()
        x_idx = voxel_coords[:, 3].long()
        
        voxel_feature_dense[batch_idx, :, z_idx, y_idx, x_idx] = voxel_features
        
        # Apply 3D convolutions (dense convolutions as in original VoxelNet)
        x = self.conv3d_1(voxel_feature_dense)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        
        # Convert back to sparse format for HeightCompression
        # Create a fake sparse tensor that wraps the dense output
        # This is necessary for compatibility with OpenPCDet's pipeline
        batch_dict['spatial_features_3d'] = x
        
        # Create a sparse tensor wrapper for HeightCompression
        # The HeightCompression module will call .dense() on it
        class DenseTensorWrapper:
            def __init__(self, dense_tensor):
                self._dense = dense_tensor
            def dense(self):
                return self._dense
        
        batch_dict['encoded_spconv_tensor'] = DenseTensorWrapper(x)
        batch_dict['encoded_spconv_tensor_stride'] = 1  # No downsampling in X,Y
        
        return batch_dict
