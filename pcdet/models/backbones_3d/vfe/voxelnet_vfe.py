"""
VoxelNet VFE (Voxel Feature Encoding) Layer
Based on: https://arxiv.org/abs/1711.06396
Reference: https://github.com/ModelBunker/VoxelNet-PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    """Fully Connected Network for point-wise feature transformation"""
    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # x: (K, T, cin) where K is num voxels, T is max points per voxel
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


class VFELayer(nn.Module):
    """Voxel Feature Encoding layer - core innovation of VoxelNet"""
    def __init__(self, cin, cout):
        super(VFELayer, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)
    
    def forward(self, x, mask):
        """
        Args:
            x: (K, T, cin) voxel features
            mask: (K, T) mask for valid points
        Returns:
            (K, T, cout) point-wise concatenated features
        """
        # Point-wise feature
        pwf = self.fcn(x)
        
        # Locally aggregated feature (max pooling)
        laf = torch.max(pwf, dim=1)[0].unsqueeze(1).repeat(1, x.shape[1], 1)
        
        # Concatenate
        pwcf = torch.cat((pwf, laf), dim=2)
        
        # Apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()
        
        return pwcf


class SVFE(nn.Module):
    """Stacked Voxel Feature Encoding - complete feature learning network"""
    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFELayer(7, 32)   # VFE-1: 7 -> 32
        self.vfe_2 = VFELayer(32, 128)  # VFE-2: 32 -> 128
        self.fcn = FCN(128, 128)        # Final FCN
    
    def forward(self, x):
        # Create mask for valid points
        mask = torch.ne(torch.max(x, dim=2)[0], 0)
        
        # Stack VFE layers
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        
        # Final max pooling for voxel-wise feature
        x = torch.max(x, dim=1)[0]
        return x


class VoxelFeatureEncoderV2(nn.Module):
    """
    OpenPCDet wrapper for VoxelNet VFE
    Adapts SVFE to OpenPCDet's batch_dict interface
    """
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.svfe = SVFE()
        
    def get_output_feature_dim(self):
        return 128
    
    def forward(self, batch_dict):
        """
        Process voxels through VFE layers
        Input: batch_dict['voxels'] - (K, T, 4) [x, y, z, intensity]
        Output: batch_dict['voxel_features'] - (K, 128)
        """
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        
        # Compute voxel centroids
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / \
                      voxel_num_points.type_as(voxels).view(-1, 1, 1)
        
        # Relative offsets
        f_cluster = voxels[:, :, :3] - points_mean
        
        # Concatenate: [x,y,z,r] + [dx,dy,dz] = 7 features (paper's input)
        f_center = torch.cat([voxels[:, :, :4], f_cluster], dim=-1)
        
        # Get voxel features
        voxel_features = self.svfe(f_center)
        batch_dict['voxel_features'] = voxel_features
        
        return batch_dict