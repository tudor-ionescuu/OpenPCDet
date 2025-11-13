"""
Voxel-FPN RPN-FPN: 2D Backbone with Feature Pyramid Network
Based on: Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection from LIDAR Point Clouds
Paper: Sensors 2020, 20, 704

This implements the RPN-FPN decoder with:
- Bottom-up pathway: 3 blocks of 2D convolutions (3, 5, 5 layers)
- Top-down pathway: Deconvolutions with lateral connections
- Concatenation-based fusion (not addition)
- Output multi-scale feature maps P1, P2, P3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RPNFPNBackbone(nn.Module):
    """
    RPN-FPN: Region Proposal Network with Feature Pyramid Network
    
    Architecture from paper:
    Bottom-up pathway (on finest scale input):
        Block 1: 3 conv layers, 64 channels, stride [2, 1, 1]
        Block 2: 5 conv layers, 128 channels, stride [2, 1, 1, 1, 1]
        Block 3: 5 conv layers, 256 channels, stride [2, 1, 1, 1, 1]
    
    Top-down pathway:
        - Lateral connections from VFE-FPN outputs
        - Deconvolutions for upsampling
        - Concatenation fusion (not addition)
        - Output feature maps at 3 scales
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        # Bottom-up pathway
        # Block 1: Process finest scale (432, 496, 64) -> (216, 248, 64)
        self.block1 = nn.Sequential(
            ConvBNReLU(input_channels, 64, kernel_size=3, stride=2, padding=1),  # stride=2
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),               # stride=1
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),               # stride=1
        )
        
        # Block 2: (216, 248, 64) -> (108, 124, 128)
        self.block2 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),   # stride=2
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),  # stride=1
        )
        
        # Block 3: (108, 124, 128) -> (54, 62, 256)
        self.block3 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1),  # stride=2
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),  # stride=1
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),  # stride=1
        )
        
        # Lateral connections for each scale (process VFE-FPN features)
        # Each lateral connection processes the VFE output with a 3x3 conv
        self.lateral_scale_1 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)  # Medium scale
        self.lateral_scale_2 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)  # Coarsest scale
        
        # Reduce channels for top-down pathway
        # Block3 (256) -> 128 for concatenation with lateral (64)
        self.reduce_block3 = ConvBNReLU(256, 128, kernel_size=1, stride=1, padding=0)
        
        # Top-down pathway - Paper Figure 3 architecture
        # Level 3 (coarsest): Process concatenated features at scale 2
        # After concat: reduced_block3(128) + lateral(64) = 192 channels
        self.fpn_scale_2_conv = nn.Sequential(
            ConvBNReLU(192, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # Deconv from scale 2 to scale 1
        self.deconv_2to1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Level 2 (medium): Process concatenated features at scale 1  
        # After concat: lateral(64) + deconv(128) + block1(64) = 256 channels
        self.fpn_scale_1_conv = nn.Sequential(
            ConvBNReLU(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # Deconv from scale 1 to scale 0 (for final output)
        self.deconv_1to0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Level 1 (finest): Final processing
        # After concat: lateral_from_vfe(64) + deconv(64) = 128 channels
        self.fpn_scale_0_conv = nn.Sequential(
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # Output channels for detection head
        self.num_bev_features = 128
        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features_scale_0: (B, 64, H0, W0) - finest scale from VFE-FPN (0.16m)
                spatial_features_scale_1: (B, 64, H1, W1) - medium scale from VFE-FPN (0.32m)
                spatial_features_scale_2: (B, 64, H2, W2) - coarsest scale from VFE-FPN (0.64m)
                
        Returns:
            data_dict with added:
                spatial_features_2d: Final feature map for detection head
        """
        # Get multi-scale features from VFE-FPN (all fused from early fusion)
        vfe_scale_0 = data_dict['spatial_features_scale_0']  # Finest (432, 496, 64)
        vfe_scale_1 = data_dict['spatial_features_scale_1']  # Medium (216, 248, 64)
        vfe_scale_2 = data_dict['spatial_features_scale_2']  # Coarsest (108, 124, 64)
        
        # Bottom-up pathway on finest scale (Paper: Block 1, 2, 3)
        x_block1 = self.block1(vfe_scale_0)  # (216, 248, 64)
        x_block2 = self.block2(x_block1)      # (108, 124, 128)
        x_block3 = self.block3(x_block2)      # (54, 62, 256)
        
        # ===== Top-down pathway with lateral connections (FPN) =====
        # Paper: Deconv upsampling + lateral connections + concatenation fusion
        # Start from coarsest scale (block3) and fuse downward
        
        # Level 3 (coarsest, ~0.64m scale):
        # Start with block3 (54, 62, 256) - this is our coarsest RPN output
        # Upsample block3 to match block2 size (108, 124)
        up_3to2 = F.interpolate(x_block3, size=(x_block2.shape[2], x_block2.shape[3]), 
                               mode='bilinear', align_corners=False)  # (108, 124, 256)
        
        # Reduce channels of upsampled block3 from 256 to 128
        up_3to2_reduced = self.reduce_block3(up_3to2)  # (108, 124, 128)
        
        # Process VFE scale 2 through lateral connection
        lateral_2 = self.lateral_scale_2(vfe_scale_2)  # (108, 124, 64)
        
        # Concatenate: upsampled_block3(128) + lateral(64) = 192
        # This fuses top-down pathway from RPN with VFE features
        concat_2 = torch.cat([up_3to2_reduced, lateral_2], dim=1)  # (108, 124, 192)
        
        # Process concatenated features
        p2 = self.fpn_scale_2_conv(concat_2)  # (108, 124, 128)
        
        # Deconv to next level
        up_2to1 = self.deconv_2to1(p2)  # (216, 248, 128)
        
        # Level 2 (medium, ~0.32m scale):
        # Process VFE scale 1 through lateral connection
        lateral_1 = self.lateral_scale_1(vfe_scale_1)  # (216, 248, 64)
        
        # Concatenate: deconv(128) + lateral(64) + block1(64) = 256
        concat_1 = torch.cat([up_2to1, lateral_1, x_block1], dim=1)  # (216, 248, 256)
        
        # Process concatenated features
        p1 = self.fpn_scale_1_conv(concat_1)  # (216, 248, 128)
        
        # Deconv to finest level
        up_1to0 = self.deconv_1to0(p1)  # (432, 496, 64)
        
        # Level 1 (finest, ~0.16m scale):
        # Concatenate with original VFE finest scale
        concat_0 = torch.cat([up_1to0, vfe_scale_0], dim=1)  # (432, 496, 128)
        
        # Final processing
        p0 = self.fpn_scale_0_conv(concat_0)  # (432, 496, 128)
        
        # Downsample p0 for detection head (reduce computational cost)
        # Paper uses multi-scale detection, but for simplicity we use downsampled finest scale
        final_features = F.avg_pool2d(p0, kernel_size=2, stride=2)  # (216, 248, 128)
        
        # Store outputs
        data_dict['spatial_features_2d'] = final_features
        data_dict['spatial_features_p0'] = p0              # Finest
        data_dict['spatial_features_p1'] = p1              # Medium  
        data_dict['spatial_features_p2'] = p2              # Coarsest
        
        return data_dict
