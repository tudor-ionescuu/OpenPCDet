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
        self.lateral_scale_0 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.lateral_scale_1 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.lateral_scale_2 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Top-down pathway
        # From Block 3 (coarsest) to Block 2
        self.deconv3to2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Process concatenated features at scale 2
        # Input: upsampled (128) + lateral VFE (64) + Block2 output (128) = 320 channels
        # But according to the paper, we concatenate upsampled + lateral, then process
        # Upsampled: 128, Lateral: 64 -> 192 channels
        self.smooth_scale_2 = nn.Sequential(
            ConvBNReLU(192, 128, kernel_size=3, stride=2, padding=1),  # Downsample for next stage
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        # From processed Scale 2 to Scale 1
        self.deconv2to1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Process concatenated features at scale 1
        # Upsampled: 64, Lateral: 64 -> 128 channels
        self.smooth_scale_1 = nn.Sequential(
            ConvBNReLU(128, 64, kernel_size=3, stride=2, padding=1),  # Downsample for next stage
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # Final deconv for detection output at scale 0
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Output channels for each pyramid level
        self.num_bev_features = 128  # Final output channels (after concatenation)
        
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features_scale_0: (B, 64, H0, W0) - finest scale from VFE-FPN
                spatial_features_scale_1: (B, 64, H1, W1) - medium scale from VFE-FPN  
                spatial_features_scale_2: (B, 64, H2, W2) - coarsest scale from VFE-FPN
                
        Returns:
            data_dict with added:
                spatial_features_2d: Final feature map for detection head
                spatial_features_p1, p2, p3: Multi-scale pyramid features
        """
        # Get multi-scale features from VFE-FPN
        vfe_scale_0 = data_dict['spatial_features_scale_0']  # Finest (432, 496, 64)
        vfe_scale_1 = data_dict['spatial_features_scale_1']  # Medium (216, 248, 64)
        vfe_scale_2 = data_dict['spatial_features_scale_2']  # Coarsest (108, 124, 64)
        
        # Bottom-up pathway on finest scale
        x_block1 = self.block1(vfe_scale_0)  # (216, 248, 64)
        x_block2 = self.block2(x_block1)      # (108, 124, 128)
        x_block3 = self.block3(x_block2)      # (54, 62, 256)
        
        # Top-down pathway with lateral connections
        
        # Level 3 (coarsest): Process VFE scale 2 features
        lateral_2 = self.lateral_scale_2(vfe_scale_2)  # (108, 124, 64)
        
        # Upsample from block3 to match scale 2
        up_3to2 = self.deconv3to2(x_block3)  # (108, 124, 128)
        
        # Concatenate upsampled + lateral
        concat_2 = torch.cat([up_3to2, lateral_2], dim=1)  # (108, 124, 192)
        
        # Smooth and process scale 2
        smooth_2 = self.smooth_scale_2(concat_2)  # (54, 62, 128) - downsampled
        
        # Level 2 (medium): Process VFE scale 1 features
        lateral_1 = self.lateral_scale_1(vfe_scale_1)  # (216, 248, 64)
        
        # Upsample from smooth_2 to match scale 1
        up_2to1 = self.deconv2to1(smooth_2)  # (108, 124, 64) - needs another 2x upsample
        up_2to1 = F.interpolate(up_2to1, size=(vfe_scale_1.shape[2], vfe_scale_1.shape[3]), 
                               mode='bilinear', align_corners=False)  # (216, 248, 64)
        
        # Concatenate upsampled + lateral
        concat_1 = torch.cat([up_2to1, lateral_1], dim=1)  # (216, 248, 128)
        
        # Smooth and process scale 1
        smooth_1 = self.smooth_scale_1(concat_1)  # (108, 124, 64) - downsampled
        
        # Level 1 (finest): Process VFE scale 0 features
        lateral_0 = self.lateral_scale_0(vfe_scale_0)  # (432, 496, 64)
        
        # Upsample from smooth_1 to match scale 0
        up_1to0 = F.interpolate(smooth_1, size=(vfe_scale_0.shape[2], vfe_scale_0.shape[3]),
                               mode='bilinear', align_corners=False)  # (432, 496, 64)
        
        # Concatenate upsampled + lateral
        concat_0 = torch.cat([up_1to0, lateral_0], dim=1)  # (432, 496, 128)
        
        # Final processing
        # Apply final deconv to get detection features
        # Note: The paper suggests we create feature pyramid outputs
        # For simplicity, we'll use the finest scale as main output
        p1 = concat_0  # Finest scale features
        
        # Optionally downsample p1 for final detection head
        # The paper uses feature maps at different scales for detection
        # Here we'll downsample to a reasonable size for the detection head
        final_features = F.avg_pool2d(p1, kernel_size=2, stride=2)  # (216, 248, 128)
        
        # Store outputs
        data_dict['spatial_features_2d'] = final_features
        data_dict['spatial_features_p1'] = p1              # Finest
        data_dict['spatial_features_p2'] = concat_1        # Medium  
        data_dict['spatial_features_p3'] = concat_2        # Coarsest
        
        return data_dict
