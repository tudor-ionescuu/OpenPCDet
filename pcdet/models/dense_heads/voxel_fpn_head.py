"""
Voxel-FPN Detection Head
Based on: Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection from LIDAR Point Clouds
Paper: Sensors 2020, 20, 704

SSD-style detection head with:
- Focal Loss for classification (α=0.25, γ=2.0)
- Smooth L1 for localization
- Direction classification for orientation refinement
- Loss weights: β₀=1.0 (cls), β₁=2.0 (loc), β₂=0.2 (dir)
"""

import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class VoxelFPNHead(AnchorHeadTemplate):
    """
    Detection head for Voxel-FPN
    
    Anchor Configuration (for Cars):
    - Size: (1.6m, 3.9m, 1.5m) - (w, l, h)
    - Z center: -1.0m (fixed height)
    - Rotations: 0, π/2 (two orientations)
    
    Loss Configuration from paper:
    - Classification: Focal Loss with α=0.25, γ=2.0
    - Localization: Smooth L1 with weight 2.0
    - Direction: Cross-Entropy with weight 0.2
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, 
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # Classification head
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        
        # Box regression head (7 parameters: x, y, z, w, l, h, θ)
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # Direction classification head (resolves 180° ambiguity)
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
            
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights following standard practice:
        - Classification bias initialized for low prior probability
        - Box regression initialized with small variance
        """
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        """
        Forward pass through detection head
        
        Args:
            data_dict with:
                spatial_features_2d: (B, C, H, W) - feature map from backbone
                
        Returns:
            data_dict with predictions:
                cls_preds: (B, H, W, num_anchors * num_class)
                box_preds: (B, H, W, num_anchors * 7)
                dir_cls_preds: (B, H, W, num_anchors * num_dir_bins)
        """
        spatial_features_2d = data_dict['spatial_features_2d']

        # Classification prediction
        cls_preds = self.conv_cls(spatial_features_2d)
        
        # Box regression prediction
        box_preds = self.conv_box(spatial_features_2d)

        # Reshape to (B, H, W, C) for easier processing
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        # Direction classification
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # Assign targets during training
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        # Generate predicted boxes
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
