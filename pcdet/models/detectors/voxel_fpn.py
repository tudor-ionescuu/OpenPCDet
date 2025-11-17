"""
Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection
Based on: Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection from LIDAR Point Clouds
Paper: Sensors 2020, 20, 704

Complete one-stage 3D object detector with:
- Multi-scale voxelization (0.16m, 0.32m, 0.64m)
- VFE-FPN encoder (early fusion)
- RPN-FPN decoder (late fusion)
- Focal Loss + Smooth L1 + Direction classification
"""

from .detector3d_template import Detector3DTemplate


class VoxelFPN(Detector3DTemplate):
    """
    Voxel-FPN: Multi-Scale Voxel Feature Aggregation for 3D Object Detection
    
    Architecture:
    1. Multi-scale voxelization: Three voxel sizes (S, 2S, 4S) processed in parallel
    2. VFE-FPN: Early fusion of multi-scale voxel features (bottom-up)
    3. RPN-FPN: Late fusion with feature pyramid network (top-down)
    4. Detection Head: SSD-style anchor-based detection with Focal Loss
    
    Performance on KITTI (Moderate, Car, IoU=0.7):
    - 3D AP: 76.70%
    - BEV AP: 87.21%
    - Speed: 50 FPS
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def build_networks(self):
        """
        Custom build_networks for Voxel-FPN
        VFE-FPN directly outputs 2D spatial features, so we skip BACKBONE_3D and MAP_TO_BEV
        """
        from ..backbones_3d import vfe
        from .. import backbones_2d
        from .. import dense_heads
        
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        
        # Build VFE-FPN (outputs 2D spatial features directly)
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        self.add_module('vfe', vfe_module)
        model_info_dict['module_list'].append(vfe_module)
        model_info_dict['num_bev_features'] = vfe_module.get_output_feature_dim()
        
        # Build RPN-FPN backbone (processes 2D features)
        if self.model_cfg.get('BACKBONE_2D', None) is not None:
            backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
                model_cfg=self.model_cfg.BACKBONE_2D,
                input_channels=model_info_dict['num_bev_features']
            )
            self.add_module('backbone_2d', backbone_2d_module)
            model_info_dict['module_list'].append(backbone_2d_module)
            model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        
        # Build detection head
        if self.model_cfg.get('DENSE_HEAD', None) is not None:
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=True
            )
            self.add_module('dense_head', dense_head_module)
            model_info_dict['module_list'].append(dense_head_module)
        
        return model_info_dict['module_list']

    def forward(self, batch_dict):
        """
        Forward pass through all modules:
        VFE-FPN -> RPN-FPN -> Detection Head
        
        Args:
            batch_dict: Contains multi-scale voxels and coordinates
            
        Returns:
            During training: loss dictionary
            During inference: predicted boxes and scores
        """
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        """
        Compute training loss from detection head
        
        Loss components:
        - Classification: Focal Loss (α=0.25, γ=2.0, weight=1.0)
        - Localization: Smooth L1 (weight=2.0)
        - Direction: Cross-Entropy (weight=0.2)
        """
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        return loss, tb_dict, disp_dict
