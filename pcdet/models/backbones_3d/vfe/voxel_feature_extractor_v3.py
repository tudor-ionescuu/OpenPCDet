import torch

from .vfe_template import VFETemplate


class VoxelFeatureExtractorV3(VFETemplate):
    """Lightweight voxel feature extractor used by SE-SSD.

    This module performs a simple mean pooling over all points within
    each voxel without any learnable parameters. The behaviour matches the
    description in the SE-SSD paper where the voxel encoder keeps the four
    point-wise features (x, y, z, intensity) intact.
    """

    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """Apply mean pooling to voxelised points.

        Args:
            batch_dict (dict): Dictionary produced by the data loader that
                contains the voxel features under ``'voxels'`` and the number
                of points per voxel under ``'voxel_num_points'``.

        Returns:
            dict: Updated batch dictionary that stores the pooled features in
                ``'voxel_features'``.
        """

        voxel_features = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']

        pooled = voxel_features.sum(dim=1)
        normaliser = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(pooled)
        pooled = pooled / normaliser

        batch_dict['voxel_features'] = pooled.contiguous()
        return batch_dict