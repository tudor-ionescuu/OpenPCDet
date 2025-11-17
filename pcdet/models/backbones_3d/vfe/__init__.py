from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFESimple2D
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .voxelnet_vfe import VoxelFeatureEncoderV2
from .voxel_fpn_vfe import VoxelFPNVFE

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'VoxelFeatureEncoderV2': VoxelFeatureEncoderV2,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynamicPillarVFESimple2D': DynamicPillarVFESimple2D,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'VoxelFPNVFE': VoxelFPNVFE,
}
