from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .voxelnext_head_iou import VoxelNeXtHeadIoU
from .voxelnext_head_maxpool import VoxelNeXtHeadMaxPool
from .voxelnext_head_onnx import VoxelNeXtHeadONNX
from .transfusion_head import TransFusionHead
from .IASSD_head import IASSD_Head
from ..roi_heads.ted_head import TEDSHead, TEDMHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'VoxelNeXtHeadIoU': VoxelNeXtHeadIoU,
    'VoxelNeXtHeadMaxPool': VoxelNeXtHeadMaxPool,
    'VoxelNeXtHeadONNX': VoxelNeXtHeadONNX,
    'TransFusionHead': TransFusionHead,
    'IASSD_Head': IASSD_Head,
    'TEDSHead': TEDSHead,
    'TEDMHead': TEDMHead,
}
