import torch.nn as nn

from ...utils.spconv_utils import spconv


def _build_norm(norm_cfg, num_features):
    if norm_cfg is None:
        return nn.Identity()

    norm_type = norm_cfg.get('type', 'BN1d').lower()
    eps = norm_cfg.get('eps', 1e-3)
    momentum = norm_cfg.get('momentum', 0.01)

    if norm_type == 'bn1d':
        return nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    raise NotImplementedError(f'Unsupported norm type: {norm_type}')


class SpMiddleFHD(nn.Module):
    """Sparse 3D backbone used in SE-SSD.

    The architecture follows the original implementation that downsamples
    the sparse tensor four times with an overall down-sampling ratio of 8.
    """

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_cfg = model_cfg.get('NORM_CFG', None)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm0'),
            *_maybe_norm_relu(norm_cfg, 16),
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='subm0'),
            *_maybe_norm_relu(norm_cfg, 16),
        )

        self.conv1 = spconv.SparseSequential(
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            *_maybe_norm_relu(norm_cfg, 32),
            spconv.SubMConv3d(32, 32, 3, bias=False, indice_key='subm1'),
            *_maybe_norm_relu(norm_cfg, 32),
            spconv.SubMConv3d(32, 32, 3, bias=False, indice_key='subm1'),
            *_maybe_norm_relu(norm_cfg, 32),
        )

        self.conv2 = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm2'),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm2'),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm2'),
            *_maybe_norm_relu(norm_cfg, 64),
        )

        self.conv3 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, 3, stride=2, padding=(0, 1, 1), bias=False),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm3'),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm3'),
            *_maybe_norm_relu(norm_cfg, 64),
            spconv.SubMConv3d(64, 64, 3, bias=False, indice_key='subm3'),
            *_maybe_norm_relu(norm_cfg, 64),
        )

        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), bias=False),
            *_maybe_norm_relu(norm_cfg, 64),
        )

        self.num_point_features = 128
        # After all downsampling, conv4 outputs 64 channels with 2 depth slices -> 128 BEV features
        self.num_bev_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        dense = x_conv4.dense()
        batch_size, channels, depth, height, width = dense.shape
        spatial_features = dense.view(batch_size, channels * depth, height, width)

        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8,
            'spatial_features': spatial_features,
        })

        batch_dict['multi_scale_3d_features'] = {
            'x_conv1': x_conv1,
            'x_conv2': x_conv2,
            'x_conv3': x_conv3,
            'x_conv4': x_conv4,
        }
        batch_dict['multi_scale_3d_strides'] = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
        }

        return batch_dict


def _maybe_norm_relu(norm_cfg, num_features):
    layer = _build_norm(norm_cfg, num_features)
    if isinstance(layer, nn.Identity):
        return (nn.ReLU(),)
    return (layer, nn.ReLU())