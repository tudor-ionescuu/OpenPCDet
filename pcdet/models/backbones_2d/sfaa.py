import torch
import torch.nn as nn


def _build_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, norm_cfg=None):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
    if norm_cfg is not None:
        norm_type = norm_cfg.get('type', 'BN2d').lower()
        eps = norm_cfg.get('eps', 1e-3)
        momentum = norm_cfg.get('momentum', 0.01)
        if norm_type == 'bn2d':
            layers.append(nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum))
        else:
            raise NotImplementedError(f'Unsupported norm type: {norm_type}')
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _build_deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False, norm_cfg=None):
    layers = [nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, output_padding=output_padding, bias=bias
    )]
    if norm_cfg is not None:
        norm_type = norm_cfg.get('type', 'BN2d').lower()
        eps = norm_cfg.get('eps', 1e-3)
        momentum = norm_cfg.get('momentum', 0.01)
        if norm_type == 'bn2d':
            layers.append(nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum))
        else:
            raise NotImplementedError(f'Unsupported norm type: {norm_type}')
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SSFA(nn.Module):
    """Spatial-Semantic Feature Aggregation neck for SE-SSD."""

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        norm_cfg = model_cfg.get('NORM_CFG', {'type': 'BN2d', 'eps': 1e-3, 'momentum': 0.01})

        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, bias=False),
            *_norm_relu(norm_cfg, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 128),
        )

        self.bottom_up_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, stride=2, padding=1, bias=False),
            *_norm_relu(norm_cfg, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 256),
        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            *_norm_relu(norm_cfg, 128),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            *_norm_relu(norm_cfg, 256),
        )

        self.deconv_block_0 = _build_deconv(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, norm_cfg=norm_cfg)
        self.deconv_block_1 = _build_deconv(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, norm_cfg=norm_cfg)

        self.conv_0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 128),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            *_norm_relu(norm_cfg, 128),
        )

        self.weight_0 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            *_norm_only(norm_cfg, 1),
        )
        self.weight_1 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, bias=False),
            *_norm_only(norm_cfg, 1),
        )

        self.num_bev_features = 128

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        x0 = self.bottom_up_block_0(spatial_features)
        x1 = self.bottom_up_block_1(spatial_features)

        x0_trans = self.trans_0(x0)
        x1_trans = self.trans_1(x1)

        up0 = self.deconv_block_0(x1_trans) + x0_trans
        up1 = self.deconv_block_1(x1_trans)

        out0 = self.conv_0(up0)
        out1 = self.conv_1(up1)

        w0 = self.weight_0(out0)
        w1 = self.weight_1(out1)
        weights = torch.cat([w0, w1], dim=1)
        weights = torch.softmax(weights, dim=1)

        fused = out0 * weights[:, 0:1] + out1 * weights[:, 1:2]

        data_dict['spatial_features_2d'] = fused
        data_dict['ssfa_intermediate'] = {
            'local_branch': out0,
            'semantic_branch': out1,
            'attention_weights': weights,
        }
        return data_dict


def _norm_relu(norm_cfg, num_features):
    layers = []
    if norm_cfg is not None:
        norm_type = norm_cfg.get('type', 'BN2d').lower()
        eps = norm_cfg.get('eps', 1e-3)
        momentum = norm_cfg.get('momentum', 0.01)
        if norm_type == 'bn2d':
            layers.append(nn.BatchNorm2d(num_features, eps=eps, momentum=momentum))
        else:
            raise NotImplementedError(f'Unsupported norm type: {norm_type}')
    layers.append(nn.ReLU())
    return tuple(layers)


def _norm_only(norm_cfg, num_features):
    if norm_cfg is None:
        return tuple()

    norm_type = norm_cfg.get('type', 'BN2d').lower()
    eps = norm_cfg.get('eps', 1e-3)
    momentum = norm_cfg.get('momentum', 0.01)
    if norm_type == 'bn2d':
        return (nn.BatchNorm2d(num_features, eps=eps, momentum=momentum),)
    raise NotImplementedError(f'Unsupported norm type: {norm_type}')