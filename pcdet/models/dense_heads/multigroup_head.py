import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import loss_utils


class MultiGroupHead(AnchorHeadTemplate):
    """Multi-task anchor head with IoU prediction branch for SE-SSD."""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training,
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1)
        self.conv_box = nn.Conv2d(input_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1)
        self.conv_iou = nn.Conv2d(input_channels, self.num_anchors_per_location, kernel_size=1)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', True):
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
            )
        else:
            self.conv_dir_cls = None

        self.iou_loss_func = loss_utils.WeightedSmoothL1Loss(beta=1.0 / 9.0, code_weights=[1.0])
        if isinstance(self.reg_loss_func, loss_utils.ODIoULoss3D):
            self.reg_loss_func = loss_utils.ODIoULoss3D()

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv_iou.weight, mean=0, std=0.001)
        if self.conv_iou.bias is not None:
            nn.init.constant_(self.conv_iou.bias, 0)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        iou_preds = self.conv_iou(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] = iou_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds,
                box_preds=box_preds,
                dir_cls_preds=dir_cls_preds,
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            batch_iou_preds = iou_preds.view(batch_cls_preds.shape[0], -1, 1)
            data_dict['batch_iou_preds'] = batch_iou_preds

        return data_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        box_preds = box_preds.view(batch_size, -1, self.box_coder.code_size)
        box_reg_targets = box_reg_targets.view(batch_size, -1, self.box_coder.code_size)

        if isinstance(self.reg_loss_func, loss_utils.ODIoULoss3D):
            pred_boxes = self.box_coder.decode_torch(box_preds, anchors)
            target_boxes = self.box_coder.decode_torch(box_reg_targets, anchors)
            loc_loss_src = self.reg_loss_func(pred_boxes, target_boxes, weights=reg_weights)
        else:
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)

        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {'rpn_loss_loc': loc_loss.item()}

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS,
            )
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_iou_prediction_loss(self):
        iou_preds = self.forward_ret_dict['iou_preds']
        box_preds = self.forward_ret_dict['box_preds']
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(iou_preds.shape[0])

        positives = box_cls_labels > 0
        if positives.sum() == 0:
            return iou_preds.new_zeros(1), {'rpn_loss_iou': 0.0}

        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        box_preds = box_preds.view(batch_size, -1, self.box_coder.code_size)
        box_reg_targets = box_reg_targets.view(batch_size, -1, self.box_coder.code_size)

        pred_boxes = self.box_coder.decode_torch(box_preds, anchors)
        target_boxes = self.box_coder.decode_torch(box_reg_targets, anchors)

        iou_preds = iou_preds.view(batch_size, -1, 1)

        # Compute IoU targets only for positive anchors
        iou_targets_full = iou_preds.new_zeros(iou_preds.shape)
        if positives.any():
            pred_boxes_pos = pred_boxes[positives]
            target_boxes_pos = target_boxes[positives]
            iou_targets = iou3d_nms_utils.boxes_aligned_iou3d_gpu(
                pred_boxes_pos.contiguous(), target_boxes_pos.contiguous()).squeeze(-1).clamp(min=0.0, max=1.0)
            iou_targets = 2 * iou_targets - 1
            mask = positives.unsqueeze(-1)
            iou_targets_full[mask] = iou_targets.view(-1)

        # Apply loss only to positive anchors via weights (negatives have weight 0)
        loss_src = self.iou_loss_func(iou_preds, iou_targets_full, weights=reg_weights)
        loss = loss_src.sum() / batch_size

        loss = loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('iou_weight', 1.0)
        tb_dict = {'rpn_loss_iou': loss.item()}
        return loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        iou_loss, tb_dict_iou = self.get_iou_prediction_loss()
        tb_dict.update(tb_dict_box)
        tb_dict.update(tb_dict_iou)

        rpn_loss = cls_loss + box_loss + iou_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        batch_cls_preds, batch_box_preds = super().generate_predicted_boxes(batch_size, cls_preds, box_preds, dir_cls_preds)
        return batch_cls_preds, batch_box_preds