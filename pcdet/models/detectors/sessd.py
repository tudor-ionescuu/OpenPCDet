import copy
import math

import torch
import torch.nn.functional as F

from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils


class SESSD(Detector3DTemplate):
    """Self-Ensembling Single-Stage Detector implementation."""

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.enable_ssl = self.model_cfg.get('ENABLE_SSL', False)
        self.consistency_cfg = self.model_cfg.get('CONSISTENCY', {})
        self.ema_alpha = self.model_cfg.get('EMA_ALPHA', 0.999)

        if self.enable_ssl:
            self.teacher_module_list = self._build_teacher_modules()
        else:
            self.teacher_module_list = None

    def _build_teacher_modules(self):
        teacher_modules = torch.nn.ModuleList()
        for module in self.module_list:
            teacher_module = copy.deepcopy(module)
            teacher_module.eval()
            for param in teacher_module.parameters():
                param.requires_grad_(False)
            teacher_modules.append(teacher_module)
        return teacher_modules

    def _clone_input_dict(self, batch_dict):
        cloned = {}
        for key, val in batch_dict.items():
            if isinstance(val, torch.Tensor):
                cloned[key] = val.clone()
            elif isinstance(val, list):
                cloned[key] = [item.clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item) for item in val]
            else:
                cloned[key] = copy.deepcopy(val)
        return cloned

    def _forward_module_list(self, module_list, batch_dict):
        for module in module_list:
            batch_dict = module(batch_dict)
        return batch_dict

    def _consistency_weight(self):
        base_weight = self.consistency_cfg.get('WEIGHT', 1.0)
        rampup_iters = self.consistency_cfg.get('RAMPUP_ITERS', 1)
        if rampup_iters <= 0:
            return base_weight

        cur_step = float(self.global_step.item())
        progress = max(0.0, min(cur_step / float(rampup_iters), 1.0))
        weight = base_weight * math.exp(-5 * (1 - progress) ** 2)
        return weight

    def _update_teacher_weights(self):
        if not self.enable_ssl:
            return

        alpha = min(1 - 1 / (self.global_step.item() + 1), self.ema_alpha)
        for student_module, teacher_module in zip(self.module_list, self.teacher_module_list):
            for student_param, teacher_param in zip(student_module.parameters(), teacher_module.parameters()):
                teacher_param.data.mul_(alpha).add_(student_param.data * (1 - alpha))

    def compute_consistency_loss(self, student_head, teacher_head):
        student_cls = torch.sigmoid(student_head['cls_preds'])
        teacher_cls = torch.sigmoid(teacher_head['cls_preds'])
        cls_loss = F.smooth_l1_loss(student_cls, teacher_cls, reduction='mean')

        student_boxes = student_head['box_preds']
        teacher_boxes = teacher_head['box_preds']
        box_loss = F.smooth_l1_loss(student_boxes, teacher_boxes, reduction='mean')

        # Transform IoU predictions from [-1, 1] to [0, 1] before computing consistency loss
        student_iou = (student_head['iou_preds'] + 1) * 0.5
        teacher_iou = (teacher_head['iou_preds'] + 1) * 0.5
        iou_loss = F.smooth_l1_loss(student_iou, teacher_iou, reduction='mean')

        total_loss = cls_loss + box_loss + iou_loss
        return total_loss, {
            'cons_cls_loss': cls_loss.item(),
            'cons_box_loss': box_loss.item(),
            'cons_iou_loss': iou_loss.item(),
        }

    def forward(self, batch_dict):
        teacher_input = self._clone_input_dict(batch_dict) if (self.training and self.enable_ssl) else None

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            teacher_ret = None
            if self.enable_ssl:
                teacher_head = self.teacher_module_list[-1]
                teacher_head.forward_ret_dict = {}
                self._forward_module_list(self.teacher_module_list, teacher_input)
                teacher_ret = teacher_head.forward_ret_dict.copy()

            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict, teacher_ret)

            ret_dict = {'loss': loss}

            if self.enable_ssl:
                self._update_teacher_weights()

            return ret_dict, tb_dict, disp_dict

        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict, teacher_ret=None):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        total_loss = loss_rpn

        if teacher_ret is not None:
            student_head = self.dense_head.forward_ret_dict
            consistency_loss, cons_tb = self.compute_consistency_loss(student_head, teacher_ret)
            weight = self._consistency_weight()
            total_loss = total_loss + weight * consistency_loss
            tb_dict.update(cons_tb)
            tb_dict['consistency_loss'] = consistency_loss.item()
            tb_dict['consistency_weight'] = weight

        return total_loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            cls_preds = batch_dict['batch_cls_preds'][batch_mask]
            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)

            iou_preds = batch_dict.get('batch_iou_preds', None)
            if iou_preds is not None:
                iou_scores = iou_preds[batch_mask]
                # Transform from [-1, 1] to [0, 1] (no tanh needed, network already outputs in [-1, 1])
                iou_scores = torch.clamp((iou_scores + 1) * 0.5, min=0.0, max=1.0)
                cls_preds = cls_preds * torch.pow(iou_scores, 4)

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds + 1

            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict