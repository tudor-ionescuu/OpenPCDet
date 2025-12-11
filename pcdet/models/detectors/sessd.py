import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils
from ..losses.weighted_smooth_l1_loss import WeightedSmoothL1Loss


def add_sin_difference(boxes1, boxes2):
    """
    Encode angle difference properly using sin for regression.
    This is critical for proper angle consistency loss.
    
    From original SE-SSD: det3d/models/bbox_heads/mg_head_sessd.py lines 15-44
    """
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


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
            # Initialize consistency loss functions (from original SE-SSD)
            self.loss_size_consistency = nn.MSELoss(reduction='mean')
            self.loss_iou_consistency = WeightedSmoothL1Loss(
                sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0
            )
            self.loss_score_consistency = WeightedSmoothL1Loss(
                sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0
            )
            self.loss_dir_consistency = nn.MSELoss(reduction='mean')
            self.loss_reg = WeightedSmoothL1Loss(
                sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0
            )
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

    def nn_distance(self, box1, box2, iou_thres=0.7):
        """
        Nearest neighbor matching with IoU threshold.
        From original SE-SSD: det3d/models/bbox_heads/mg_head_sessd.py lines 577-607
        
        Args:
            box1: (N, 7) student boxes
            box2: (M, 7) teacher boxes
            iou_thres: IoU threshold for matching (default 0.7)
            
        Returns:
            box_consistency_loss: Consistency loss for matched boxes
            idx1, idx2: Matching indices
            mask1, mask2: Valid match masks
        """
        from ...ops.iou3d_nms import iou3d_nms_utils
        
        ans_iou = iou3d_nms_utils.boxes_iou_bev(box1, box2)
        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)
        mask1, mask2 = iou1 > iou_thres, iou2 > iou_thres
        ans_iou = ans_iou[mask1]
        ans_iou = ans_iou[:, mask2]

        if ans_iou.shape[0] == 0 or ans_iou.shape[1] == 0:
            return None, None, None, None, None

        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)
        val_box1, val_box2 = box1[mask1], box2[mask2]
        aligned_box1, aligned_box2 = val_box1[idx2], val_box2[idx1]

        # Use add_sin_difference for proper angle encoding
        encoded_box_preds, encoded_reg_targets = add_sin_difference(val_box1, aligned_box2)
        loss1 = self.loss_reg(encoded_box_preds.unsqueeze(0), 
                             encoded_reg_targets.unsqueeze(0)).sum(-1) / 7.
        
        box_consistency_loss = loss1.sum() / loss1.shape[0]
        
        return box_consistency_loss, idx1, idx2, mask1, mask2

    def compute_consistency_loss(self, student_head, teacher_head, batch_dict):
        """
        Compute consistency loss following original SE-SSD implementation.
        From: det3d/models/bbox_heads/mg_head_sessd.py lines 618-703
        
        Key differences from simple implementation:
        1. Uses WeightedSmoothL1Loss instead of F.smooth_l1_loss
        2. Includes direction consistency loss (CRITICAL - was missing)
        3. Uses nn_distance with add_sin_difference for proper angle encoding
        4. Separate loss components with explicit 1.0 weights
        """
        from ...ops.iou3d_nms import iou3d_nms_utils
        
        batch_size = student_head['box_preds'].shape[0]
        
        # Get predictions - these are still in anchor offset form
        batch_box_preds_stu = student_head['box_preds'].view(batch_size, -1, 7)
        batch_cls_preds_stu = student_head['cls_preds'].view(batch_size, -1, 1)
        batch_iou_preds_stu = student_head['iou_preds'].view(batch_size, -1, 1)
        batch_dir_preds_stu = student_head.get('dir_cls_preds', None)
        if batch_dir_preds_stu is not None:
            batch_dir_preds_stu = batch_dir_preds_stu.view(batch_size, -1, 2)
        
        batch_box_preds_tea = teacher_head['box_preds'].view(batch_size, -1, 7)
        batch_cls_preds_tea = teacher_head['cls_preds'].view(batch_size, -1, 1)
        batch_iou_preds_tea = teacher_head['iou_preds'].view(batch_size, -1, 1)
        batch_dir_preds_tea = teacher_head.get('dir_cls_preds', None)
        if batch_dir_preds_tea is not None:
            batch_dir_preds_tea = batch_dir_preds_tea.view(batch_size, -1, 2)
        
        # Get anchors from dense_head - same anchors used for all batch elements
        # In original SE-SSD: example["anchors"][0][0] gives [70400, 7]
        # In OpenPCDet: self.dense_head.anchors is a list with shape [z, y, x, num_size, num_rot, 7]
        # We need to flatten it to [num_anchors, 7]
        if isinstance(self.dense_head.anchors, list):
            anchors = torch.cat(self.dense_head.anchors, dim=-3)
        else:
            anchors = self.dense_head.anchors
        anchors = anchors.view(-1, anchors.shape[-1])  # Flatten to [num_anchors, 7], e.g. [70400, 7]
        
        # Post-processing center range - FIXED to match original SE-SSD
        post_center_range = torch.tensor([0, -40.0, -5.0, 70.4, 40.0, 5.0], 
                                         device=batch_box_preds_stu.device)
        
        batch_box_loss = torch.tensor(0., dtype=torch.float32, device=batch_box_preds_stu.device)
        batch_cls_loss = torch.tensor(0., dtype=torch.float32, device=batch_box_preds_stu.device)
        batch_iou_loss = torch.tensor(0., dtype=torch.float32, device=batch_box_preds_stu.device)
        batch_dir_loss = torch.tensor(0., dtype=torch.float32, device=batch_box_preds_stu.device)
        
        num_samples = 0
        
        for idx in range(batch_size):
            # Decode boxes from anchor offsets
            box_preds_stu = self.dense_head.box_coder.decode_torch(
                batch_box_preds_stu[idx], anchors
            )
            box_preds_tea = self.dense_head.box_coder.decode_torch(
                batch_box_preds_tea[idx], anchors
            )
            
            cls_preds_stu = batch_cls_preds_stu[idx]
            cls_preds_tea = batch_cls_preds_tea[idx]
            iou_preds_stu = batch_iou_preds_stu[idx]
            iou_preds_tea = batch_iou_preds_tea[idx]
            
            dir_preds_stu = batch_dir_preds_stu[idx] if batch_dir_preds_stu is not None else None
            dir_preds_tea = batch_dir_preds_tea[idx] if batch_dir_preds_tea is not None else None
            
            # Filter by score threshold 0.3 (original SE-SSD uses this)
            top_scores_keep_stu = torch.sigmoid(cls_preds_stu).squeeze(-1) >= 0.3
            top_scores_keep_tea = torch.sigmoid(cls_preds_tea).squeeze(-1) >= 0.3
            
            # Filter by center range
            mask_stu = (box_preds_stu[:, :3] >= post_center_range[:3]).all(1)
            mask_stu &= (box_preds_stu[:, :3] <= post_center_range[3:]).all(1)
            mask_stu &= top_scores_keep_stu
            
            mask_tea = (box_preds_tea[:, :3] >= post_center_range[:3]).all(1)
            mask_tea &= (box_preds_tea[:, :3] <= post_center_range[3:]).all(1)
            mask_tea &= top_scores_keep_tea
            
            if mask_stu.sum() == 0 or mask_tea.sum() == 0:
                continue
                
            # Get filtered boxes
            top_box_preds_stu = box_preds_stu[mask_stu]
            top_cls_preds_stu = cls_preds_stu[mask_stu]
            top_iou_preds_stu = iou_preds_stu[mask_stu]
            top_dir_preds_stu = dir_preds_stu[mask_stu] if dir_preds_stu is not None else None
            
            top_box_preds_tea = box_preds_tea[mask_tea]
            top_cls_preds_tea = cls_preds_tea[mask_tea]
            top_iou_preds_tea = iou_preds_tea[mask_tea]
            top_dir_preds_tea = dir_preds_tea[mask_tea] if dir_preds_tea is not None else None
            
            # Box consistency loss using nn_distance (with proper angle encoding)
            box_consistency_loss, idx1, idx2, mask1, mask2 = self.nn_distance(
                top_box_preds_stu, top_box_preds_tea, iou_thres=0.7
            )
            
            if box_consistency_loss is None:
                continue
                
            batch_box_loss += box_consistency_loss
            
            # Classification score consistency loss
            aligned_cls_preds_tea = top_cls_preds_tea[mask2][idx1]
            scores_stu = torch.sigmoid(top_cls_preds_stu[mask1])
            scores_tea = torch.sigmoid(aligned_cls_preds_tea)
            score_consistency_loss = self.loss_score_consistency(
                scores_stu.unsqueeze(0), scores_tea.unsqueeze(0)
            ).mean()
            batch_cls_loss += score_consistency_loss
            
            # IoU prediction consistency loss
            aligned_iou_preds_tea = (top_iou_preds_tea[mask2][idx1] + 1) * 0.5
            top_iou_preds_stu_norm = (top_iou_preds_stu[mask1] + 1) * 0.5
            iou_consistency_loss = self.loss_iou_consistency(
                top_iou_preds_stu_norm.unsqueeze(0), 
                aligned_iou_preds_tea.unsqueeze(0)
            ).mean()
            batch_iou_loss += iou_consistency_loss
            
            # Direction consistency loss (CRITICAL - was completely missing before)
            if top_dir_preds_stu is not None and top_dir_preds_tea is not None:
                aligned_dir_preds_tea = top_dir_preds_tea[mask2][idx1]
                aligned_dir_preds_tea = F.softmax(aligned_dir_preds_tea, dim=-1)
                top_dir_preds_stu_softmax = F.softmax(top_dir_preds_stu[mask1], dim=-1)
                dir_consistency_loss = self.loss_dir_consistency(
                    top_dir_preds_stu_softmax, aligned_dir_preds_tea
                )
                batch_dir_loss += dir_consistency_loss
            
            num_samples += 1
        
        if num_samples == 0:
            # No matched pairs found
            total_loss = torch.tensor(0.0, device=batch_box_preds_stu.device)
            return total_loss, {
                'cons_box_loss': 0.0,
                'cons_cls_loss': 0.0,
                'cons_iou_loss': 0.0,
                'cons_dir_loss': 0.0,
            }
        
        # Average over batch
        batch_box_loss /= num_samples
        batch_cls_loss /= num_samples
        batch_iou_loss /= num_samples
        batch_dir_loss /= num_samples
        
        # Apply explicit 1.0 weights to each component (from original SE-SSD line 698)
        total_loss = (1.0 * batch_box_loss + 1.0 * batch_cls_loss + 1.0 * batch_iou_loss)
        
        return total_loss, {
            'cons_box_loss': batch_box_loss.item(),
            'cons_cls_loss': batch_cls_loss.item(),
            'cons_iou_loss': batch_iou_loss.item(),
            'cons_dir_loss': batch_dir_loss.item(),
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
            consistency_loss, cons_tb = self.compute_consistency_loss(student_head, teacher_ret, batch_dict)
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

            # Get max class scores BEFORE IoU rectification
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds + 1

            # Apply score threshold FIRST (like working SE-SSD implementation)
            score_thresh = post_process_cfg.SCORE_THRESH
            if score_thresh > 0.0:
                score_mask = cls_preds >= score_thresh
                if score_mask.sum() == 0:
                    # No detections after threshold - create empty result
                    record_dict = {
                        'pred_boxes': box_preds.new_zeros((0, box_preds.shape[-1])),
                        'pred_scores': cls_preds.new_zeros(0),
                        'pred_labels': label_preds.new_zeros(0, dtype=label_preds.dtype)
                    }
                    pred_dicts.append(record_dict)
                    continue

                # Filter by score threshold
                cls_preds = cls_preds[score_mask]
                box_preds = box_preds[score_mask]
                label_preds = label_preds[score_mask]

                # NOW apply IoU rectification (AFTER score filtering, like working SE-SSD)
                iou_preds = batch_dict.get('batch_iou_preds', None)
                if iou_preds is not None:
                    iou_scores = iou_preds[batch_mask][score_mask]
                    # Transform from [-1, 1] to [0, 1]
                    iou_scores = torch.clamp((iou_scores + 1) * 0.5, min=0.0, max=1.0)
                    # Apply IoU^4 rectification
                    cls_preds = cls_preds * torch.pow(iou_scores.squeeze(-1), 4)

            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=0.0  # Already filtered above
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