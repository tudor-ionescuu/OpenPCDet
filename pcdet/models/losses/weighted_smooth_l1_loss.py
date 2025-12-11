import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted Smooth L1 Loss from original SE-SSD implementation.
    
    The smooth L1_loss is defined elementwise as:
    - 0.5 * (sigma * x)^2 if |x| < 1/sigma^2
    - |x| - 0.5/sigma^2 otherwise
    
    where x is the difference between predictions and target.
    """

    def __init__(self, sigma=3.0, reduction="mean", code_weights=None, 
                 codewise=True, loss_weight=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self._sigma = sigma               # 3
        self._code_weights = code_weights
        self._codewise = codewise         # True
        self._reduction = reduction       # mean
        self._loss_weight = loss_weight   # 1.0

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """
        Compute loss function.
        
        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors, code_size] 
                              representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors, code_size] 
                          representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        
        if self._code_weights is not None:
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff

        # Smooth L1: 0.5*(3x)^2 if |x|<1/9 else |x|-0.5/9
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
        
        # if abs_diff_lt_1 = 1 (abs_diff < 1/9), loss = 0.5 * 9 * (abs_diff)^2
        # else if abs_diff_lt_1 = 0 (abs_diff >= 1/9), loss = abs_diff - 0.5/9
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + \
               (abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)

        if self._codewise:    # True
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)
            if weights is not None:
                anchorwise_smooth_l1norm *= weights

        return anchorwise_smooth_l1norm
