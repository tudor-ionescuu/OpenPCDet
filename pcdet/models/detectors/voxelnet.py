from .detector3d_template import Detector3DTemplate


class VoxelNet(Detector3DTemplate):
    """
    VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
    Paper: https://arxiv.org/abs/1711.06396
    
    Architecture:
    1. VFE (Voxel Feature Encoding): Extracts features from point clouds within voxels
    2. 3D CNN Middle Layers: Processes voxel features with sparse 3D convolutions  
    3. RPN (Region Proposal Network): Generates 3D bounding box proposals
    """
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        Forward pass through all modules:
        VFE -> 3D Backbone -> BEV -> 2D Backbone -> Dense Head
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
        Compute training loss from RPN head
        """
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        return loss, tb_dict, disp_dict
