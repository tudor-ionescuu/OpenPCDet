#!/usr/bin/env python
"""Debug NaN in forward pass"""
import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def check_tensor(name, tensor):
    """Check tensor for NaN/Inf"""
    if tensor is None:
        print(f"{name}: None")
        return
    
    # Convert numpy to torch if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    # Handle integer tensors
    if tensor.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        print(f"✓ {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, min={min_val}, max={max_val}")
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item() if not has_nan else float('nan')
    max_val = tensor.max().item() if not has_nan else float('nan')
    mean_val = tensor.mean().item() if not has_nan else float('nan')
    
    status = "✓" if not (has_nan or has_inf) else "✗"
    print(f"{status} {name}: shape={tuple(tensor.shape)}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, NaN={has_nan}, Inf={has_inf}")

def main():
    # Load config
    config_file = 'cfgs/kitti_models/voxel_fpn.yaml'
    cfg_from_yaml_file(config_file, cfg)
    
    # Build dataset
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1
    )
    
    # Get one batch
    print("Getting batch...")
    batch = next(iter(train_loader))
    
    print("\n" + "="*80)
    print("INPUT DATA CHECK")
    print("="*80)
    
    # Check input voxels
    for scale_idx in range(3):
        print(f"\nScale {scale_idx}:")
        voxels = batch[f'voxels_scale_{scale_idx}']
        coords = batch[f'voxel_coords_scale_{scale_idx}']
        num_points = batch[f'voxel_num_points_scale_{scale_idx}']
        
        check_tensor(f"  voxels", voxels)
        check_tensor(f"  coords", coords)
        check_tensor(f"  num_points", num_points)
        
        # Check for empty voxels
        empty = (num_points == 0).sum().item()
        print(f"  Empty voxels: {empty}/{len(num_points)}")
    
    print("\n" + "="*80)
    print("MODEL FORWARD PASS")
    print("="*80)
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()
    model.eval()
    
    # Move batch to GPU
    batch = common_utils.move_to_device(batch, torch.device('cuda'))
    
    # Hook to intercept VFE outputs
    def vfe_hook(module, input, output):
        print(f"\nVFE Layer output:")
        if isinstance(output, dict):
            for key in ['voxels_scale_0', 'voxels_scale_1', 'voxels_scale_2']:
                if key in output:
                    check_tensor(f"  {key}", output[key])
        else:
            check_tensor(f"  output", output)
    
    # Register hook
    if hasattr(model, 'vfe'):
        model.vfe.register_forward_hook(vfe_hook)
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            batch_dict = model(batch)
        
        print("\n" + "="*80)
        print("OUTPUT CHECK")
        print("="*80)
        
        if 'spatial_features_2d' in batch_dict:
            check_tensor("spatial_features_2d", batch_dict['spatial_features_2d'])
        
        if 'cls_preds' in batch_dict:
            check_tensor("cls_preds", batch_dict['cls_preds'])
        
        if 'box_preds' in batch_dict:
            check_tensor("box_preds", batch_dict['box_preds'])
            
        print("\n✓ Forward pass completed without crash")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
