#!/usr/bin/env python3
"""
Debug script to identify VoxelFPN training issues
"""

import sys
import numpy as np
import torch

# Calculate voxel grid dimensions
def check_voxel_config():
    print("=" * 80)
    print("Checking VoxelFPN Configuration")
    print("=" * 80)
    
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    voxel_sizes = [[0.16, 0.16, 4.0], [0.32, 0.32, 4.0], [0.64, 0.64, 4.0]]
    
    print(f"\nPoint Cloud Range: {point_cloud_range}")
    print(f"Range dimensions: X=[{point_cloud_range[0]}, {point_cloud_range[3]}], "
          f"Y=[{point_cloud_range[1]}, {point_cloud_range[4]}], "
          f"Z=[{point_cloud_range[2]}, {point_cloud_range[5]}]")
    
    for i, voxel_size in enumerate(voxel_sizes):
        print(f"\n--- Scale {i+1}: Voxel Size {voxel_size} ---")
        
        grid_size_x = (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]
        grid_size_y = (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]
        grid_size_z = (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
        
        print(f"Grid dimensions: X={grid_size_x:.2f}, Y={grid_size_y:.2f}, Z={grid_size_z:.2f}")
        print(f"Grid dimensions (int): X={int(grid_size_x)}, Y={int(grid_size_y)}, Z={int(grid_size_z)}")
        print(f"Total voxels: {int(grid_size_x) * int(grid_size_y) * int(grid_size_z)}")
        
        # Check for potential issues
        if grid_size_x <= 0 or grid_size_y <= 0 or grid_size_z <= 0:
            print(f"❌ ERROR: Invalid grid dimensions (zero or negative)!")
            return False
            
        if grid_size_z < 1:
            print(f"⚠️  WARNING: Z dimension less than 1 voxel!")
    
    print("\n" + "=" * 80)
    print("✓ Voxel configuration looks valid")
    print("=" * 80)
    return True

def test_data_loading():
    print("\n" + "=" * 80)
    print("Testing Data Loading")
    print("=" * 80)
    
    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        import os
        
        cfg_file = 'cfgs/kitti_models/voxel_fpn.yaml'
        if not os.path.exists(cfg_file):
            print(f"❌ Config file not found: {cfg_file}")
            return False
            
        cfg_from_yaml_file(cfg_file, cfg)
        
        print(f"\nLoading dataset from: {cfg.DATA_CONFIG.DATA_PATH}")
        
        # Build train dataloader
        train_set, train_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=None,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=1
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(train_set)}")
        
        # Try to load one batch
        print("\nTrying to load first batch...")
        data_iter = iter(train_loader)
        batch = next(data_iter)
        
        print(f"✓ Batch loaded successfully")
        print(f"  Batch keys: {batch.keys()}")
        
        # Check voxel data
        for i in range(3):
            if f'voxels_scale_{i}' in batch:
                voxels = batch[f'voxels_scale_{i}']
                coords = batch[f'voxel_coords_scale_{i}']
                num_points = batch[f'voxel_num_points_scale_{i}']
                
                print(f"\n  Scale {i}:")
                print(f"    Voxels shape: {voxels.shape}")
                print(f"    Coords shape: {coords.shape}")
                print(f"    Num points shape: {num_points.shape}")
                print(f"    Num voxels: {voxels.shape[0]}")
                
                if voxels.shape[0] == 0:
                    print(f"    ⚠️  WARNING: No voxels at scale {i}!")
                    
                # Check for NaN or Inf
                if torch.isnan(voxels).any():
                    print(f"    ❌ ERROR: NaN detected in voxels!")
                    return False
                if torch.isinf(voxels).any():
                    print(f"    ❌ ERROR: Inf detected in voxels!")
                    return False
        
        print("\n✓ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward():
    print("\n" + "=" * 80)
    print("Testing Model Forward Pass")
    print("=" * 80)
    
    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.datasets import build_dataloader
        
        cfg_file = 'cfgs/kitti_models/voxel_fpn.yaml'
        cfg_from_yaml_file(cfg_file, cfg)
        
        # Build model
        print("\nBuilding model...")
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
        model.cuda()
        model.eval()
        print(f"✓ Model built successfully")
        
        # Load one batch
        print("\nLoading batch...")
        train_set, train_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=None,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=1
        )
        
        data_iter = iter(train_loader)
        batch = next(data_iter)
        
        # Move batch to GPU
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.cuda()
        
        print(f"✓ Batch loaded and moved to GPU")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch)
        
        print(f"✓ Forward pass completed successfully!")
        print(f"  Prediction keys: {pred_dicts[0].keys()}")
        print(f"  Return dict keys: {ret_dict.keys()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import os
    os.chdir('/home/ionesctn/Code/openpcdet_project/OpenPCDet/tools')
    
    # Run checks
    config_ok = check_voxel_config()
    
    if config_ok:
        data_ok = test_data_loading()
        
        if data_ok:
            model_ok = test_model_forward()
            
            if model_ok:
                print("\n" + "=" * 80)
                print("✓ ALL TESTS PASSED - Training should work!")
                print("=" * 80)
                sys.exit(0)
    
    print("\n" + "=" * 80)
    print("❌ TESTS FAILED - Please fix the issues above")
    print("=" * 80)
    sys.exit(1)
