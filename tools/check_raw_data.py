#!/usr/bin/env python
"""Check raw point cloud data before voxelization"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

def main():
    # Load config
    config_file = 'cfgs/kitti_models/voxel_fpn.yaml'
    cfg_from_yaml_file(config_file, cfg)
    
    # Build dataset WITHOUT data augmentation to check raw data
    train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        training=False,  # No augmentation
        merge_all_iters_to_one_epoch=False,
        total_epochs=1
    )
    
    # Get raw data
    print("Loading raw data sample 0...")
    info = train_set.kitti_infos[0]
    print(f"Frame ID: {info['point_cloud']['lidar_idx']}")
    
    # Load points directly
    points = train_set.get_lidar(info['point_cloud']['lidar_idx'])
    print(f"\nRaw points shape: {points.shape}")
    print(f"Points dtype: {points.dtype}")
    print(f"Points min: {points.min(axis=0)}")
    print(f"Points max: {points.max(axis=0)}")
    print(f"Points mean: {points.mean(axis=0)}")
    print(f"Has NaN: {np.isnan(points).any()}")
    print(f"Has Inf: {np.isinf(points).any()}")
    
    # Now test voxelization manually
    print("\n" + "="*80)
    print("Testing voxelization manually")
    print("="*80)
    
    from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
    
    # Filter points to point cloud range
    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    mask = np.all(points[:, 0:3] >= point_cloud_range[0:3], axis=1) & \
           np.all(points[:, 0:3] <= point_cloud_range[3:6], axis=1)
    points = points[mask]
    
    print(f"\nFiltered points shape: {points.shape}")
    print(f"Points dtype: {points.dtype}")
    print(f"Has NaN after filtering: {np.isnan(points).any()}")
    
    # Test voxelization at scale 0
    voxel_size = [0.16, 0.16, 4.0]
    voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        num_point_features=points.shape[-1],
        max_num_points_per_voxel=100,
        max_num_voxels=16000
    )
    
    print(f"\nVoxelizing with size {voxel_size}...")
    voxels, coords, num_points = voxel_generator.generate(points)
    
    print(f"Voxels shape: {voxels.shape}")
    print(f"Voxels dtype: {voxels.dtype}")
    print(f"Coords shape: {coords.shape}")
    print(f"Coords dtype: {coords.dtype}")
    print(f"Num points shape: {num_points.shape}")
    print(f"Num points dtype: {num_points.dtype}")
    
    print(f"\nVoxels min: {voxels.min()}")
    print(f"Voxels max: {voxels.max()}")
    print(f"Voxels mean: {voxels.mean()}")
    print(f"Voxels has NaN: {np.isnan(voxels).any()}")
    print(f"Voxels has Inf: {np.isinf(voxels).any()}")
    
    print(f"\nCoords min: {coords.min()}")
    print(f"Coords max: {coords.max()}")
    print(f"Coords sample: {coords[:5]}")
    
    print(f"\nNum points min: {num_points.min()}")
    print(f"Num points max: {num_points.max()}")
    print(f"Num points sample: {num_points[:10]}")
    print(f"Empty voxels: {(num_points == 0).sum()}/{len(num_points)}")

if __name__ == '__main__':
    main()
