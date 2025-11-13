#!/usr/bin/env python3
"""Test script to isolate the FP exception"""
import sys
import os
os.chdir('/home/ionesctn/Code/openpcdet_project/OpenPCDet/tools')
sys.path.insert(0, '/home/ionesctn/Code/openpcdet_project/OpenPCDet/tools')

print("=" * 80)
print("STEP 1: Import modules")
print("=" * 80)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

print("✓ Imports successful")

print("\n" + "=" * 80)
print("STEP 2: Load config")
print("=" * 80)

cfg_file = 'cfgs/kitti_models/voxel_fpn.yaml'
cfg_from_yaml_file(cfg_file, cfg)
print(f"✓ Config loaded from {cfg_file}")

print("\n" + "=" * 80)
print("STEP 3: Build dataloader (this is where it might crash)")
print("=" * 80)

try:
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
    print(f"✓ Dataloader built, {len(train_set)} samples")
except Exception as e:
    print(f"✗ CRASHED during dataloader build: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 4: Get dataset item directly (bypass dataloader)")
print("=" * 80)

try:
    print("Attempting train_set[0]...")
    data = train_set[0]
    print(f"✓ Got item, keys: {data.keys()}")
    
    # Check voxel data
    for i in range(3):
        if f'voxels_scale_{i}' in data:
            print(f"  Scale {i}: {data[f'voxels_scale_{i}'].shape} voxels, " 
                  f"{data[f'voxel_num_points_scale_{i}'].shape} num_points")
except Exception as e:
    print(f"✗ CRASHED getting dataset item: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 5: Iterate dataloader (this is where training crashes)")
print("=" * 80)

try:
    print("Creating iterator...")
    data_iter = iter(train_loader)
    print("Getting first batch...")
    batch = next(data_iter)
    print(f"✓ Got batch, keys: {batch.keys()}")
except Exception as e:
    print(f"✗ CRASHED during dataloader iteration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL STEPS PASSED - No FP exception!")
print("=" * 80)
