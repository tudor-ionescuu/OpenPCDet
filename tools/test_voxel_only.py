#!/usr/bin/env python3
"""Minimal test of multiscale voxelization"""
import numpy as np
import sys

print("Testing spconv voxelization...")

try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    print("✓ Using VoxelGeneratorV2")
except:
    print("✗ Cannot import VoxelGeneratorV2")
    sys.exit(1)

# Create test point cloud
np.random.seed(42)
n_points = 1000
points = np.random.rand(n_points, 4).astype(np.float32)
points[:, 0] = points[:, 0] * 70.4  # X: 0 to 70.4
points[:, 1] = points[:, 1] * 80 - 40  # Y: -40 to 40
points[:, 2] = points[:, 2] * 4 - 3  # Z: -3 to 1
print(f"Created {n_points} test points")

# Test parameters from voxel_fpn config
voxel_sizes = [[0.2, 0.2, 0.4], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]
max_points = [100, 200, 300]
max_voxels = [16000, 12000, 8000]
pcr = [0, -40, -3, 70.4, 40, 1]

for i, (vsize, maxp, maxv) in enumerate(zip(voxel_sizes, max_points, max_voxels)):
    print(f"\nScale {i}: voxel_size={vsize}, max_points={maxp}, max_voxels={maxv}")
    try:
        gen = VoxelGenerator(
            voxel_size=vsize,
            point_cloud_range=pcr,
            max_num_points=maxp,
            max_voxels=maxv
        )
        print(f"  ✓ Generator created")
        
        result = gen.generate(points)
        if isinstance(result, dict):
            voxels = result['voxels']
            coords = result['coordinates']
            num_points = result['num_points_per_voxel']
        else:
            voxels, coords, num_points = result
        
        print(f"  ✓ Voxelization successful: {len(voxels)} voxels")
        print(f"    voxels shape: {voxels.shape}")
        print(f"    coords shape: {coords.shape}")
        print(f"    num_points shape: {num_points.shape}")
        print(f"    num_points min/max: {num_points.min()}/{num_points.max()}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n✓ ALL VOXELIZATION TESTS PASSED")
