# PointPillar Model Degradation - KITTI Dataset

## Overview
Three variants of PointPillar were created for KITTI with progressively reduced computational complexity: **Full** (baseline), **Lite**, and **Ultralight**. Each reduction targets specific network components to lower memory usage and inference time while maintaining reasonable detection performance.

## Degradation Strategy

### 1. **Voxelization (Data Processing Layer)**
Controls how the point cloud is discretized into voxels.

| Model | Voxel Size | Max Points/Voxel | Max Voxels (train/test) |
|-------|------------|------------------|-------------------------|
| **Full** | 0.16 × 0.16 | 32 | 16,000 / 40,000 |
| **Lite** | 0.32 × 0.32 | 20 | 10,000 / 25,000 |
| **Ultralight** | 0.32 × 0.32 | 16 | 6,000 / 15,000 |

**Logic:** Larger voxels = fewer voxels to process. Reducing max voxels limits the computational graph size, especially during training.

---

### 2. **Pillar Feature Encoding (VFE - Voxel Feature Encoder)**
Extracts features from points within each pillar/voxel.

| Model | Feature Dimensions |
|-------|-------------------|
| **Full** | 64 |
| **Lite** | 32 (50% reduction) |
| **Ultralight** | 16 (75% reduction) |

**Logic:** Lower dimensional features = smaller intermediate representations throughout the network. Reduces both memory and computation.

---

### 3. **2D Backbone Network**
Processes the bird's-eye-view (BEV) feature map.

| Model | Layer Depths | Channel Counts | Upsample Filters |
|-------|-------------|----------------|------------------|
| **Full** | [3, 5, 5] | [64, 128, 256] | [128, 128, 128] |
| **Lite** | [2, 3, 3] | [32, 64, 128] | [64, 64, 64] |
| **Ultralight** | [1, 2, 2] | [16, 32, 64] | [32, 32, 32] |

**Logic:** 
- **Layer depth** (e.g., [3, 5, 5]): Number of convolutional blocks per stage. Fewer blocks = faster forward pass.
- **Channel counts**: Halved in Lite, quartered in Ultralight. Directly scales down parameter count and FLOPS.
- **Upsample filters**: Match the reduction to maintain consistency in feature fusion.

---

### 4. **Detection Head & Post-Processing**
Unchanged across all variants.

**Logic:** Anchor configurations, loss functions, and NMS settings remain identical. This isolates the performance impact to architectural changes only, not detection methodology.

---

## What Stays the Same

- **Training hyperparameters** (learning rate, batch size, epochs)
- **Data augmentation** (GT sampling, flipping, rotation, scaling)
- **Anchor generator configurations** (sizes, rotations per class)
- **Loss weights and optimization strategy**

---

## Performance Trade-offs

| Aspect | Full | Lite | Ultralight |
|--------|------|------|------------|
| **Accuracy** | Highest | Medium | Lowest |
| **Inference Speed** | Slowest | Medium | Fastest |
| **Memory Usage** | Highest | Medium | Lowest |
| **Use Case** | High-accuracy research | Balanced | Real-time/edge devices |

---

## Key Insight
The degradation follows a **systematic capacity reduction** pattern:
1. Reduce input granularity (voxel size, count)
2. Reduce feature dimensions (VFE channels)
3. Reduce network depth and width (backbone layers and channels)

This creates models suitable for different deployment scenarios: from server-side inference (Full) to embedded systems (Ultralight).
