# Deadline-Aware Adaptive Model Routing System for OpenPCDet

## Executive Summary

This document outlines a comprehensive design for implementing a **2-dimensional adaptive routing system** for 3D object detection in autonomous driving scenarios. The system dynamically selects both the model architecture (horizontal routing) and its computational complexity level (vertical routing) based on deadline constraints and scene complexity.

---

## 1. Model Selection Analysis

Based on the screenshot and OpenPCDet benchmarks, here's the recommended model hierarchy:

### Tier 1: Fast & Lightweight (Real-time, Lower Accuracy)
- **PointPillar** 
  - Training time: ~1.2 hours
  - Performance: 77.28% Car, 52.29% Ped, 62.68% Cyc
  - Speed: **FASTEST** (~62Hz potential)
  - Architecture: Simple pillar encoding → 2D CNN
  - Best for: Tight deadlines, simple scenes

### Tier 2: Balanced (Medium Speed & Accuracy)
- **VoxelNeXt**
  - Fully sparse 3D detection
  - Good balance between speed and accuracy
  - Speed: ~30-40Hz
  - Best for: Standard operation, moderate deadlines

### Tier 3: Accurate & Slow (High Accuracy, Compute-Intensive)
- **PV-RCNN**
  - Training time: ~5 hours
  - Performance: 83.61% Car, 57.90% Ped, 70.47% Cyc
  - Speed: **SLOWEST** (~10-15Hz)
  - Architecture: Voxel + Point features + RoI refinement
  - Best for: Complex scenes, relaxed deadlines

---

## 2. System Architecture

### 2.1 Core Components

```
Input Frame
     ↓
[Scene Complexity Estimator]
     ↓
[Deadline Manager] ← Current Deadline
     ↓
[2D Router] ← Scene Metrics + Time Budget
     ↓
├── Horizontal: Model Selection (PointPillar/VoxelNeXt/PV-RCNN)
└── Vertical: Complexity Level (Lite/Standard/Full)
     ↓
[Dynamic Model Executor]
     ↓
Detections
```

### 2.2 Directory Structure

```
OpenPCDet/
├── pcdet/
│   ├── models/
│   │   ├── detectors/
│   │   │   ├── adaptive_router.py          # Main router
│   │   │   ├── pointpillar_variants.py     # Multi-level PointPillar
│   │   │   ├── voxelnext_variants.py       # Multi-level VoxelNeXt
│   │   │   └── pv_rcnn_variants.py         # Multi-level PV-RCNN
│   │   └── router_modules/
│   │       ├── __init__.py
│   │       ├── scene_analyzer.py           # Scene complexity estimation
│   │       ├── deadline_scheduler.py       # Deadline-aware scheduling
│   │       ├── model_executor.py           # Dynamic model switching
│   │       └── profiler.py                 # Runtime profiling
│   └── utils/
│       └── router_utils.py
└── tools/
    ├── cfgs/
    │   └── router_configs/
    │       ├── router_pointpillar_multi.yaml
    │       ├── router_voxelnext_multi.yaml
    │       └── router_pv_rcnn_multi.yaml
    └── adaptive_inference.py                # Main inference script
```

---

## 3. Scene Complexity Estimation

### 3.1 Key Metrics

```python
class SceneComplexityEstimator:
    def compute_complexity(self, point_cloud, prev_detections=None):
        metrics = {
            # Static metrics (computed per frame)
            'point_density': self._compute_density(point_cloud),
            'spatial_distribution': self._compute_distribution(point_cloud),
            'num_objects_estimate': self._estimate_object_count(point_cloud),
            
            # Dynamic metrics (temporal)
            'motion_complexity': self._estimate_motion(prev_detections),
            'occlusion_level': self._estimate_occlusions(point_cloud),
            
            # Derived complexity score
            'complexity_score': 0.0  # 0.0 (simple) to 1.0 (complex)
        }
        return metrics
```

### 3.2 Complexity Indicators

1. **Point Density** (Fast, O(1))
   - Total points / volume
   - Indicates scene richness
   - Low density → simpler scene

2. **Spatial Clustering** (Fast, O(n log n))
   - DBSCAN or voxel-based clustering
   - More clusters → more objects → higher complexity

3. **Height Variance** (Fast, O(n))
   - Z-axis distribution
   - High variance → complex terrain/urban

4. **Temporal Consistency** (from previous frame)
   - Number of objects tracked
   - High motion → higher complexity

5. **Distance Distribution** (Fast, O(n))
   - Nearby objects harder to detect
   - Many close objects → higher complexity

### 3.3 Lightweight Implementation

```python
def fast_complexity_score(points):
    """Compute in <1ms for real-time operation"""
    
    # 1. Point density (instant)
    density = len(points) / (70 * 80 * 4)  # normalized to range
    
    # 2. Voxel occupancy (very fast with GPU)
    voxel_size = [0.5, 0.5, 0.5]
    voxels = voxelize_gpu(points, voxel_size)
    occupancy = len(voxels) / expected_voxels
    
    # 3. Z-variance (instant)
    z_std = torch.std(points[:, 2])
    
    # 4. Front region density (critical for AD)
    front_mask = (points[:, 0] > 0) & (points[:, 0] < 30)
    front_density = front_mask.sum() / len(points)
    
    # Weighted combination
    complexity = (
        0.3 * density +
        0.3 * occupancy + 
        0.2 * z_std +
        0.2 * front_density
    )
    
    return float(complexity)  # 0.0 to ~1.0
```

---

## 4. Vertical Routing: Multi-Level Model Variants

### 4.1 Strategy Per Model

#### PointPillar Variants (3 levels)

```yaml
LITE:
  VOXEL_SIZE: [0.24, 0.24, 6]       # Coarser voxels
  MAX_POINTS_PER_VOXEL: 20          # Fewer points
  NUM_FILTERS: [32, 64, 128]        # Smaller channels
  BACKBONE_LAYERS: [2, 3, 3]        # Fewer layers
  NMS_PRE_MAXSIZE: 2048
  EXPECTED_FPS: ~80Hz

STANDARD:
  VOXEL_SIZE: [0.16, 0.16, 4]       # Original
  MAX_POINTS_PER_VOXEL: 32
  NUM_FILTERS: [64, 128, 256]
  BACKBONE_LAYERS: [3, 5, 5]
  NMS_PRE_MAXSIZE: 4096
  EXPECTED_FPS: ~62Hz

FULL:
  VOXEL_SIZE: [0.12, 0.12, 4]       # Finer voxels
  MAX_POINTS_PER_VOXEL: 48
  NUM_FILTERS: [96, 192, 384]       # More channels
  BACKBONE_LAYERS: [4, 6, 6]        # More layers
  NMS_PRE_MAXSIZE: 6144
  EXPECTED_FPS: ~45Hz
```

#### VoxelNeXt Variants (3 levels)

```yaml
LITE:
  VOXEL_SIZE: [0.12, 0.12, 0.2]
  NUM_SPARSE_LAYERS: 2              # Reduce sparse conv blocks
  FEATURE_CHANNELS: 64
  NUM_DECODER_LAYERS: 1
  EXPECTED_FPS: ~50Hz

STANDARD:
  VOXEL_SIZE: [0.1, 0.1, 0.15]      # Original config
  NUM_SPARSE_LAYERS: 4
  FEATURE_CHANNELS: 128
  NUM_DECODER_LAYERS: 2
  EXPECTED_FPS: ~35Hz

FULL:
  VOXEL_SIZE: [0.075, 0.075, 0.1]
  NUM_SPARSE_LAYERS: 6
  FEATURE_CHANNELS: 192
  NUM_DECODER_LAYERS: 3
  EXPECTED_FPS: ~20Hz
```

#### PV-RCNN Variants (3 levels)

```yaml
LITE:
  NUM_KEYPOINTS: 1024               # Reduce keypoints
  SA_LAYERS: 4                      # Fewer set abstraction layers
  ROI_GRID_SIZE: 4                  # Smaller RoI grid
  SKIP_ROI_HEAD: false              # Keep but simplify
  NUM_PROPOSALS: 256
  EXPECTED_FPS: ~18Hz

STANDARD:
  NUM_KEYPOINTS: 2048               # Original
  SA_LAYERS: 6
  ROI_GRID_SIZE: 6
  NUM_PROPOSALS: 512
  EXPECTED_FPS: ~12Hz

FULL:
  NUM_KEYPOINTS: 4096               # More keypoints
  SA_LAYERS: 8
  ROI_GRID_SIZE: 8
  NUM_PROPOSALS: 1024
  EXPECTED_FPS: ~8Hz
```

### 4.2 Dynamic Layer Skipping (Advanced)

For even finer control, implement **early exit strategies**:

```python
class AdaptiveBackbone(nn.Module):
    def forward(self, x, complexity_budget=1.0):
        features = []
        
        # Layer 1 (always execute)
        x = self.layer1(x)
        features.append(x)
        
        # Layer 2 (skip if budget < 0.3)
        if complexity_budget >= 0.3:
            x = self.layer2(x)
            features.append(x)
        
        # Layer 3 (skip if budget < 0.6)
        if complexity_budget >= 0.6:
            x = self.layer3(x)
            features.append(x)
        
        # Layer 4 (only if budget >= 0.9)
        if complexity_budget >= 0.9:
            x = self.layer4(x)
            features.append(x)
        
        return self.fuse_features(features)
```

---

## 5. Deadline-Aware Scheduler

### 5.1 Core Logic

```python
class DeadlineScheduler:
    def __init__(self):
        # Model runtime profiles (ms)
        self.runtime_profiles = {
            ('pointpillar', 'lite'): 12,
            ('pointpillar', 'standard'): 16,
            ('pointpillar', 'full'): 22,
            ('voxelnext', 'lite'): 20,
            ('voxelnext', 'standard'): 28,
            ('voxelnext', 'full'): 50,
            ('pv_rcnn', 'lite'): 55,
            ('pv_rcnn', 'standard'): 85,
            ('pv_rcnn', 'full'): 125,
        }
    
    def select_model_and_level(self, deadline_ms, scene_complexity):
        """
        Args:
            deadline_ms: Available time budget (e.g., 100ms for 10Hz)
            scene_complexity: 0.0 (simple) to 1.0 (complex)
        
        Returns:
            (model_name, level_name): Selected configuration
        """
        
        # Account for overhead (pre/post processing)
        available_time = deadline_ms - 10  # 10ms overhead
        
        # Strategy 1: Complexity-guided selection
        if scene_complexity < 0.3:
            # Simple scene - prioritize speed
            preferred_models = [
                ('pointpillar', 'lite'),
                ('pointpillar', 'standard'),
                ('voxelnext', 'lite'),
            ]
        elif scene_complexity < 0.7:
            # Medium complexity - balanced approach
            preferred_models = [
                ('voxelnext', 'standard'),
                ('pointpillar', 'full'),
                ('pv_rcnn', 'lite'),
            ]
        else:
            # Complex scene - prioritize accuracy
            preferred_models = [
                ('pv_rcnn', 'standard'),
                ('voxelnext', 'full'),
                ('pv_rcnn', 'lite'),
            ]
        
        # Strategy 2: Find best model within deadline
        for model, level in preferred_models:
            runtime = self.runtime_profiles[(model, level)]
            if runtime <= available_time:
                return model, level
        
        # Fallback: fastest configuration
        return 'pointpillar', 'lite'
```

### 5.2 Adaptive Deadline Management

```python
class AdaptiveDeadlineManager:
    def __init__(self, target_fps=10):
        self.target_deadline = 1000.0 / target_fps  # ms
        self.history = deque(maxlen=100)
        self.safety_margin = 0.85  # Use 85% of deadline
    
    def get_current_deadline(self):
        """Adaptive deadline based on recent performance"""
        if len(self.history) < 10:
            return self.target_deadline * self.safety_margin
        
        # If consistently meeting deadline, can be more aggressive
        recent_runtimes = list(self.history)[-20:]
        avg_runtime = np.mean(recent_runtimes)
        
        if avg_runtime < self.target_deadline * 0.7:
            # We have headroom - can use more complex models
            return self.target_deadline * 0.95
        else:
            # Cutting it close - be conservative
            return self.target_deadline * 0.80
    
    def update_runtime(self, actual_runtime_ms):
        self.history.append(actual_runtime_ms)
```

---

## 6. Implementation: Adaptive Router Detector

### 6.1 Main Router Class

```python
# pcdet/models/detectors/adaptive_router.py

import torch
import time
from .detector3d_template import Detector3DTemplate
from ..router_modules.scene_analyzer import SceneComplexityEstimator
from ..router_modules.deadline_scheduler import DeadlineScheduler
from ..router_modules.model_executor import DynamicModelExecutor

class AdaptiveRouter(Detector3DTemplate):
    """
    Deadline-aware adaptive router that selects model and complexity level
    dynamically based on scene complexity and available time budget.
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # Initialize all model variants
        self.model_executor = DynamicModelExecutor(
            model_cfg=model_cfg,
            num_class=num_class,
            dataset=dataset
        )
        
        # Scene analysis and scheduling
        self.scene_estimator = SceneComplexityEstimator(
            point_cloud_range=dataset.point_cloud_range
        )
        self.deadline_scheduler = DeadlineScheduler(
            target_fps=model_cfg.get('TARGET_FPS', 10)
        )
        
        # Runtime statistics
        self.stats = {
            'total_frames': 0,
            'model_usage': {},
            'avg_complexity': 0,
            'deadline_violations': 0
        }
    
    def forward(self, batch_dict):
        """
        Main forward pass with adaptive routing
        """
        start_time = time.time()
        
        # 1. Estimate scene complexity (very fast, <1ms)
        complexity = self.scene_estimator.compute_complexity(
            batch_dict['points'],
            batch_dict.get('prev_detections', None)
        )
        
        # 2. Get current deadline
        deadline_ms = self.deadline_scheduler.get_current_deadline()
        
        # 3. Select model and level
        model_name, level_name = self.deadline_scheduler.select_model_and_level(
            deadline_ms=deadline_ms,
            scene_complexity=complexity['complexity_score']
        )
        
        # 4. Execute selected model
        batch_dict = self.model_executor.execute(
            batch_dict=batch_dict,
            model_name=model_name,
            level_name=level_name
        )
        
        # 5. Post-processing
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            return {'loss': loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            
            # Update runtime statistics
            runtime_ms = (time.time() - start_time) * 1000
            self.deadline_scheduler.update_runtime(runtime_ms)
            
            # Add routing metadata
            pred_dicts[0]['routing_info'] = {
                'model': model_name,
                'level': level_name,
                'complexity': complexity['complexity_score'],
                'runtime_ms': runtime_ms,
                'deadline_ms': deadline_ms
            }
            
            return pred_dicts, recall_dicts
```

### 6.2 Dynamic Model Executor

```python
# pcdet/models/router_modules/model_executor.py

import torch
import torch.nn as nn
from ..detectors import PointPillar, VoxelNeXt, PVRCNN

class DynamicModelExecutor(nn.Module):
    """
    Manages multiple model variants and dynamically switches between them
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        
        # Load all model configurations
        self.models = nn.ModuleDict()
        
        # PointPillar variants
        for level in ['lite', 'standard', 'full']:
            cfg = self._get_config('pointpillar', level, model_cfg)
            model = PointPillar(cfg, num_class, dataset)
            self.models[f'pointpillar_{level}'] = model
        
        # VoxelNeXt variants
        for level in ['lite', 'standard', 'full']:
            cfg = self._get_config('voxelnext', level, model_cfg)
            model = VoxelNeXt(cfg, num_class, dataset)
            self.models[f'voxelnext_{level}'] = model
        
        # PV-RCNN variants
        for level in ['lite', 'standard', 'full']:
            cfg = self._get_config('pv_rcnn', level, model_cfg)
            model = PVRCNN(cfg, num_class, dataset)
            self.models[f'pv_rcnn_{level}'] = model
        
        # Set all to eval mode initially
        for model in self.models.values():
            model.eval()
    
    def execute(self, batch_dict, model_name, level_name):
        """Execute the selected model variant"""
        key = f'{model_name}_{level_name}'
        model = self.models[key]
        
        with torch.no_grad():
            batch_dict = model(batch_dict)
        
        return batch_dict
    
    def _get_config(self, model_name, level, base_cfg):
        """Load specific configuration for model variant"""
        # This would load from YAML configs
        # Implementation depends on your config structure
        pass
    
    def load_pretrained_weights(self, model_name, level, checkpoint_path):
        """Load pretrained weights for specific variant"""
        key = f'{model_name}_{level}'
        if key in self.models:
            checkpoint = torch.load(checkpoint_path)
            self.models[key].load_state_dict(checkpoint['model_state'])
```

---

## 7. Training Strategy

### 7.1 Multi-Level Training Pipeline

```bash
# Train all variants
# Lite versions (fast training)
python tools/train.py --cfg_file tools/cfgs/router_configs/pointpillar_lite.yaml
python tools/train.py --cfg_file tools/cfgs/router_configs/voxelnext_lite.yaml

# Standard versions (original configs)
python tools/train.py --cfg_file tools/cfgs/kitti_models/pointpillar.yaml
python tools/train.py --cfg_file tools/cfgs/kitti_models/voxelnext.yaml
python tools/train.py --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml

# Full versions (enhanced configs)
python tools/train.py --cfg_file tools/cfgs/router_configs/pointpillar_full.yaml
python tools/train.py --cfg_file tools/cfgs/router_configs/pv_rcnn_full.yaml
```

### 7.2 Knowledge Distillation (Optional Enhancement)

Train lite models using heavy models as teachers:

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, targets):
        # Standard classification loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Distillation loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

---

## 8. Evaluation & Profiling

### 8.1 Benchmark Script

```python
# tools/benchmark_router.py

def benchmark_all_variants():
    """Benchmark all model variants on KITTI val set"""
    
    results = {}
    
    for model in ['pointpillar', 'voxelnext', 'pv_rcnn']:
        for level in ['lite', 'standard', 'full']:
            print(f"Benchmarking {model}-{level}...")
            
            metrics = run_inference(model, level)
            
            results[f'{model}_{level}'] = {
                'mAP': metrics['car_3d_ap'],
                'avg_runtime_ms': metrics['avg_time'],
                'fps': 1000.0 / metrics['avg_time'],
                'memory_mb': metrics['peak_memory']
            }
    
    # Save results
    with open('router_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

### 8.2 Runtime Profiling

```python
class RuntimeProfiler:
    """Profile model execution time for different scene complexities"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_model(self, model, level, dataset):
        """Profile model across complexity bins"""
        
        complexity_bins = {
            'simple': (0.0, 0.3),
            'medium': (0.3, 0.7),
            'complex': (0.7, 1.0)
        }
        
        results = {bin_name: [] for bin_name in complexity_bins}
        
        for batch_dict in dataset:
            complexity = estimate_complexity(batch_dict)
            
            # Find appropriate bin
            for bin_name, (low, high) in complexity_bins.items():
                if low <= complexity < high:
                    # Profile execution
                    torch.cuda.synchronize()
                    start = time.time()
                    
                    _ = model(batch_dict)
                    
                    torch.cuda.synchronize()
                    runtime = (time.time() - start) * 1000
                    
                    results[bin_name].append(runtime)
                    break
        
        # Compute statistics
        stats = {}
        for bin_name, runtimes in results.items():
            stats[bin_name] = {
                'mean': np.mean(runtimes),
                'std': np.std(runtimes),
                'p50': np.percentile(runtimes, 50),
                'p95': np.percentile(runtimes, 95)
            }
        
        return stats
```

---

## 9. Real-World Deployment Considerations

### 9.1 Model Loading Strategy

**Option A: Load All Models (High Memory)**
- All 9 variants loaded in GPU memory
- Zero switching latency
- Requires ~6-8GB GPU memory

**Option B: On-Demand Loading (Lower Memory)**
- Load 2-3 most frequently used models
- Swap models when needed (~50-100ms overhead)
- Requires ~2-3GB GPU memory

**Recommended: Hybrid Approach**
```python
class HybridModelManager:
    def __init__(self):
        # Always keep fast models loaded
        self.resident_models = ['pointpillar_lite', 'pointpillar_standard']
        
        # Cache for recently used models
        self.cache = LRUCache(capacity=3)
    
    def get_model(self, model_name, level):
        key = f'{model_name}_{level}'
        
        if key in self.resident_models:
            return self.models[key]
        
        if key in self.cache:
            return self.cache[key]
        
        # Load on demand
        model = self._load_model(key)
        self.cache[key] = model
        return model
```

### 9.2 Temporal Consistency

Avoid rapid model switching:

```python
class TemporalSmoothing:
    def __init__(self, min_stay_frames=5):
        self.current_model = None
        self.frames_on_current = 0
        self.min_stay = min_stay_frames
    
    def should_switch(self, new_model, new_level):
        """Dampen model switching"""
        
        if self.current_model is None:
            return True
        
        # Force stay for minimum frames
        if self.frames_on_current < self.min_stay:
            return False
        
        # Allow switch if significant improvement
        current_cost = get_model_cost(self.current_model)
        new_cost = get_model_cost(new_model, new_level)
        
        return abs(new_cost - current_cost) > 0.2
```

### 9.3 Failure Handling

```python
class RobustRouter:
    def forward(self, batch_dict):
        try:
            # Primary routing logic
            return self._adaptive_forward(batch_dict)
        
        except RuntimeError as e:
            # GPU OOM or other runtime error
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                # Fallback to lightest model
                return self._fallback_forward(batch_dict, 'pointpillar', 'lite')
            raise
        
        except Exception as e:
            # Log and fallback
            logging.error(f"Router error: {e}")
            return self._fallback_forward(batch_dict, 'pointpillar', 'lite')
```

---

## 10. Configuration Example

```yaml
# tools/cfgs/router_configs/adaptive_router.yaml

MODEL:
  NAME: AdaptiveRouter
  
  TARGET_FPS: 10
  SAFETY_MARGIN: 0.85
  
  SCENE_ESTIMATOR:
    METRICS: ['density', 'occupancy', 'z_variance', 'front_density']
    WEIGHTS: [0.3, 0.3, 0.2, 0.2]
  
  SCHEDULER:
    STRATEGY: 'complexity_guided'  # or 'deadline_first', 'accuracy_first'
    MIN_STAY_FRAMES: 5
    ENABLE_TEMPORAL_SMOOTHING: True
  
  MODELS:
    - NAME: pointpillar
      LEVELS: [lite, standard, full]
      WEIGHT_PATHS:
        lite: '../pretrained_models/pointpillar_lite.pth'
        standard: '../pretrained_models/pointpillar_7728.pth'
        full: '../pretrained_models/pointpillar_full.pth'
    
    - NAME: voxelnext
      LEVELS: [lite, standard, full]
      WEIGHT_PATHS:
        lite: '../pretrained_models/voxelnext_lite.pth'
        standard: '../pretrained_models/VoxelNeXt.pth'
        full: '../pretrained_models/voxelnext_full.pth'
    
    - NAME: pv_rcnn
      LEVELS: [lite, standard, full]
      WEIGHT_PATHS:
        lite: '../pretrained_models/pv_rcnn_lite.pth'
        standard: '../pretrained_models/pv_rcnn_8369.pth'
        full: '../pretrained_models/pv_rcnn_full.pth'

DATA_CONFIG:
  _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml

OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 1  # For real-time inference
```

---

## 11. Expected Performance Matrix

| Model-Level | mAP (Car) | Runtime (ms) | FPS | Memory | Use Case |
|-------------|-----------|--------------|-----|---------|----------|
| PP-Lite | ~70% | 12 | 83 | 1.2GB | Emergency/Simple |
| PP-Std | ~77% | 16 | 62 | 1.8GB | Standard Operation |
| PP-Full | ~80% | 22 | 45 | 2.5GB | Complex Scenes |
| VN-Lite | ~75% | 20 | 50 | 2.0GB | Balanced Light |
| VN-Std | ~80% | 28 | 35 | 3.0GB | Balanced Standard |
| VN-Full | ~84% | 50 | 20 | 4.5GB | High Accuracy |
| PV-Lite | ~79% | 55 | 18 | 3.5GB | Complex Light |
| PV-Std | ~84% | 85 | 12 | 5.0GB | Maximum Accuracy |
| PV-Full | ~86% | 125 | 8 | 6.5GB | Offline/Highest |

---

## 12. Next Steps & Recommendations

### Phase 1: Prototype (2-3 weeks)
1. Implement `SceneComplexityEstimator` with basic metrics
2. Create lite variants of PointPillar and VoxelNeXt
3. Implement basic `DeadlineScheduler`
4. Benchmark all variants on KITTI

### Phase 2: Core Router (3-4 weeks)
1. Implement full `AdaptiveRouter` class
2. Integrate all 9 model variants
3. Add temporal smoothing and robustness
4. Test on real-world driving sequences

### Phase 3: Optimization (2-3 weeks)
1. Profile and optimize scene estimator (<1ms)
2. Implement model caching strategies
3. Fine-tune routing policies based on real data
4. Add knowledge distillation for lite models

### Phase 4: Validation (2 weeks)
1. Extensive testing on KITTI/Waymo
2. Real-time performance validation
3. Compare against fixed-model baselines
4. Document performance trade-offs

---

## 13. Alternative Approaches to Consider

### A. Neural Architecture Search (NAS) Based Routing
Instead of manual variants, use NAS to discover optimal sub-networks:
- **Pros**: Potentially better accuracy/speed trade-offs
- **Cons**: Requires significant training resources

### B. Attention-Based Dynamic Execution
Use attention mechanisms to decide which layers to execute:
- **Pros**: Very fine-grained control
- **Cons**: Added overhead, complex training

### C. Ensemble with Confidence Gating
Run fast model first, escalate to heavy model only if confidence is low:
- **Pros**: Adaptive per-object granularity
- **Cons**: Variable latency, harder to guarantee deadlines

### D. Slimmable Networks
Train a single super-network that can be sliced to different widths:
- **Pros**: Single model, flexible width selection
- **Cons**: Requires custom training, may sacrifice individual variant performance

---

## 14. Conclusion

The proposed 2D routing system provides:

✅ **Horizontal Routing**: Choose between 3 architectures (PointPillar, VoxelNeXt, PV-RCNN)
✅ **Vertical Routing**: 3 complexity levels per architecture (Lite, Standard, Full)
✅ **Deadline Awareness**: Dynamic selection based on time budget
✅ **Scene Adaptivity**: Complexity-guided model selection
✅ **Real-time Feasibility**: Fast scene estimation (<1ms overhead)
✅ **Graceful Degradation**: Always meets deadline with appropriate model

This system enables autonomous vehicles to intelligently trade off accuracy and speed based on situational demands, ensuring both safety (meeting deadlines) and performance (maximizing accuracy when possible).

---

## References & Resources

1. PointPillars: Fast Encoders for Object Detection from Point Clouds
2. VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection
3. PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
4. Anytime Neural Networks via Joint Optimization
5. Dynamic Neural Networks: A Survey
6. OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds

