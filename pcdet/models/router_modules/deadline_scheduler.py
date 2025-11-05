"""
Deadline-Aware Scheduler for Adaptive Model Selection
Selects optimal model and complexity level based on deadlines and scene complexity
"""

import numpy as np
from collections import deque


class DeadlineScheduler:
    """
    Manages deadline-aware model selection using runtime profiles
    """
    
    def __init__(self, target_fps=10, safety_margin=0.85, min_stay_frames=5):
        """
        Args:
            target_fps: Target inference frequency (Hz)
            safety_margin: Use only this fraction of deadline (0.0-1.0)
            min_stay_frames: Minimum frames before allowing model switch
        """
        self.target_fps = target_fps
        self.target_deadline_ms = 1000.0 / target_fps
        self.safety_margin = safety_margin
        self.min_stay_frames = min_stay_frames
        
        # Runtime history
        self.runtime_history = deque(maxlen=100)
        
        # Current model state
        self.current_model = None
        self.current_level = None
        self.frames_on_current = 0
        
        # Model runtime profiles (ms) - will be updated from real measurements
        # Format: (model_name, level_name): average_runtime_ms
        self.runtime_profiles = {
            ('pointpillar', 'lite'): 12.0,
            ('pointpillar', 'standard'): 16.0,
            ('pointpillar', 'full'): 22.0,
            ('voxelnext', 'lite'): 20.0,
            ('voxelnext', 'standard'): 28.0,
            ('voxelnext', 'full'): 50.0,
            ('pv_rcnn', 'lite'): 55.0,
            ('pv_rcnn', 'standard'): 85.0,
            ('pv_rcnn', 'full'): 125.0,
        }
        
        # Accuracy profiles (mAP for Car on KITTI)
        self.accuracy_profiles = {
            ('pointpillar', 'lite'): 70.0,
            ('pointpillar', 'standard'): 77.3,
            ('pointpillar', 'full'): 80.0,
            ('voxelnext', 'lite'): 75.0,
            ('voxelnext', 'standard'): 80.0,
            ('voxelnext', 'full'): 84.0,
            ('pv_rcnn', 'lite'): 79.0,
            ('pv_rcnn', 'standard'): 83.6,
            ('pv_rcnn', 'full'): 86.0,
        }
    
    def select_model_and_level(self, scene_complexity, deadline_ms=None):
        """
        Select optimal model and level based on complexity and deadline
        
        Args:
            scene_complexity: float, 0.0 (simple) to 1.0 (complex)
            deadline_ms: Optional specific deadline, otherwise uses adaptive deadline
            
        Returns:
            (model_name, level_name): Selected configuration
        """
        # Get current deadline
        if deadline_ms is None:
            deadline_ms = self.get_current_deadline()
        
        # Account for overhead (pre/post processing, data transfer, etc.)
        available_time = deadline_ms - 10.0  # 10ms safety buffer
        
        # Strategy 1: Get preferred models based on scene complexity
        preferred_models = self._get_complexity_preferences(scene_complexity)
        
        # Strategy 2: Filter by deadline constraint
        feasible_models = self._filter_by_deadline(preferred_models, available_time)
        
        if len(feasible_models) == 0:
            # Emergency fallback: fastest model always
            return 'pointpillar', 'lite'
        
        # Strategy 3: Apply temporal smoothing (avoid rapid switching)
        selected = self._apply_temporal_smoothing(feasible_models)
        
        # Update state
        if (selected[0] != self.current_model or selected[1] != self.current_level):
            self.current_model = selected[0]
            self.current_level = selected[1]
            self.frames_on_current = 0
        else:
            self.frames_on_current += 1
        
        return selected
    
    def _get_complexity_preferences(self, scene_complexity):
        """
        Get ordered list of preferred models based on scene complexity
        
        Args:
            scene_complexity: 0.0 (simple) to 1.0 (complex)
            
        Returns:
            List of (model, level) tuples in preference order
        """
        if scene_complexity < 0.3:
            # Simple scene - prioritize speed, lighter models sufficient
            return [
                ('pointpillar', 'lite'),
                ('pointpillar', 'standard'),
                ('voxelnext', 'lite'),
                ('pointpillar', 'full'),
                ('voxelnext', 'standard'),
            ]
        
        elif scene_complexity < 0.7:
            # Medium complexity - balanced approach
            return [
                ('voxelnext', 'standard'),
                ('pointpillar', 'full'),
                ('voxelnext', 'lite'),
                ('pv_rcnn', 'lite'),
                ('pointpillar', 'standard'),
                ('voxelnext', 'full'),
            ]
        
        else:
            # Complex scene - prioritize accuracy
            return [
                ('pv_rcnn', 'standard'),
                ('voxelnext', 'full'),
                ('pv_rcnn', 'lite'),
                ('pv_rcnn', 'full'),
                ('voxelnext', 'standard'),
            ]
    
    def _filter_by_deadline(self, model_list, available_time):
        """
        Filter models that can complete within available time
        
        Args:
            model_list: List of (model, level) tuples
            available_time: Available time in ms
            
        Returns:
            Filtered list that meets deadline
        """
        feasible = []
        
        for model, level in model_list:
            runtime = self.runtime_profiles.get((model, level), float('inf'))
            
            # Add 20% safety margin on runtime estimate
            conservative_runtime = runtime * 1.2
            
            if conservative_runtime <= available_time:
                feasible.append((model, level))
        
        return feasible
    
    def _apply_temporal_smoothing(self, feasible_models):
        """
        Apply temporal smoothing to avoid rapid model switching
        
        Args:
            feasible_models: List of feasible (model, level) options
            
        Returns:
            Selected (model, level) tuple
        """
        if len(feasible_models) == 0:
            return ('pointpillar', 'lite')
        
        # If current model is still feasible and we haven't stayed min frames, keep it
        current_config = (self.current_model, self.current_level)
        
        if (current_config in feasible_models and 
            self.frames_on_current < self.min_stay_frames):
            return current_config
        
        # Otherwise, select best from feasible options
        # "Best" = highest accuracy among feasible models
        best_model = max(
            feasible_models,
            key=lambda x: self.accuracy_profiles.get(x, 0.0)
        )
        
        return best_model
    
    def get_current_deadline(self):
        """
        Compute adaptive deadline based on recent performance
        
        Returns:
            deadline_ms: Current deadline in milliseconds
        """
        if len(self.runtime_history) < 10:
            # Not enough history, use conservative deadline
            return self.target_deadline_ms * self.safety_margin
        
        # Analyze recent performance
        recent_runtimes = list(self.runtime_history)[-20:]
        avg_runtime = np.mean(recent_runtimes)
        max_runtime = np.max(recent_runtimes)
        
        # If consistently under deadline, can be more aggressive
        if avg_runtime < self.target_deadline_ms * 0.6:
            # We have significant headroom
            return self.target_deadline_ms * 0.95
        
        elif avg_runtime < self.target_deadline_ms * 0.8:
            # Comfortable margin
            return self.target_deadline_ms * 0.90
        
        else:
            # Cutting it close, be conservative
            return self.target_deadline_ms * 0.75
    
    def update_runtime(self, actual_runtime_ms):
        """
        Update runtime statistics with actual measured time
        
        Args:
            actual_runtime_ms: Measured inference time in milliseconds
        """
        self.runtime_history.append(actual_runtime_ms)
        
        # Update runtime profile for current model (exponential moving average)
        if self.current_model is not None and self.current_level is not None:
            key = (self.current_model, self.current_level)
            old_estimate = self.runtime_profiles.get(key, actual_runtime_ms)
            
            # EMA with alpha=0.1 (90% old, 10% new)
            new_estimate = 0.9 * old_estimate + 0.1 * actual_runtime_ms
            self.runtime_profiles[key] = new_estimate
    
    def get_statistics(self):
        """
        Get current scheduler statistics
        
        Returns:
            dict with runtime stats and violations
        """
        if len(self.runtime_history) == 0:
            return {
                'avg_runtime_ms': 0.0,
                'max_runtime_ms': 0.0,
                'deadline_violations': 0,
                'violation_rate': 0.0
            }
        
        runtimes = list(self.runtime_history)
        violations = sum(1 for t in runtimes if t > self.target_deadline_ms)
        
        return {
            'avg_runtime_ms': np.mean(runtimes),
            'max_runtime_ms': np.max(runtimes),
            'p95_runtime_ms': np.percentile(runtimes, 95),
            'deadline_violations': violations,
            'violation_rate': violations / len(runtimes),
            'current_model': self.current_model,
            'current_level': self.current_level,
            'frames_on_current': self.frames_on_current
        }
    
    def update_profile(self, model, level, runtime_ms, accuracy=None):
        """
        Manually update runtime/accuracy profile for a model variant
        
        Args:
            model: Model name
            level: Level name
            runtime_ms: Measured runtime
            accuracy: Optional accuracy measurement
        """
        key = (model, level)
        self.runtime_profiles[key] = runtime_ms
        
        if accuracy is not None:
            self.accuracy_profiles[key] = accuracy
