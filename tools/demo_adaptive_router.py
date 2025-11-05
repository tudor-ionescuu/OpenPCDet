"""
Example usage of the Adaptive Router System
Demonstrates scene complexity estimation and model selection
"""

import sys
import numpy as np
import torch

# Add OpenPCDet to path
sys.path.append('/home/tudor_ionescu/openpcdet_project/OpenPCDet')

from pcdet.models.router_modules.scene_analyzer import SceneComplexityEstimator
from pcdet.models.router_modules.deadline_scheduler import DeadlineScheduler


def simulate_driving_scenario():
    """
    Simulate a driving scenario with varying scene complexity
    """
    
    # Initialize components
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # KITTI range
    estimator = SceneComplexityEstimator(point_cloud_range)
    scheduler = DeadlineScheduler(target_fps=10, safety_margin=0.85, min_stay_frames=5)
    
    print("=" * 80)
    print("Adaptive Router Demo - Simulated Driving Scenario")
    print("=" * 80)
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Empty Highway',
            'num_points': 15000,
            'description': 'Simple scene, few objects'
        },
        {
            'name': 'Urban Street',
            'num_points': 45000,
            'description': 'Medium complexity, moderate traffic'
        },
        {
            'name': 'Complex Intersection',
            'num_points': 80000,
            'description': 'High complexity, many objects'
        },
        {
            'name': 'Parking Lot',
            'num_points': 60000,
            'description': 'High density, many static objects'
        },
        {
            'name': 'Rural Road',
            'num_points': 20000,
            'description': 'Low complexity, open space'
        }
    ]
    
    print("\nSimulating 5 different driving scenarios:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*80}")
        
        # Generate synthetic point cloud
        points = generate_synthetic_pointcloud(
            num_points=scenario['num_points'],
            pc_range=point_cloud_range
        )
        
        # Estimate scene complexity
        complexity_metrics = estimator.compute_complexity(points)
        
        print(f"\n📊 Scene Complexity Analysis:")
        print(f"  • Point Density:     {complexity_metrics['density']:.3f}")
        print(f"  • Voxel Occupancy:   {complexity_metrics['occupancy']:.3f}")
        print(f"  • Height Variance:   {complexity_metrics['z_variance']:.3f}")
        print(f"  • Front Density:     {complexity_metrics['front_density']:.3f}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  ⭐ Overall Complexity: {complexity_metrics['complexity_score']:.3f}")
        
        # Select model based on deadline and complexity
        model, level = scheduler.select_model_and_level(
            scene_complexity=complexity_metrics['complexity_score']
        )
        
        # Get expected runtime and accuracy
        runtime = scheduler.runtime_profiles[(model, level)]
        accuracy = scheduler.accuracy_profiles[(model, level)]
        deadline = scheduler.get_current_deadline()
        
        print(f"\n🎯 Routing Decision:")
        print(f"  • Selected Model:    {model.upper()}")
        print(f"  • Complexity Level:  {level.upper()}")
        print(f"  • Expected Runtime:  {runtime:.1f} ms")
        print(f"  • Expected Accuracy: {accuracy:.1f}% mAP")
        print(f"  • Deadline:          {deadline:.1f} ms")
        print(f"  • Margin:            {deadline - runtime:.1f} ms")
        
        # Simulate execution with some variance
        actual_runtime = runtime * np.random.uniform(0.9, 1.1)
        scheduler.update_runtime(actual_runtime)
        
        # Show status indicator
        if actual_runtime < deadline:
            status = "✅ MEET DEADLINE"
            color = "green"
        else:
            status = "⚠️  DEADLINE MISS"
            color = "red"
        
        print(f"\n{status} (actual: {actual_runtime:.1f} ms)")
        
        # Print reasoning
        if complexity_metrics['complexity_score'] < 0.3:
            print(f"\n💡 Reasoning: Simple scene → Fast model (PointPillar) is sufficient")
        elif complexity_metrics['complexity_score'] < 0.7:
            print(f"\n💡 Reasoning: Medium complexity → Balanced model (VoxelNeXt)")
        else:
            print(f"\n💡 Reasoning: Complex scene → Accurate model (PV-RCNN)")
    
    # Summary statistics
    print(f"\n\n{'='*80}")
    print("📈 Session Statistics")
    print(f"{'='*80}")
    
    stats = scheduler.get_statistics()
    print(f"Average Runtime:       {stats['avg_runtime_ms']:.2f} ms")
    print(f"Max Runtime:           {stats['max_runtime_ms']:.2f} ms")
    print(f"P95 Runtime:           {stats['p95_runtime_ms']:.2f} ms")
    print(f"Deadline Violations:   {stats['deadline_violations']} / {len(scheduler.runtime_history)}")
    print(f"Violation Rate:        {stats['violation_rate']*100:.1f}%")
    print(f"Current Model:         {stats['current_model']} - {stats['current_level']}")


def generate_synthetic_pointcloud(num_points, pc_range):
    """
    Generate synthetic point cloud for testing
    
    Args:
        num_points: Number of points to generate
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        (N, 4) tensor: [x, y, z, intensity]
    """
    # Generate random points within range
    x = np.random.uniform(pc_range[0], pc_range[3], num_points)
    y = np.random.uniform(pc_range[1], pc_range[4], num_points)
    z = np.random.uniform(pc_range[2], pc_range[5], num_points)
    intensity = np.random.uniform(0, 1, num_points)
    
    # Add more points in front region (realistic for AD)
    front_points = int(num_points * 0.4)
    x[:front_points] = np.random.uniform(0, 30, front_points)
    
    # Stack into point cloud
    points = np.column_stack([x, y, z, intensity])
    
    return torch.from_numpy(points).float()


def test_complexity_estimator_speed():
    """
    Benchmark scene complexity estimator speed
    """
    print("\n" + "="*80)
    print("⏱️  Performance Benchmark: Scene Complexity Estimator")
    print("="*80)
    
    import time
    
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    estimator = SceneComplexityEstimator(point_cloud_range)
    
    point_counts = [10000, 30000, 50000, 80000, 120000]
    
    print("\nMeasuring computation time for different point cloud sizes:\n")
    
    for num_points in point_counts:
        points = generate_synthetic_pointcloud(num_points, point_cloud_range)
        
        # Warmup
        for _ in range(10):
            _ = estimator.compute_complexity(points)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            _ = estimator.compute_complexity(points)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        status = "✅ Real-time capable" if avg_time < 1.0 else "⚠️  Slower than target"
        
        print(f"Points: {num_points:6d}  →  "
              f"Time: {avg_time:.3f} ± {std_time:.3f} ms  {status}")
    
    print("\nTarget: <1ms for real-time operation")


if __name__ == '__main__':
    print("\n🚗 Deadline-Aware Adaptive Routing System Demo\n")
    
    # Run main simulation
    simulate_driving_scenario()
    
    # Run performance benchmark
    test_complexity_estimator_speed()
    
    print("\n" + "="*80)
    print("✨ Demo complete! Check ROUTING_SYSTEM_DESIGN.md for full implementation details.")
    print("="*80 + "\n")
