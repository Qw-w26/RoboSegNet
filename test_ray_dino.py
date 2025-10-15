"""
Test script for Ray-parallelized DINO feature extraction.

This script tests the Ray parallelization implementation for DINO feature precomputation.
"""

from pathlib import Path
import time
import ray

from data import AgiBotHFDirectDataset
from training import SegmentBoundaryDataset


def test_ray_parallelization():
    """Test Ray parallelization for DINO feature extraction."""

    print("="*70)
    print("Testing Ray-Parallelized DINO Feature Extraction")
    print("="*70)

    # Configuration
    cache_dir = Path("/home/syx/ucsf/task_seg/agibot_full_data")
    task_ids = [327]
    max_trajectories = 2  # Small number for testing

    window_size = 15
    stride = 1
    max_token_length = 600
    boundary_tolerance = 2
    device = 'cuda'

    print(f"\nConfiguration:")
    print(f"  Task IDs: {task_ids}")
    print(f"  Max trajectories: {max_trajectories}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")

    # Load dataset
    print(f"\nLoading dataset from HuggingFace...")
    dataset = AgiBotHFDirectDataset(
        task_ids=task_ids,
        max_trajectories=max_trajectories,
        cache_dir=str(cache_dir),
        download_videos=True,
        download_proprio=True
    )

    print(f"\nDataset loaded:")
    print(f"  Trajectories: {len(dataset.trajectories)}")

    # Test 1: Sequential processing
    print("\n" + "="*70)
    print("Test 1: Sequential Processing (use_ray=False)")
    print("="*70)

    start_time = time.time()
    sequential_dataset = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=window_size,
        stride=stride,
        max_token_length=max_token_length,
        boundary_tolerance=boundary_tolerance,
        device=device,
        precompute_features=True,
        use_ray=False  # Disable Ray
    )
    sequential_time = time.time() - start_time

    print(f"\nSequential processing completed:")
    print(f"  Total samples: {len(sequential_dataset.samples)}")
    print(f"  Cached DINO features: {len(sequential_dataset.cached_dino_features)}")
    print(f"  Time taken: {sequential_time:.2f} seconds")

    # Test 2: Ray parallel processing with 2 workers
    print("\n" + "="*70)
    print("Test 2: Ray Parallel Processing (2 workers)")
    print("="*70)

    # Shutdown Ray if running from previous test
    if ray.is_initialized():
        ray.shutdown()

    start_time = time.time()
    ray_dataset_2 = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=window_size,
        stride=stride,
        max_token_length=max_token_length,
        boundary_tolerance=boundary_tolerance,
        device=device,
        precompute_features=True,
        use_ray=True,
        num_ray_workers=2
    )
    ray_time_2 = time.time() - start_time

    print(f"\nRay parallel processing (2 workers) completed:")
    print(f"  Total samples: {len(ray_dataset_2.samples)}")
    print(f"  Cached DINO features: {len(ray_dataset_2.cached_dino_features)}")
    print(f"  Time taken: {ray_time_2:.2f} seconds")
    print(f"  Speedup vs sequential: {sequential_time / ray_time_2:.2f}x")

    # Test 3: Ray parallel processing with 4 workers
    print("\n" + "="*70)
    print("Test 3: Ray Parallel Processing (4 workers)")
    print("="*70)

    if ray.is_initialized():
        ray.shutdown()

    start_time = time.time()
    ray_dataset_4 = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=window_size,
        stride=stride,
        max_token_length=max_token_length,
        boundary_tolerance=boundary_tolerance,
        device=device,
        precompute_features=True,
        use_ray=True,
        num_ray_workers=4
    )
    ray_time_4 = time.time() - start_time

    print(f"\nRay parallel processing (4 workers) completed:")
    print(f"  Total samples: {len(ray_dataset_4.samples)}")
    print(f"  Cached DINO features: {len(ray_dataset_4.cached_dino_features)}")
    print(f"  Time taken: {ray_time_4:.2f} seconds")
    print(f"  Speedup vs sequential: {sequential_time / ray_time_4:.2f}x")
    print(f"  Speedup vs 2 workers: {ray_time_2 / ray_time_4:.2f}x")

    # Verify correctness
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    # Check that all methods produced the same number of samples
    assert len(sequential_dataset.samples) == len(ray_dataset_2.samples) == len(ray_dataset_4.samples), \
        "Different number of samples!"
    print("✓ All methods produced the same number of samples")

    # Check that all methods cached the same number of features
    assert len(sequential_dataset.cached_dino_features) == len(ray_dataset_2.cached_dino_features) == len(ray_dataset_4.cached_dino_features), \
        "Different number of cached features!"
    print("✓ All methods cached the same number of features")

    # Check feature shapes
    for idx in range(min(3, len(sequential_dataset.cached_dino_features))):
        seq_shape = sequential_dataset.cached_dino_features[idx].shape
        ray2_shape = ray_dataset_2.cached_dino_features[idx].shape
        ray4_shape = ray_dataset_4.cached_dino_features[idx].shape
        assert seq_shape == ray2_shape == ray4_shape == (3 * 1024,), \
            f"Feature shapes don't match for sample {idx}: {seq_shape}, {ray2_shape}, {ray4_shape}"
    print("✓ All feature shapes are correct (3 * 1024 = 3072)")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Sequential:           {sequential_time:.2f}s")
    print(f"Ray (2 workers):      {ray_time_2:.2f}s ({sequential_time / ray_time_2:.2f}x speedup)")
    print(f"Ray (4 workers):      {ray_time_4:.2f}s ({sequential_time / ray_time_4:.2f}x speedup)")
    print("="*70)
    print("✓ All tests passed!")
    print("="*70)

    # Cleanup
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    test_ray_parallelization()
