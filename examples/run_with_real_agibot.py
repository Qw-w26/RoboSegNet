"""
Run benchmark with real AgiBotWorld trajectories and Fast-DROID embeddings.

This is the main example showing how to use Fast-DROID + DINOv3 embeddings
with the task segmentation benchmark framework.

Requirements:
- pip install transformers torch h5py opencv-python Pillow
- (Optional) yourdfpy for forward kinematics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    AgiBotHFDirectDataset,
    SpectralClusteringMethod,
    ChangePointDetectionMethod,
    GMMMethod,
    BenchmarkRunner,
    create_embedding_function
)


def main():
    print("="*70)
    print("Task Segmentation with Fast-DROID + DINOv3 Embeddings")
    print("="*70)
    print("\nThis example demonstrates:")
    print("1. Loading real AgiBotWorld trajectories with ground truth")
    print("2. Using Fast-DROID tokenizer for action embeddings")
    print("3. (Optional) DINOv3 visual features if videos available")
    print("4. (Optional) Forward kinematics if robot model available")
    print("5. Benchmarking multiple segmentation methods")
    print("="*70 + "\n")

    # Load real AgiBotWorld dataset
    print("Step 1: Loading AgiBotWorld data from HuggingFace...")
    print("This will download:")
    print("  - JSON annotations (~KB)")
    print("  - Proprioceptive H5 files (~MB per trajectory)")
    print("  - Video files (~100MB per trajectory)\n")

    dataset = AgiBotHFDirectDataset(
        task_ids=[327],  # "Picking items in Supermarket" task
        max_trajectories=5,
        download_proprio=True,  # Download H5 files
        download_videos=True,   # Download videos for DINOv3
        cache_dir="./agibot_full_data"
    )

    # Print statistics
    dataset.print_statistics()

    if len(dataset) == 0:
        print("Error: No trajectories loaded!")
        return

    # Show sample trajectories
    print("\nSample Trajectories:")
    for i in range(min(3, len(dataset))):
        traj = dataset[i]
        print(f"\n  Trajectory {traj.trajectory_id}:")
        print(f"    Segments: {traj.num_segments}")
        print(f"    Total frames: {traj.total_frames}")
        for seg in traj.segments[:2]:  # Show first 2 segments
            print(f"      • [{seg.start_frame:4d}-{seg.end_frame:4d}] {seg.action_text[:50]}")

    # Define segmentation methods
    print("\n" + "="*70)
    print("Step 2: Initializing segmentation methods...")
    methods = [
        SpectralClusteringMethod(
            n_clusters=6,  # Typical number of segments
            temporal_weight=0.3,
            window_size=15,
            stride=1
        ),
        ChangePointDetectionMethod(
            penalty=3.0,
            min_size=5,
            max_segments=10,
            window_size=15,
            stride=1
        ),
        GMMMethod(
            n_components=6,
            window_size=15,
            stride=1
        )
    ]

    print(f"Methods: {[m.name for m in methods]}")

    # Create Fast-DROID embedding function
    print("\n" + "="*70)
    print("Step 3: Initializing Fast-DROID + DINOv3 embedder...")
    print("="*70)

    # Check for optional components
    urdf_path = "/home/kushtimusprime/ur5_policy_training/urdf/ur5_with_robotiq_gripper.urdf"
    has_urdf = Path(urdf_path).exists()

    if has_urdf:
        print(f"✓ Found URDF model: {urdf_path}")
        print("  → Forward kinematics will be used (Cartesian space)")
    else:
        print(f"✗ URDF not found at {urdf_path}")
        print("  → Will use joint space directly")
        urdf_path = None

    print("\nDownloading proprioceptive and video data...")
    print("Creating Fast-DROID + DINOv3 embeddings from real data...\n")

    # Create embedding function with full pipeline
    embedding_fn = create_embedding_function(
        window_size=15,
        stride=1,
        use_cartesian=has_urdf,
        use_dino=True,  # Use DINOv3 visual features
        urdf_path=urdf_path
    )

    # Create benchmark runner
    print("="*70)
    print("Step 4: Running benchmark...")
    print("="*70)
    runner = BenchmarkRunner(
        dataset=dataset,
        methods=methods,
        embedding_fn=embedding_fn,
        output_dir=Path("./fastdroid_results"),
        boundary_tolerance=10  # 10 frames tolerance for boundary detection
    )

    # Run benchmark
    results = runner.run_all_methods(save_predictions=True)

    # Print best method
    print("\n" + "="*70)
    try:
        best_method = runner.get_best_method(metric='macro_f1')
        print(f"Best Method (by Macro F1): {best_method}")
    except Exception as e:
        print(f"Could not determine best method: {e}")
    print("="*70)

    print(f"\nResults saved to: ./fastdroid_results/")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. For better embeddings with real data:")
    print("   - Download full AgiBotWorld dataset (includes H5 & videos)")
    print("   - Or use your local robot dataset")
    print("   - Set use_dino=True to enable visual features")
    print("\n2. Customize the embedding function:")
    print("   from benchmark import create_embedding_function")
    print("   embedding_fn = create_embedding_function(")
    print("       window_size=20,")
    print("       use_cartesian=True,")
    print("       use_dino=True,")
    print("       urdf_path='/path/to/robot.urdf'")
    print("   )")
    print("\n3. Add your own segmentation methods:")
    print("   - Inherit from WindowBasedSegmentationMethod")
    print("   - Implement fit() and predict() methods")
    print("   - Add to methods list")
    print("\n4. Train deep learning models:")
    print("   - Use dataset.trajectories to get training data")
    print("   - Use traj.get_segment_labels() for ground truth")
    print("   - Train your model and wrap it as a method")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print("="*70)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()

        print("\n" + "="*70)
        print("Troubleshooting:")
        print("="*70)
        print("\n1. Missing dependencies:")
        print("   pip install transformers torch h5py opencv-python Pillow")
        print("\n2. HuggingFace connection issues:")
        print("   - Check internet connection")
        print("   - Dataset may require authentication")
        print("\n3. For local datasets:")
        print("   - Ensure H5 files contain 'joint_states' or similar")
        print("   - Videos should be MP4 format")
        print("   - URDF file path should be valid")
        print("="*70)
