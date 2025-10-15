"""
DEPRECATED: This script is for reference only.

Use run_with_real_agibot.py instead - it demonstrates the full Fast-DROID
embedding pipeline and is the recommended example.

This file shows command-line argument handling for advanced use cases.
"""

import sys
from pathlib import Path
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    AgiBotWorldDataset,
    SpectralClusteringMethod,
    ChangePointDetectionMethod,
    GMMMethod,
    HierarchicalClusteringMethod,
    KMeansMethod,
    BenchmarkRunner
)


def create_embedding_function(embedding_type='dummy'):
    """
    Create an embedding function for trajectories.

    In practice, you would replace this with your actual embedding method
    (e.g., using Fast-DROID + DINOv3 as in plot_fast_droid_embeddings.py).

    Args:
        embedding_type: Type of embedding ('dummy', 'fastdroid', etc.)

    Returns:
        Function that takes TrajectoryData and returns embeddings
    """
    if embedding_type == 'dummy':
        # Simple dummy embeddings for demonstration
        def dummy_embedding(trajectory_data):
            """Create dummy embeddings from trajectory data."""
            # Load proprioceptive data
            proprio = trajectory_data.load_proprioception()

            if len(proprio) == 0:
                return None

            # Create simple windows
            window_size = 15
            stride = 1
            windows = []

            for i in range(0, len(proprio) - window_size + 1, stride):
                window = proprio[i:i+window_size]
                # Simple features: mean, std
                features = np.concatenate([
                    window.mean(axis=0),
                    window.std(axis=0)
                ])
                windows.append(features)

            if not windows:
                return None

            return np.array(windows)

        return dummy_embedding

    elif embedding_type == 'fastdroid':
        # TODO: Implement Fast-DROID + DINOv3 embeddings
        # This would use the code from plot_fast_droid_embeddings.py
        raise NotImplementedError("Fast-DROID embeddings not yet implemented")

    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Run task segmentation benchmark on AgiBotWorld dataset'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        help='Path to AgiBotWorld dataset root directory (not required if --streaming)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Stream data directly from HuggingFace Hub instead of loading locally'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        help='HuggingFace token for accessing the dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./benchmark_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--task_ids',
        type=int,
        nargs='+',
        help='Specific task IDs to evaluate on'
    )
    parser.add_argument(
        '--max_trajectories',
        type=int,
        default=5,
        help='Maximum number of trajectories to evaluate (default: 5)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['spectral', 'changepoint', 'gmm', 'hierarchical', 'kmeans'],
        choices=['spectral', 'changepoint', 'gmm', 'hierarchical', 'kmeans'],
        help='Segmentation methods to evaluate'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=5,
        help='Number of clusters for clustering methods'
    )
    parser.add_argument(
        '--boundary_tolerance',
        type=int,
        default=5,
        help='Tolerance for boundary detection (frames)'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save per-trajectory predictions'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.streaming and not args.data_root:
        parser.error("--data_root is required when not using --streaming")

    # Load dataset
    if args.streaming:
        print("Loading AgiBotWorld dataset from HuggingFace Hub in streaming mode...")
        dataset = AgiBotWorldDataset(
            streaming=True,
            task_ids=args.task_ids,
            max_trajectories=args.max_trajectories,
            hf_token=args.hf_token
        )
    else:
        print(f"Loading AgiBotWorld dataset from {args.data_root}...")
        dataset = AgiBotWorldDataset(
            data_root=Path(args.data_root),
            task_ids=args.task_ids,
            max_trajectories=args.max_trajectories
        )

    # Print dataset statistics
    dataset.print_statistics()

    # Define methods to benchmark
    methods = []

    if 'spectral' in args.methods:
        methods.append(
            SpectralClusteringMethod(
                n_clusters=args.n_clusters,
                temporal_weight=0.3,
                window_size=15,
                stride=1
            )
        )

    if 'changepoint' in args.methods:
        methods.append(
            ChangePointDetectionMethod(
                penalty=3.0,
                min_size=5,
                max_segments=args.n_clusters,
                window_size=15,
                stride=1
            )
        )

    if 'gmm' in args.methods:
        methods.append(
            GMMMethod(
                n_components=args.n_clusters,
                window_size=15,
                stride=1
            )
        )

    if 'hierarchical' in args.methods:
        methods.append(
            HierarchicalClusteringMethod(
                n_clusters=args.n_clusters,
                window_size=15,
                stride=1
            )
        )

    if 'kmeans' in args.methods:
        methods.append(
            KMeansMethod(
                n_clusters=args.n_clusters,
                window_size=15,
                stride=1
            )
        )

    print(f"\nMethods to evaluate: {[m.name for m in methods]}")

    # Create embedding function
    embedding_fn = create_embedding_function('dummy')

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        methods=methods,
        embedding_fn=embedding_fn,
        output_dir=Path(args.output_dir),
        boundary_tolerance=args.boundary_tolerance
    )

    # Run benchmark
    print("\nStarting benchmark...")
    results = runner.run_all_methods(save_predictions=args.save_predictions)

    # Print best method
    best_method = runner.get_best_method(metric='macro_f1')
    print(f"\n{'='*70}")
    print(f"Best Method (by Macro F1): {best_method}")
    print(f"{'='*70}\n")

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
