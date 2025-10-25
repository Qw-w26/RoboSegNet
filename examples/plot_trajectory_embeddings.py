#!/usr/bin/env python3
"""
Standalone script to visualize trajectory embeddings.

Plots windows from trajectories colored by temporal position,
similar to the original plot_fast_droid_embeddings.py.

Usage:
    python examples/plot_trajectory_embeddings.py --splits_dir ./splits
    python examples/plot_trajectory_embeddings.py --splits_dir ./splits --method tsne --num_traj 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trajectory_embedding_visualizer import visualize_trajectory_embeddings
from training import BoundaryDataset, TrainingDatasetConfig
from data import load_parquet_split
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trajectory embeddings showing temporal progression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--splits_dir',
        type=str,
        default='./splits',
        help='Path to splits directory'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='train',
        help='Data split to visualize'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['pca', 'tsne'],
        default='pca',
        help='Dimensionality reduction method'
    )

    parser.add_argument(
        '--num_traj',
        type=int,
        default=3,
        help='Number of trajectories to visualize'
    )

    parser.add_argument(
        '--max_windows',
        type=int,
        default=None,
        help='Maximum windows per trajectory (None = all)'
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=180,
        help='Window size for dataset'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Stride for window sampling'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: show plot)'
    )

    args = parser.parse_args()

    print("="*80)
    print("Trajectory Embedding Visualization")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Method: {args.method}")
    print(f"Number of trajectories: {args.num_traj}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print()

    # Load data
    print("Loading data split...")
    trajectories = load_parquet_split(args.split, splits_dir=args.splits_dir)
    print(f"Loaded {len(trajectories)} trajectories")

    # Create dataset
    print("\nCreating dataset (this will extract DINO features)...")
    config = TrainingDatasetConfig(
        window_size=args.window_size,
        stride=args.stride,
        vocab_size=2048,
        fps=5.0,
        num_workers=4,
        device='cuda',
        cache_path='./cache'
    )

    dataset = BoundaryDataset(trajectories, config)
    print(f"Dataset size: {len(dataset)} samples")

    # Create visualization
    print(f"\nCreating {args.method.upper()} visualization...")
    print(f"Processing {args.num_traj} trajectories (may take a moment for t-SNE)...")

    fig = visualize_trajectory_embeddings(
        trajectories,
        dataset,
        method=args.method,
        num_trajectories=args.num_traj,
        max_windows_per_traj=args.max_windows,
        title=f'{args.split.capitalize()} Trajectory Embeddings ({args.method.upper()})'
    )

    # Save or show
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
    else:
        print("\nDisplaying plot...")
        plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
