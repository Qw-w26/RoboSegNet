#!/usr/bin/env python3
"""Data preparation script for boundary prediction.

This script handles:
- Downloading AgiBotWorld dataset if it doesn't exist
- Loading trajectory data from different sources
- Splitting into train/validation/test sets
- Creating split manifests for reproducible experiments
- Optionally pre-extracting and caching DINO features

Example usage:
    # Prepare data for task 327
    python scripts/prepare_data.py --task-ids 327 --output-dir ./splits

    # Prepare with custom split ratios
    python scripts/prepare_data.py --task-ids 327 352 \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

    # Pre-extract DINO features
    python scripts/prepare_data.py --task-ids 327 \\
        --extract-dino --cache-dir ./cache

Author: Claude Code
Date: October 24, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.splits import DatasetSplitter, SplitConfig, SplitManifest
from data.agibot_dataset import AgiBotWorldDataset, download_agibot_dataset
from data.dataset_builder import DatasetBuilder
from utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare data splits for boundary prediction training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data source
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/syx/ucsf/agibot_full_data/extracted',
        help='Root directory containing trajectory data'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download the dataset from HuggingFace if it does not exist'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        required=True,
        help='Task IDs to include in the dataset'
    )

    parser.add_argument(
        '--max-trajectories',
        type=int,
        default=None,
        help='Maximum number of trajectories to use (None = use all)'
    )

    # Split configuration
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion of data for training'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for validation'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for testing'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--stratify-by-task',
        action='store_true',
        default=True,
        help='Stratify splits by task ID'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./splits',
        help='Output directory for split manifests'
    )

    parser.add_argument(
        '--split-name',
        type=str,
        default=None,
        help='Name for the split (auto-generated if not specified)'
    )

    # Feature extraction and dataset building
    parser.add_argument(
        '--extract-dino',
        action='store_true',
        default=True,
        help='Extract DINO features from videos'
    )

    parser.add_argument(
        '--extract-frames',
        default=True,
        action='store_true',
        help='Extract individual frames from videos'
    )

    parser.add_argument(
        '--dino-model',
        type=str,
        default='dinov2_vitb14',
        help='DINO model variant to use'
    )

    parser.add_argument(
        '--target-fps',
        type=float,
        default=5.0,
        help='Target FPS for frame/feature extraction'
    )

    parser.add_argument(
        '--build-dataset',
        action='store_true',
        default=True,
        help='Build comprehensive processed dataset'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of Ray workers for parallel processing (None = auto-detect)'
    )

    return parser.parse_args()


def load_trajectories(data_dir: Path, task_ids: List[int], download: bool = False) -> List:
    """Load trajectory data for specified tasks.

    Args:
        data_dir: Root data directory
        task_ids: List of task IDs to load
        download: Whether to download the dataset if it doesn't exist

    Returns:
        List of trajectory objects
    """
    print(f"Loading trajectories from {data_dir} for tasks: {task_ids}")

    # Check if data exists or download is requested
    if not data_dir.exists() or download:
        if download or not data_dir.exists():
            print(f"Data directory {data_dir} does not exist or download requested.")
            response = input("Download AgiBotWorld dataset from HuggingFace? (y/n): ").strip().lower()
            if response == 'y':
                print("Downloading dataset...")
                success = download_agibot_dataset(data_dir, task_ids)
                if not success:
                    print("Failed to download dataset")
                    return []
            else:
                print("Dataset download cancelled")
                return []

    # Try multiple data loading approaches
    trajectories = []

    # First try: Use AgiBotWorldDataset from our agibot_dataset module
    try:
        print("Loading with AgiBotWorldDataset...")
        dataset = AgiBotWorldDataset(
            data_root=data_dir,
            task_ids=task_ids,
            split='train'  # We'll split later
        )

        trajectories = dataset.trajectories
        print(f"Loaded {len(trajectories)} trajectories using AgiBotWorldDataset")
        return trajectories

    except Exception as e:
        print(f"Failed to load with AgiBotWorldDataset: {e}")


    if not trajectories:
        print("Error: Could not load trajectories using any available method")
        print("Please ensure the dataset is properly downloaded and formatted")

    return trajectories


def create_splits(
    trajectories: List,
    config: SplitConfig,
    output_dir: Path,
    split_name: Optional[str] = None
) -> None:
    """Create and save train/val/test splits.

    Args:
        trajectories: List of trajectory objects
        config: Split configuration
        output_dir: Output directory for manifests
        split_name: Optional name for the split
    """
    print(f"\nCreating splits with config:")
    print(f"  Train ratio: {config.train_ratio}")
    print(f"  Val ratio: {config.val_ratio}")
    print(f"  Test ratio: {config.test_ratio}")
    print(f"  Seed: {config.seed}")
    print(f"  Stratify by task: {config.stratify_by_task}")

    # Create splitter
    splitter = DatasetSplitter(config)

    # Get task IDs from trajectories
    task_ids = []
    for t in trajectories:
        if hasattr(t, 'task_id'):
            task_ids.append(t.task_id)
        else:
            task_ids.append(0)  # Default task ID if not present

    # Split trajectories into indices
    train_indices, val_indices, test_indices = splitter.split_trajectories(trajectories, task_ids)

    # Generate split name if not provided
    if split_name is None:
        unique_task_ids = set(task_ids)
        task_str = '_'.join(map(str, sorted(unique_task_ids)))
        split_name = f"task_{task_str}"

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_indices)} trajectories")
    print(f"  Val: {len(val_indices)} trajectories")
    print(f"  Test: {len(test_indices)} trajectories")

    # Create manifest
    manifest = splitter.create_manifest(
        trajectories=trajectories,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        task_ids=task_ids
    )

    # Save manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split_name}_manifest.json"
    manifest.save(manifest_path)

    print(f"\nSaved split manifest to: {manifest_path}")

    # Also save individual split files with just the indices for backward compatibility
    train_path = output_dir / f"{split_name}_train.json"
    val_path = output_dir / f"{split_name}_val.json"
    test_path = output_dir / f"{split_name}_test.json"

    with open(train_path, 'w') as f:
        json.dump({'indices': train_indices, 'size': len(train_indices)}, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump({'indices': val_indices, 'size': len(val_indices)}, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump({'indices': test_indices, 'size': len(test_indices)}, f, indent=2)

    print(f"\nAlso saved individual splits to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")

    # Print detailed statistics
    from data.splits import print_split_summary
    print_split_summary(manifest)


def build_processed_datasets(
    trajectories: List,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    output_dir: Path,
    args: argparse.Namespace
) -> None:
    """Build comprehensive processed datasets for all splits.

    Args:
        trajectories: List of trajectory objects
        train_indices: Training split indices
        val_indices: Validation split indices
        test_indices: Test split indices
        output_dir: Output directory for processed datasets
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print("BUILDING COMPREHENSIVE DATASET (PARQUET FORMAT)")
    print(f"{'='*80}")

    # Create dataset builder
    dataset_dir = output_dir / 'processed_dataset'
    builder = DatasetBuilder(
        output_dir=dataset_dir,
        extract_dino=args.extract_dino,
        extract_frames=args.extract_frames,
        dino_model=args.dino_model,
        target_fps=args.target_fps,
        num_workers=args.num_workers
    )

    # Build datasets for each split
    splits = {}

    print("\nProcessing train split...")
    train_processed = builder.build_dataset(trajectories, 'train', train_indices)
    splits['train'] = train_processed

    print("\nProcessing validation split...")
    val_processed = builder.build_dataset(trajectories, 'val', val_indices)
    splits['val'] = val_processed

    print("\nProcessing test split...")
    test_processed = builder.build_dataset(trajectories, 'test', test_indices)
    splits['test'] = test_processed

    print(f"\n{'='*80}")
    print("DATASET BUILD COMPLETE")
    print(f"{'='*80}")
    print(f"\nProcessed datasets saved to: {dataset_dir}")
    print(f"  Format: Parquet with relative paths")
    print(f"  Images: PNG files in frames/ directory")
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_processed)} trajectories")
    print(f"  Val: {len(val_processed)} trajectories")
    print(f"  Test: {len(test_processed)} trajectories")

    # Print feature extraction info
    if args.extract_dino:
        print(f"\n  DINO features: Extracted ({args.dino_model}, subsampled)")
    else:
        print(f"\n  DINO features: Not extracted")

    if args.extract_frames:
        print(f"  Frames: Extracted at {args.target_fps} FPS (subsampled)")
    else:
        print(f"  Frames: Not extracted")

    print(f"\nDatasets are ready for use in downstream modules!")
    print(f"\nTo load a split:")
    print(f"  df = pd.read_parquet('{dataset_dir}/train_processed.parquet')")
    print(f"  # Images at: {dataset_dir}/frames/<trajectory_id>/frame_XXXXXX.png")


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed
    set_all_seeds(args.seed)

    # Load trajectories
    data_dir = Path(args.data_dir)
    trajectories = load_trajectories(data_dir, args.task_ids, args.download)

    if len(trajectories) == 0:
        print("Error: No trajectories loaded!")
        sys.exit(1)

    # Limit number of trajectories if requested
    if args.max_trajectories is not None and args.max_trajectories < len(trajectories):
        print(f"\nLimiting trajectories from {len(trajectories)} to {args.max_trajectories}")
        trajectories = trajectories[:args.max_trajectories]

    # Create split configuration
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_task=args.stratify_by_task
    )

    # Create splits and get indices
    output_dir = Path(args.output_dir)

    # Create splitter
    splitter = DatasetSplitter(split_config)

    # Get task IDs from trajectories
    task_ids = []
    for t in trajectories:
        if hasattr(t, 'task_id'):
            task_ids.append(t.task_id)
        else:
            task_ids.append(0)

    # Split trajectories into indices
    train_indices, val_indices, test_indices = splitter.split_trajectories(trajectories, task_ids)

    # Save split manifests (keeping the old functionality)
    create_splits(trajectories, split_config, output_dir, args.split_name)

    # Build comprehensive processed datasets if requested
    if args.build_dataset:
        build_processed_datasets(
            trajectories=trajectories,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            output_dir=output_dir,
            args=args
        )

    print("\n✓ Data preparation complete!")


if __name__ == '__main__':
    main()
