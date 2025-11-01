#!/usr/bin/env python3
"""Data preparation script for boundary prediction with parallel processing using Ray.

This script handles:
- Downloading tasks from HuggingFace in parallel (using HF Hub API + TAR extraction)
- Loading trajectory metadata in parallel to create stratified splits
- Processing multiple tasks concurrently (extract features, frames)
- Cleaning up downloaded raw data after processing to save disk space
- Merging processed datasets into final train/val/test splits
- Creating split manifests for reproducible experiments

Key Features:
- Parallel processing: Uses Ray to process multiple tasks concurrently
- Memory efficient: Only loads raw data needed for active tasks
- Disk efficient: Automatically cleans up downloaded data after processing
- Uses HuggingFace Hub API to download TAR files directly
- Maintains proper stratification across all tasks in splits
- Configurable parallelism: Control max parallel tasks with --max-parallel-tasks

Example usage:
    # Download and process 4 tasks in parallel
    python scripts/prepare_data.py --task-ids 327 352 357 --download \\
        --max-parallel-tasks 4 --output-dir ./splits

    # Process ALL tasks with 8 parallel workers
    python scripts/prepare_data.py --download --max-parallel-tasks 8 \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

    # Single task (no parallelism needed)
    python scripts/prepare_data.py --task-ids 327 --download --output-dir ./splits

    # Keep downloaded data (disable cleanup)
    python scripts/prepare_data.py --task-ids 327 --download --no-cleanup

    # Use already downloaded data (no download, no cleanup)
    python scripts/prepare_data.py --task-ids 327 --output-dir ./splits

Workflow:
1. Load trajectory metadata for all tasks in parallel (downloads JSON annotations)
2. Create stratified train/val/test splits across all tasks
3. Process tasks in parallel batches:
   a. Download task data TARs (if --download enabled)
   b. Extract TARs and process features
   c. Save processed data
   d. Delete raw downloaded data and TARs (if cleanup enabled)
4. Merge all processed data into final datasets

Author: Claude Code
Date: October 30, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import os
import tarfile
import shutil
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
from tqdm import tqdm
import ray

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.splits import DatasetSplitter, SplitConfig, SplitManifest
from data.agibot_dataset import AgiBotWorldDataset
from data.dataset_builder import DatasetBuilder
from utils.reproducibility import set_all_seeds


REPO_ID = "agibot-world/AgiBotWorld-Alpha"


def discover_all_task_ids() -> List[int]:
    """Discover all available task IDs from HuggingFace dataset.

    Returns:
        List of task IDs found in the dataset
    """
    print("Discovering all available task IDs from HuggingFace...")

    try:
        api = HfApi()
        # List files in task_info directory
        files = api.list_repo_files(REPO_ID, repo_type="dataset")

        # Extract task IDs from task_*.json files
        task_ids = []
        for file_path in files:
            if file_path.startswith('task_info/task_') and file_path.endswith('.json'):
                filename = file_path.split('/')[-1]
                task_id_str = filename.replace('task_', '').replace('.json', '')
                try:
                    task_id = int(task_id_str)
                    task_ids.append(task_id)
                except ValueError:
                    continue

        task_ids.sort()
        print(f"Found {len(task_ids)} tasks: {task_ids}")
        return task_ids

    except Exception as e:
        print(f"Error: Failed to discover tasks via HuggingFace API: {e}")
        print("Please install huggingface_hub: pip install huggingface_hub")
        print("Or specify task IDs manually with --task-ids")
        return []


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
        help='Download tasks from HuggingFace if they do not exist (uses HF Hub API + TAR extraction)'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        default = True,
        help='Keep downloaded raw data after processing (default: cleanup to save space)'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        default=None,
        help='Task IDs to include in the dataset (default: discover all available tasks from HuggingFace)'
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

    # Feature extraction and dataset building (enabled by default)
    parser.add_argument(
        '--no-extract-dino',
        action='store_true',
        help='Disable DINO feature extraction (default: enabled)'
    )

    parser.add_argument(
        '--no-extract-frames',
        action='store_true',
        help='Disable frame extraction (default: enabled)'
    )

    parser.add_argument(
        '--dino-model',
        type=str,
        default='facebook/dinov3-vitl16-pretrain-lvd1689m',
        help='DINO model variant to use (HuggingFace model name)'
    )

    parser.add_argument(
        '--target-fps',
        type=float,
        default=2.0,
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

    parser.add_argument(
        '--max-parallel-tasks',
        type=int,
        default=2,
        help='[DEPRECATED] No longer used. GPU parallelism now happens at trajectory level for better efficiency.'
    )

    args = parser.parse_args()

    # Set defaults for feature extraction (enabled by default, disabled if --no-* flag provided)
    args.extract_dino = not args.no_extract_dino
    args.extract_frames = not args.no_extract_frames

    return args


def _download_proprio_stats(
    data_dir: Path,
    cache_dir: Path,
    episode_ids: List[int],
    episode_data_cache: Dict[int, Dict[str, Any]]
):
    """Download and extract proprioceptive stats for given episode IDs."""
    print(f"Downloading proprioceptive data...")
    try:
        # Find all proprio_stats TAR files
        all_files = list_repo_files(REPO_ID, repo_type='dataset')
        proprio_tar_files = [
            f for f in all_files
            if f.startswith('proprio_stats/') and f.endswith('.tar')
        ]

        if not proprio_tar_files:
            print(f"  Warning: No proprio_stats TAR files found")
            return

        print(f"  Found {len(proprio_tar_files)} proprio_stats TAR files")

        proprio_extract_dir = data_dir / 'proprio_stats'
        proprio_extract_dir.mkdir(parents=True, exist_ok=True)

        # Check for already extracted H5 files first
        already_extracted = 0
        for ep_id in episode_ids:
            flat_path = proprio_extract_dir / f"{ep_id}.h5"
            if flat_path.exists():
                if ep_id not in episode_data_cache:
                    episode_data_cache[ep_id] = {'videos': {}, 'h5': None}
                episode_data_cache[ep_id]['h5'] = flat_path
                already_extracted += 1

        # Download and extract from each TAR file
        total_extracted_count = 0
        for tar_filename in proprio_tar_files:
            tar_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=tar_filename,
                repo_type='dataset',
                cache_dir=str(cache_dir)
            )

            # Extract only needed H5 files
            extracted_count = 0
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('proprio_stats.h5'):
                        parts = Path(member.name).parts
                        if len(parts) >= 3:
                            try:
                                ep_id = int(parts[1])
                                if ep_id in episode_ids:
                                    # Skip if already extracted or cached
                                    if ep_id in episode_data_cache and episode_data_cache[ep_id].get('h5'):
                                        continue

                                    # Extract to flat structure
                                    target_path = proprio_extract_dir / f"{ep_id}.h5"
                                    if not target_path.exists():
                                        tar.extract(member, path=proprio_extract_dir)
                                        extracted_path = proprio_extract_dir / member.name
                                        extracted_path.rename(target_path)
                                        # Cleanup dirs
                                        try:
                                            extracted_path.parent.rmdir()
                                            extracted_path.parent.parent.rmdir()
                                        except:
                                            pass
                                        extracted_count += 1
                                        total_extracted_count += 1

                                    # Cache it
                                    if ep_id not in episode_data_cache:
                                        episode_data_cache[ep_id] = {'videos': {}, 'h5': None}
                                    episode_data_cache[ep_id]['h5'] = target_path
                            except (ValueError, IndexError):
                                pass

            if extracted_count > 0:
                print(f"    Extracted {extracted_count} H5 files from {tar_filename}")

        print(f"  ✓ Extracted {total_extracted_count} new H5 files ({already_extracted} already existed)")

    except Exception as e:
        print(f"Warning: Failed to download proprioceptive stats: {e}")


def download_and_extract_task(
    data_dir: Path,
    task_id: int,
    cache_dir: Path,
    download: bool = False
) -> Tuple[bool, Dict[int, Dict[str, Any]]]:
    """Download and extract TAR files for a single task from HuggingFace.

    Args:
        data_dir: Root data directory for extracted files
        task_id: Task ID to download
        cache_dir: Directory for HuggingFace cache
        download: Whether to download if not already extracted

    Returns:
        Tuple of (success, episode_data_cache) where episode_data_cache maps
        episode_id to {'videos': {type: path}, 'h5': path}
    """
    print(f"\n{'='*80}")
    print(f"Processing task {task_id}")
    print(f"{'='*80}")

    # Support both flat and nested directory structures
    # Flat structure: data_dir/327/ (episode dirs)
    # Nested structure: data_dir/task_327/observations/ (episode dirs)
    extract_dir = data_dir / str(task_id)
    extract_dir_nested = data_dir / f'task_{task_id}' / 'observations'

    # Use nested structure if it exists
    if extract_dir_nested.exists():
        extract_dir = extract_dir_nested

    episode_data_cache = {}

    # Check for task info in both possible locations
    task_info_file = data_dir / 'task_info' / f'task_{task_id}.json'
    task_info_file_nested = data_dir / f'task_{task_id}' / 'task_info' / f'task_{task_id}.json'

    if task_info_file_nested.exists():
        task_info_file = task_info_file_nested

    videos_already_extracted = extract_dir.exists() and task_info_file.exists()

    if videos_already_extracted:
        print(f"✓ Task {task_id} data already extracted")

        # Load episode IDs to check if we need to download proprio data
        with open(task_info_file, 'r') as f:
            episodes = json.load(f)
        episode_ids = [ep.get('episode_id') for ep in episodes]

        # Check if proprioception data is missing for any episodes
        proprio_dir = data_dir / 'proprio_stats'
        missing_proprio = []
        if proprio_dir.exists():
            for ep_id in episode_ids:
                h5_path = proprio_dir / f"{ep_id}.h5"
                if not h5_path.exists():
                    missing_proprio.append(ep_id)
        else:
            missing_proprio = episode_ids

        # Download proprioception data if missing
        if missing_proprio and download:
            print(f"Downloading missing proprioception data for {len(missing_proprio)} episodes...")
            _download_proprio_stats(data_dir, cache_dir, episode_ids, episode_data_cache)

        # Index all files (videos and H5) - do this after proprio download
        _index_extracted_files(extract_dir, task_id, episode_data_cache)

        return True, episode_data_cache

    if not download:
        print(f"✗ Task {task_id} data not found and download not enabled")
        return False, {}

    # Download task info JSON
    print(f"Downloading task info for task {task_id}...")
    try:
        json_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f'task_info/task_{task_id}.json',
            repo_type='dataset',
            cache_dir=str(cache_dir)
        )

        # Copy to data_dir structure
        task_info_dir = data_dir / 'task_info'
        task_info_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(json_path, task_info_file)
        print(f"  ✓ Downloaded task info")

        # Load to get episode IDs
        with open(json_path, 'r') as f:
            episodes = json.load(f)
        episode_ids = [ep.get('episode_id') for ep in episodes]
        print(f"  Found {len(episode_ids)} episodes")

    except Exception as e:
        print(f"✗ Failed to download task info: {e}")
        return False, {}

    # Download and extract observation TARs (videos)
    print(f"Downloading observation TARs for task {task_id}...")
    try:
        all_files = list_repo_files(REPO_ID, repo_type='dataset')
        tar_files = [
            f for f in all_files
            if f.startswith(f'observations/{task_id}/') and f.endswith('.tar')
        ]

        if not tar_files:
            print(f"  No observation TAR files found")
        else:
            print(f"  Found {len(tar_files)} TAR files")
            extract_dir.mkdir(parents=True, exist_ok=True)

            for tar_filename in tqdm(tar_files, desc="Extracting TARs"):
                tar_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=tar_filename,
                    repo_type='dataset',
                    cache_dir=str(cache_dir)
                )

                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=extract_dir)

            # Index extracted files
            _index_extracted_files(extract_dir, task_id, episode_data_cache)
            print(f"  ✓ Extracted {len(tar_files)} TARs")

    except Exception as e:
        print(f"Warning: Failed to download observations: {e}")

    # Download and extract proprioceptive stats
    _download_proprio_stats(data_dir, cache_dir, episode_ids, episode_data_cache)

    return True, episode_data_cache


def _index_extracted_files(
    extract_dir: Path,
    task_id: int,
    episode_data_cache: Dict[int, Dict[str, Any]]
):
    """Index extracted video and H5 files by episode ID."""
    indexed_videos = 0
    indexed_h5 = 0

    # Index videos from task-specific directory
    for file_path in extract_dir.rglob('*'):
        if not file_path.is_file():
            continue

        try:
            parts = file_path.relative_to(extract_dir).parts
            episode_id = None

            # Try directory structure first (e.g., 648649/videos/head_color.mp4)
            if len(parts) >= 2:
                try:
                    episode_id = int(parts[0])
                except ValueError:
                    pass

            # Fallback: extract from filename
            if episode_id is None:
                filename = file_path.stem.split('_')[0]
                episode_id = int(filename)

            if episode_id is None:
                continue

            if episode_id not in episode_data_cache:
                episode_data_cache[episode_id] = {'videos': {}, 'h5': None}

            if file_path.suffix == '.h5':
                episode_data_cache[episode_id]['h5'] = file_path
                indexed_h5 += 1
            elif file_path.suffix == '.mp4':
                stem_parts = file_path.stem.split('_')
                try:
                    int(stem_parts[0])
                    video_type = '_'.join(stem_parts[1:])
                except ValueError:
                    video_type = file_path.stem

                if video_type:
                    episode_data_cache[episode_id]['videos'][video_type] = file_path
                    indexed_videos += 1

        except (ValueError, IndexError, TypeError):
            pass

    # Index H5 files from proprio_stats directory (stored separately)
    # Check multiple possible locations for proprio_stats
    # Location 1: data_dir/proprio_stats (for nested structure like data_dir/task_327/observations)
    # Location 2: extract_dir.parent/proprio_stats (for flat structure like data_dir/327)
    proprio_dir_root = None

    # Try to find the root data directory
    if 'observations' in extract_dir.parts:
        # Nested structure: go up to root
        idx = extract_dir.parts.index('observations')
        proprio_dir_root = Path(*extract_dir.parts[:idx-1]) / 'proprio_stats'
    else:
        # Flat structure
        proprio_dir_root = extract_dir.parent / 'proprio_stats'

    proprio_dir = proprio_dir_root
    if proprio_dir.exists():
        # Try flat structure first: proprio_stats/648642.h5
        for h5_file in proprio_dir.glob('*.h5'):
            try:
                # H5 files are named like: 648642.h5
                episode_id = int(h5_file.stem)

                if episode_id not in episode_data_cache:
                    episode_data_cache[episode_id] = {'videos': {}, 'h5': None}

                # Update cache if not already set, but always count
                if episode_data_cache[episode_id]['h5'] is None:
                    episode_data_cache[episode_id]['h5'] = h5_file

                # Count all H5 files found, not just newly added ones
                indexed_h5 += 1

            except (ValueError, IndexError):
                pass

        # Also try nested structure: proprio_stats/TASK_ID/EPISODE_ID/proprio_stats.h5
        task_proprio_dir = proprio_dir / str(task_id)
        if task_proprio_dir.exists():
            for episode_dir in task_proprio_dir.iterdir():
                if episode_dir.is_dir():
                    try:
                        episode_id = int(episode_dir.name)
                        h5_file = episode_dir / 'proprio_stats.h5'

                        if h5_file.exists():
                            if episode_id not in episode_data_cache:
                                episode_data_cache[episode_id] = {'videos': {}, 'h5': None}

                            # Update cache if not already set
                            if episode_data_cache[episode_id]['h5'] is None:
                                episode_data_cache[episode_id]['h5'] = h5_file
                                indexed_h5 += 1

                    except (ValueError, IndexError):
                        pass

    if indexed_videos > 0 or indexed_h5 > 0:
        print(f"  Indexed {indexed_videos} videos and {indexed_h5} H5 files")


def load_task_trajectories(
    data_dir: Path,
    task_id: int,
    episode_data_cache: Dict[int, Dict[str, Any]]
) -> List:
    """Load trajectory data for a task using extracted files.

    Args:
        data_dir: Root data directory
        task_id: Task ID
        episode_data_cache: Mapping of episode_id to file paths

    Returns:
        List of trajectory objects
    """
    from data.base_dataset import TrajectoryData, SegmentAnnotation

    try:
        # Load task info JSON - check both possible locations
        task_info_file = data_dir / 'task_info' / f'task_{task_id}.json'
        task_info_file_nested = data_dir / f'task_{task_id}' / 'task_info' / f'task_{task_id}.json'

        if task_info_file_nested.exists():
            task_info_file = task_info_file_nested

        with open(task_info_file, 'r') as f:
            episodes = json.load(f)

        trajectories = []
        for episode in episodes:
            episode_id = episode.get('episode_id')
            task_name = f"Task {task_id}"

            # Parse segments
            segments = []
            label_info = episode.get('label_info') or episode.get('lable_info')
            if label_info and 'action_config' in label_info:
                for seg_id, action in enumerate(label_info['action_config']):
                    segment = SegmentAnnotation(
                        start_frame=action['start_frame'],
                        end_frame=action['end_frame'],
                        action_text=action['action_text'],
                        skill=action.get('skill'),
                        segment_id=seg_id
                    )
                    segments.append(segment)

            # Get file paths from cache
            ep_data = episode_data_cache.get(episode_id, {})
            h5_path = ep_data.get('h5')
            video_paths = ep_data.get('videos', {})

            trajectory = TrajectoryData(
                trajectory_id=f"{task_id}_{episode_id}",
                task_id=str(task_id),
                task_name=task_name,
                proprioception_path=h5_path,
                video_paths=video_paths if video_paths else None,
                segments=segments,
                metadata={
                    'episode_id': episode_id,
                    'source': 'huggingface_direct',
                    'has_proprio': h5_path is not None,
                    'has_videos': len(video_paths) > 0
                }
            )
            trajectories.append(trajectory)

        print(f"✓ Loaded {len(trajectories)} trajectories for task {task_id}")
        return trajectories

    except Exception as e:
        print(f"✗ Failed to load task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return []




def cleanup_task_data(data_dir: Path, cache_dir: Path, task_id: int, episode_ids: List[int] = None):
    """Remove downloaded and extracted data for a specific task to free disk space.

    Args:
        data_dir: Root data directory (where files are extracted)
        cache_dir: HuggingFace cache directory
        task_id: Task ID to clean up
        episode_ids: Optional list of episode IDs to cleanup H5 files
    """
    print(f"\nCleaning up downloaded data for task {task_id}...")

    bytes_freed = 0

    # Remove extracted videos for this task
    extract_dir = data_dir / str(task_id)
    if extract_dir.exists():
        size = sum(f.stat().st_size for f in extract_dir.rglob('*') if f.is_file())
        bytes_freed += size
        shutil.rmtree(extract_dir)
        print(f"  Removed extracted videos: {size / (1024**3):.2f} GB")

    # Remove proprioception H5 files for this task's episodes
    if episode_ids:
        proprio_dir = data_dir / 'proprio_stats'
        if proprio_dir.exists():
            h5_size = 0
            for episode_id in episode_ids:
                h5_file = proprio_dir / f"{episode_id}.h5"
                if h5_file.exists():
                    h5_size += h5_file.stat().st_size
                    h5_file.unlink()
            if h5_size > 0:
                bytes_freed += h5_size
                print(f"  Removed H5 files: {h5_size / (1024**2):.2f} MB")

    print(f"  Total space freed: {bytes_freed / (1024**3):.2f} GB")

    # Note: We keep task_info JSON files as they're small and needed for manifests
    # Note: HuggingFace cache in cache_dir is kept for potential reuse




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


@ray.remote
def load_task_metadata_remote(
    task_id: int,
    data_dir: Path,
    cache_dir: Path,
    download: bool
) -> Tuple[int, List, int]:
    """Ray remote function to load metadata for a single task.

    Args:
        task_id: Task ID to load
        data_dir: Root data directory
        cache_dir: HuggingFace cache directory
        download: Whether to download if not found

    Returns:
        Tuple of (task_id, trajectories, trajectory_count)
    """
    from data.base_dataset import TrajectoryData, SegmentAnnotation

    try:
        # Check for task info JSON in multiple possible locations
        # Location 1: data_dir/task_info/task_XXX.json (downloaded structure)
        task_info_file = data_dir / 'task_info' / f'task_{task_id}.json'

        # Location 2: data_dir/task_XXX/task_info/task_XXX.json (nested structure)
        task_info_file_nested = data_dir / f'task_{task_id}' / 'task_info' / f'task_{task_id}.json'

        # Prefer nested structure if it exists
        if task_info_file_nested.exists():
            task_info_file = task_info_file_nested
        elif not task_info_file.exists():
            if download:
                print(f"[Task {task_id}] Downloading metadata...")
                json_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f'task_info/task_{task_id}.json',
                    repo_type='dataset',
                    cache_dir=str(cache_dir)
                )
                # Copy to data_dir structure
                task_info_dir = data_dir / 'task_info'
                task_info_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(json_path, task_info_file)
            else:
                print(f"[Task {task_id}] Metadata not found and download not enabled, skipping...")
                return task_id, [], 0

        # Load metadata only (no videos)
        print(f"[Task {task_id}] Loading metadata...")
        with open(task_info_file, 'r') as f:
            episodes = json.load(f)

        # Create lightweight trajectory objects (no actual file loading)
        trajectories = []
        for episode in episodes:
            episode_id = episode.get('episode_id')
            task_name = f"Task {task_id}"

            # Parse segments
            segments = []
            label_info = episode.get('label_info') or episode.get('lable_info')
            if label_info and 'action_config' in label_info:
                for seg_id, action in enumerate(label_info['action_config']):
                    segment = SegmentAnnotation(
                        start_frame=action['start_frame'],
                        end_frame=action['end_frame'],
                        action_text=action['action_text'],
                        skill=action.get('skill'),
                        segment_id=seg_id
                    )
                    segments.append(segment)

            trajectory = TrajectoryData(
                trajectory_id=f"{task_id}_{episode_id}",
                task_id=str(task_id),
                task_name=task_name,
                proprioception_path=None,  # Will be loaded later
                video_paths=None,  # Will be loaded later
                segments=segments,
                metadata={'episode_id': episode_id}
            )
            trajectories.append(trajectory)

        print(f"[Task {task_id}] Loaded {len(trajectories)} trajectories")
        return task_id, trajectories, len(trajectories)

    except Exception as e:
        print(f"[Task {task_id}] Failed to load metadata: {e}")
        import traceback
        traceback.print_exc()
        return task_id, [], 0


def process_task_iteratively(
    task_id: int,
    data_dir: Path,
    cache_dir: Path,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    trajectory_offset: int,
    output_dir: Path,
    args: argparse.Namespace,
    download: bool = False,
    cleanup: bool = True
) -> Tuple[List, List, List]:
    """Process a single task: download, extract features, and cleanup.

    Args:
        task_id: Task ID to process
        data_dir: Root data directory for extracted files
        cache_dir: HuggingFace cache directory
        train_indices: Global train indices
        val_indices: Global validation indices
        test_indices: Global test indices
        trajectory_offset: Offset to map global indices to task-specific indices
        output_dir: Output directory for processed data
        args: Command line arguments
        download: Whether to download the task
        cleanup: Whether to cleanup downloaded data after processing

    Returns:
        Tuple of (train_processed, val_processed, test_processed) for this task
    """
    # Download and extract task
    success, episode_data_cache = download_and_extract_task(
        data_dir, task_id, cache_dir, download
    )

    if not success:
        print(f"Failed to download/extract task {task_id}, skipping...")
        return [], [], []

    # Load trajectories using extracted files
    trajectories = load_task_trajectories(data_dir, task_id, episode_data_cache)

    if len(trajectories) == 0:
        print(f"No trajectories loaded for task {task_id}, skipping...")
        return [], [], []

    # Get episode IDs for cleanup
    episode_ids = [traj.metadata.get('episode_id') for traj in trajectories if traj.metadata]

    # Map global indices to task-specific indices
    num_traj = len(trajectories)
    task_range = range(trajectory_offset, trajectory_offset + num_traj)

    # Find which indices belong to this task
    task_train_indices = [i - trajectory_offset for i in train_indices if i in task_range]
    task_val_indices = [i - trajectory_offset for i in val_indices if i in task_range]
    task_test_indices = [i - trajectory_offset for i in test_indices if i in task_range]

    print(f"\nTask {task_id} split counts:")
    print(f"  Train: {len(task_train_indices)}")
    print(f"  Val: {len(task_val_indices)}")
    print(f"  Test: {len(task_test_indices)}")

    # Build processed datasets for this task
    if args.build_dataset and (task_train_indices or task_val_indices or task_test_indices):
        from data.dataset_builder import DatasetBuilder

        # Create dataset builder
        task_output_dir = output_dir / f'task_{task_id}_processed'
        builder = DatasetBuilder(
            output_dir=task_output_dir,
            extract_dino=args.extract_dino,
            extract_frames=args.extract_frames,
            dino_model=args.dino_model,
            target_fps=args.target_fps,
            num_workers=args.num_workers
        )

        # Process each split for this task
        train_processed = []
        val_processed = []
        test_processed = []

        if task_train_indices:
            print(f"\nProcessing train split for task {task_id}...")
            train_processed = builder.build_dataset(trajectories, 'train', task_train_indices)

        if task_val_indices:
            print(f"\nProcessing validation split for task {task_id}...")
            val_processed = builder.build_dataset(trajectories, 'val', task_val_indices)

        if task_test_indices:
            print(f"\nProcessing test split for task {task_id}...")
            test_processed = builder.build_dataset(trajectories, 'test', task_test_indices)

        # Cleanup downloaded data if requested
        if cleanup:
            cleanup_task_data(data_dir, cache_dir, task_id, episode_ids)

        return train_processed, val_processed, test_processed

    else:
        # Cleanup even if not building dataset
        if cleanup:
            cleanup_task_data(data_dir, cache_dir, task_id, episode_ids)

        return [], [], []


def process_task_with_builder(
    task_id: int,
    data_dir: Path,
    cache_dir: Path,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    trajectory_offset: int,
    shared_builder,  # Shared RayDatasetBuilder to avoid nested Ray
    download: bool,
    cleanup: bool
) -> Tuple[int, List, List, List]:
    """Process a single task using shared dataset builder (NO nested Ray).

    Args:
        task_id: Task ID to process
        data_dir: Root data directory for extracted files
        cache_dir: HuggingFace cache directory
        train_indices: Global train indices
        val_indices: Global validation indices
        test_indices: Global test indices
        trajectory_offset: Offset to map global indices to task-specific indices
        shared_builder: Shared DatasetBuilder instance (avoids nested Ray initialization)
        download: Whether to download the task
        cleanup: Whether to cleanup downloaded data after processing

    Returns:
        Tuple of (task_id, train_processed, val_processed, test_processed)
    """
    print(f"\n[Task {task_id}] Starting processing...")

    # Download and extract task
    success, episode_data_cache = download_and_extract_task(
        data_dir, task_id, cache_dir, download
    )

    if not success:
        print(f"[Task {task_id}] Failed to download/extract, skipping...")
        return task_id, [], [], []

    # Load trajectories using extracted files
    trajectories = load_task_trajectories(data_dir, task_id, episode_data_cache)

    if len(trajectories) == 0:
        print(f"[Task {task_id}] No trajectories loaded, skipping...")
        return task_id, [], [], []

    # Get episode IDs for cleanup
    episode_ids = [traj.metadata.get('episode_id') for traj in trajectories if traj.metadata]

    # Map global indices to task-specific indices
    num_traj = len(trajectories)
    task_range = range(trajectory_offset, trajectory_offset + num_traj)

    # Find which indices belong to this task
    task_train_indices = [i - trajectory_offset for i in train_indices if i in task_range]
    task_val_indices = [i - trajectory_offset for i in val_indices if i in task_range]
    task_test_indices = [i - trajectory_offset for i in test_indices if i in task_range]

    print(f"[Task {task_id}] Split counts: Train={len(task_train_indices)}, Val={len(task_val_indices)}, Test={len(task_test_indices)}")

    # Build processed datasets for this task using SHARED builder
    train_processed = []
    val_processed = []
    test_processed = []

    if shared_builder and (task_train_indices or task_val_indices or task_test_indices):
        if task_train_indices:
            print(f"[Task {task_id}] Processing train split...")
            train_processed = shared_builder.build_dataset(trajectories, 'train', task_train_indices)

        if task_val_indices:
            print(f"[Task {task_id}] Processing validation split...")
            val_processed = shared_builder.build_dataset(trajectories, 'val', task_val_indices)

        if task_test_indices:
            print(f"[Task {task_id}] Processing test split...")
            test_processed = shared_builder.build_dataset(trajectories, 'test', task_test_indices)

    # Cleanup downloaded data if requested
    if cleanup:
        cleanup_task_data(data_dir, cache_dir, task_id, episode_ids)

    print(f"[Task {task_id}] Completed! Processed: Train={len(train_processed)}, Val={len(val_processed)}, Test={len(test_processed)}")
    return task_id, train_processed, val_processed, test_processed


def main():
    """Main entry point with parallel task processing using Ray."""
    args = parse_args()

    # Set random seed
    set_all_seeds(args.seed)

    # Initialize Ray with GPUs
    if not ray.is_initialized():
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
        print(f"Ray initialized with {ray.cluster_resources().get('CPU', 0)} CPUs and {num_gpus} GPUs")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace cache directory
    cache_dir = output_dir / 'hf_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover all task IDs if none specified
    if args.task_ids is None:
        print("\n⚠️  No task IDs specified, discovering all available tasks...")
        task_ids = discover_all_task_ids()

        if not task_ids:
            print("\nError: Could not discover task IDs automatically.")
            print("Please specify task IDs manually with --task-ids, for example:")
            print("  python scripts/prepare_data.py --task-ids 327 352 357 --download")
            sys.exit(1)

        # Ask user for confirmation
        print(f"\n{'='*80}")
        print(f"DISCOVERED {len(task_ids)} TASKS")
        print(f"{'='*80}")
        print(f"Task IDs: {task_ids}")
        print(f"{'='*80}")

        if args.download:
            total_tasks = len(task_ids)
            print(f"\n⚠️  WARNING: You are about to download and process {total_tasks} tasks.")
            print("This will require significant time and disk space.")
            print("Each task can be several GB (videos + proprioception data).")
            print(f"Download enabled: {args.download}")
            print(f"Cleanup enabled: {not args.no_cleanup}")

            response = input("\nProceed with downloading all tasks? (y/n): ").strip().lower()
            if response != 'y':
                print("Operation cancelled.")
                sys.exit(0)

        args.task_ids = task_ids
    else:
        task_ids = args.task_ids

    print(f"\n{'='*80}")
    print("DATA PREPARATION WITH RAY (GPU-OPTIMIZED)")
    print(f"{'='*80}")
    print(f"Strategy: Sequential task processing, parallel trajectory processing")
    print(f"Task IDs: {task_ids}")
    print(f"Number of tasks: {len(task_ids)}")
    print(f"Download enabled: {args.download}")
    print(f"Cleanup enabled: {not args.no_cleanup}")
    print(f"Data dir: {data_dir}")
    print(f"Cache dir: {cache_dir}")
    print(f"GPU Parallelism: Trajectory-level (all GPUs utilized efficiently)")
    print(f"{'='*80}\n")

    # Step 1: Load trajectory metadata for all tasks in parallel
    print("Step 1: Loading trajectory metadata in parallel for split creation...")

    # Launch metadata loading tasks in parallel
    metadata_refs = [
        load_task_metadata_remote.remote(task_id, data_dir, cache_dir, args.download)
        for task_id in task_ids
    ]

    # Wait for all metadata to load
    print(f"Loading metadata for {len(task_ids)} tasks in parallel...")
    metadata_results = ray.get(metadata_refs)

    # Aggregate results
    all_trajectories = []
    task_trajectory_counts = {}

    for task_id, trajectories, count in metadata_results:
        if count > 0:
            all_trajectories.extend(trajectories)
            task_trajectory_counts[task_id] = count
            print(f"  Task {task_id}: {count} trajectories")
        else:
            print(f"  Task {task_id}: skipped (no metadata)")

    if len(all_trajectories) == 0:
        print("Error: No trajectories loaded!")
        ray.shutdown()
        sys.exit(1)

    print(f"\nTotal trajectories across all tasks: {len(all_trajectories)}")

    # Limit number of trajectories if requested
    if args.max_trajectories is not None and args.max_trajectories < len(all_trajectories):
        print(f"Limiting trajectories from {len(all_trajectories)} to {args.max_trajectories}")
        all_trajectories = all_trajectories[:args.max_trajectories]

    # Step 2: Create splits across all trajectories
    print("\nStep 2: Creating data splits...")
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_task=args.stratify_by_task
    )

    splitter = DatasetSplitter(split_config)

    # Get task IDs from trajectories
    task_ids_list = []
    for t in all_trajectories:
        if hasattr(t, 'task_id'):
            task_ids_list.append(t.task_id)
        else:
            task_ids_list.append(0)

    # Split trajectories into indices
    train_indices, val_indices, test_indices = splitter.split_trajectories(
        all_trajectories, task_ids_list
    )

    # Save split manifests
    create_splits(all_trajectories, split_config, output_dir, args.split_name)

    print(f"\nGlobal split statistics:")
    print(f"  Train: {len(train_indices)} trajectories")
    print(f"  Val: {len(val_indices)} trajectories")
    print(f"  Test: {len(test_indices)} trajectories")

    # Step 3: Process tasks sequentially with SHARED builder (avoids nested Ray)
    if args.build_dataset:
        print(f"\n{'='*80}")
        print("Step 3: Processing tasks sequentially with shared Ray builder...")
        print(f"{'='*80}")
        print("NOTE: Task-level parallelism removed to avoid nested Ray initialization.")
        print("      GPU parallelism happens at trajectory-level (using all 8 GPUs efficiently)")

        # Create ONE shared DatasetBuilder for all tasks
        # This avoids nested Ray initialization and allows all GPUs to be used
        from data.dataset_builder import DatasetBuilder
        shared_builder = DatasetBuilder(
            output_dir=output_dir / 'processed_dataset',
            extract_dino=args.extract_dino,
            extract_frames=args.extract_frames,
            dino_model=args.dino_model,
            target_fps=args.target_fps,
            num_workers=args.num_workers  # Will auto-detect and use all 8 GPUs
        )

        print(f"Shared builder initialized with {shared_builder.ray_processor.num_workers} workers")
        print(f"This will utilize ALL available GPUs efficiently\n")

        # Accumulate processed datasets
        all_train_processed = []
        all_val_processed = []
        all_test_processed = []

        # Track trajectory offset for index mapping
        trajectory_offset = 0
        task_offsets = {}
        for task_id in task_ids:
            if task_id in task_trajectory_counts:
                task_offsets[task_id] = trajectory_offset
                trajectory_offset += task_trajectory_counts[task_id]

        # Filter tasks that have metadata
        tasks_to_process = [tid for tid in task_ids if tid in task_trajectory_counts]

        if not tasks_to_process:
            print("No tasks to process!")
        else:
            print(f"Processing {len(tasks_to_process)} tasks sequentially...")

            # Process tasks sequentially (but trajectories in parallel via shared_builder)
            for idx, task_id in enumerate(tasks_to_process):
                print(f"\n{'='*60}")
                print(f"Processing task {idx+1}/{len(tasks_to_process)}: Task {task_id}")
                print(f"{'='*60}")

                # Process task with shared builder
                task_id, train_proc, val_proc, test_proc = process_task_with_builder(
                    task_id=task_id,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    test_indices=test_indices,
                    trajectory_offset=task_offsets[task_id],
                    shared_builder=shared_builder,
                    download=args.download,
                    cleanup=not args.no_cleanup
                )

                # Aggregate results
                all_train_processed.extend(train_proc)
                all_val_processed.extend(val_proc)
                all_test_processed.extend(test_proc)

                print(f"✓ Task {task_id} completed: Train={len(train_proc)}, Val={len(val_proc)}, Test={len(test_proc)}")
                print(f"Cumulative progress: Train={len(all_train_processed)}, "
                      f"Val={len(all_val_processed)}, Test={len(all_test_processed)}")

        # Step 4: Merge all processed datasets into final splits
        print(f"\n{'='*80}")
        print("Step 4: Merging processed datasets...")
        print(f"{'='*80}")

        # Save merged splits using the shared builder
        if all_train_processed:
            print("\nSaving merged train split...")
            shared_builder._save_split(all_train_processed, 'train')

        if all_val_processed:
            print("\nSaving merged validation split...")
            shared_builder._save_split(all_val_processed, 'val')

        if all_test_processed:
            print("\nSaving merged test split...")
            shared_builder._save_split(all_test_processed, 'test')

        print(f"\n{'='*80}")
        print("DATASET BUILD COMPLETE")
        print(f"{'='*80}")
        print(f"\nFinal statistics:")
        print(f"  Train: {len(all_train_processed)} trajectories")
        print(f"  Val: {len(all_val_processed)} trajectories")
        print(f"  Test: {len(all_test_processed)} trajectories")
        print(f"\nProcessed datasets saved to: {output_dir / 'processed_dataset'}")

    print("\n✓ Data preparation complete!")

    # Shutdown Ray
    ray.shutdown()
    print("Ray shutdown complete")


if __name__ == '__main__':
    main()
