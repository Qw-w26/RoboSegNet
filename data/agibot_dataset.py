"""
AgiBotWorld dataset loader for task segmentation.

This module provides a loader for the AgiBotWorld-Alpha dataset from HuggingFace.
Dataset: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

Supports both local loading and streaming directly from HuggingFace Hub.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import tempfile
import io

from .base_dataset import (
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation
)


class AgiBotTrajectoryData(TrajectoryData):
    """AgiBotWorld-specific trajectory data."""

    def _load_proprioception_impl(self) -> np.ndarray:
        """Load proprioceptive data from H5 file."""
        if not self.proprioception_path or not self.proprioception_path.exists():
            return np.array([])

        try:
            with h5py.File(self.proprioception_path, 'r') as f:
                # Try different possible data layouts
                # Andrew's format uses 'state/joint/position' and 'state/effector/position'
                if 'state' in f:
                    if 'joint' in f['state'] and 'position' in f['state']['joint']:
                        # Andrew's format - return joint positions
                        data = f['state']['joint']['position'][:]
                    else:
                        data = f['state'][:]
                elif 'proprio' in f:
                    data = f['proprio'][:]
                elif 'timestamp' in f:
                    # Another possible format with timestamp and state data
                    if 'state' in f:
                        data = f['state'][:]
                    else:
                        # Just return timestamps if no state data
                        data = f['timestamp'][:]
                else:
                    # Try to get the first dataset
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                    else:
                        data = np.array([])

            return data
        except Exception as e:
            print(f"Warning: Failed to load proprioception from {self.proprioception_path}: {e}")
            return np.array([])


class AgiBotWorldDataset(BaseSegmentationDataset):
    """
    AgiBotWorld-Alpha dataset loader.

    Dataset Structure:
        task_info/
            task_[id].json  - Contains episode metadata and segmentation annotations
        observations/
            [task_id]/
                [episode_id]/
                    *.mp4  - Video observations
        parameters/
            camera_params.json
        proprio_stats/
            [task_id]/
                [episode_id].h5  - Proprioceptive data

    JSON Structure (task_info/task_[id].json):
    [
        {
            "episode_id": 649078,
            "task_id": 327,
            "task_name": "Picking items in Supermarket",
            "init_scene_text": "The robot is in front of the fruit shelf...",
            "lable_info": {
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": 435,
                        "action_text": "Pick up onion from the shelf.",
                        "skill": "Pick"
                    },
                    ...
                ]
            }
        },
        ...
    ]
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        split: str = 'train',
        task_ids: Optional[List[int]] = None,
        max_trajectories: Optional[int] = None,
        load_video: bool = False,
        streaming: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize AgiBotWorld dataset.

        Args:
            data_root: Root directory of the dataset (required if streaming=False)
            split: Dataset split ('train', 'val', 'test')
            task_ids: Optional list of task IDs to load (loads all if None)
            max_trajectories: Maximum number of trajectories to load
            load_video: Whether to load video data (can be memory intensive)
            streaming: If True, stream data directly from HuggingFace Hub
            hf_token: HuggingFace token for accessing private datasets
        """
        self.task_ids = task_ids
        self.max_trajectories = max_trajectories
        self.load_video = load_video
        self.streaming = streaming
        self.hf_token = hf_token
        self.hf_dataset = None

        if streaming:
            # Initialize HuggingFace datasets for streaming
            try:
                from datasets import load_dataset
                print("Loading dataset from HuggingFace Hub in streaming mode...")
                self.hf_dataset = load_dataset(
                    "agibot-world/AgiBotWorld-Alpha",
                    split=split,
                    streaming=True,
                    token=hf_token
                )
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets library is required for streaming. "
                    "Install it with: pip install datasets"
                )
            except Exception as e:
                print(f"Warning: Failed to load dataset in streaming mode: {e}")
                print("Falling back to local loading...")
                self.streaming = False

        if not streaming and data_root is None:
            raise ValueError("data_root is required when streaming=False")

        data_root = Path(data_root) if data_root else Path(".")
        super().__init__(data_root, split)

    def _load_trajectories(self) -> List[TrajectoryData]:
        """Load trajectories from AgiBotWorld dataset."""
        if self.streaming:
            return self._load_trajectories_streaming()
        else:
            return self._load_trajectories_local()

    def _load_trajectories_local(self) -> List[TrajectoryData]:
        """Load trajectories from local filesystem."""
        trajectories = []

        # Try two different directory structures
        # 1. HuggingFace format: task_info/ at root
        # 2. Andrew's format: task_*/task_info/task_*.json

        json_files = []

        # First try HuggingFace format
        task_info_dir = self.data_root / 'task_info'
        if task_info_dir.exists():
            # Get all task JSON files
            if self.task_ids:
                json_files = [task_info_dir / f"task_{tid}.json" for tid in self.task_ids]
                json_files = [f for f in json_files if f.exists()]
            else:
                json_files = sorted(task_info_dir.glob("task_*.json"))

        # If no files found, try Andrew's format
        if not json_files:
            print(f"No task_info directory at root, checking task subdirectories...")
            if self.task_ids:
                for tid in self.task_ids:
                    task_dir = self.data_root / f'task_{tid}' / 'task_info' / f'task_{tid}.json'
                    if task_dir.exists():
                        json_files.append(task_dir)
            else:
                # Find all task_*/task_info/task_*.json files
                for task_dir in sorted(self.data_root.glob("task_*")):
                    task_info_path = task_dir / 'task_info'
                    if task_info_path.exists():
                        task_json_files = list(task_info_path.glob("task_*.json"))
                        json_files.extend(task_json_files)

        if not json_files:
            raise ValueError(f"No task JSON files found in {self.data_root}")

        print(f"Loading AgiBotWorld dataset from {len(json_files)} task files...")

        # Load trajectories from each task file
        for json_file in tqdm(json_files, desc="Loading tasks"):
            task_trajectories = self._load_task_file(json_file)
            trajectories.extend(task_trajectories)

            # Check if we've reached the maximum
            if self.max_trajectories and len(trajectories) >= self.max_trajectories:
                trajectories = trajectories[:self.max_trajectories]
                break

        print(f"Loaded {len(trajectories)} trajectories")
        return trajectories

    def _load_trajectories_streaming(self) -> List[TrajectoryData]:
        """Load trajectories by streaming from HuggingFace Hub."""
        trajectories = []

        print("Streaming trajectories from HuggingFace Hub...")

        try:
            # Iterate through streaming dataset
            count = 0
            for example in tqdm(self.hf_dataset, desc="Streaming trajectories"):
                # Filter by task_id if specified
                if self.task_ids and example.get('task_id') not in self.task_ids:
                    continue

                # Parse the example into TrajectoryData
                trajectory = self._parse_hf_example(example)
                if trajectory:
                    trajectories.append(trajectory)
                    count += 1

                # Check if we've reached the maximum
                if self.max_trajectories and count >= self.max_trajectories:
                    break

            print(f"Loaded {len(trajectories)} trajectories from HuggingFace Hub")
        except Exception as e:
            print(f"Error streaming from HuggingFace: {e}")
            print("Consider downloading the dataset locally or checking your internet connection.")

        return trajectories

    def _parse_hf_example(self, example: Dict[str, Any]) -> Optional[TrajectoryData]:
        """
        Parse a HuggingFace dataset example into TrajectoryData.

        Args:
            example: Dictionary from HuggingFace dataset

        Returns:
            TrajectoryData object or None if parsing fails
        """
        try:
            # Extract basic info
            episode_id = example.get('episode_id')
            task_id = example.get('task_id')
            task_name = example.get('task_name', '')
            init_scene_text = example.get('init_scene_text', '')

            # Parse segmentation annotations (handle both 'label_info' and 'lable_info' typo)
            segments = []
            label_info = example.get('label_info') or example.get('lable_info')
            if label_info:
                action_configs = label_info.get('action_config', [])

                for seg_id, action in enumerate(action_configs):
                    segment = SegmentAnnotation(
                        start_frame=action['start_frame'],
                        end_frame=action['end_frame'],
                        action_text=action['action_text'],
                        skill=action.get('skill'),
                        segment_id=seg_id
                    )
                    segments.append(segment)

            # Create trajectory data (streaming mode doesn't have local paths)
            trajectory = AgiBotTrajectoryData(
                trajectory_id=f"{task_id}_{episode_id}",
                task_id=str(task_id),
                task_name=task_name,
                proprioception_path=None,  # Not available in streaming mode
                video_paths=None,  # Not available in streaming mode
                segments=segments,
                metadata={
                    'episode_id': episode_id,
                    'init_scene_text': init_scene_text,
                    'streaming': True,
                    'hf_example': example  # Store raw example for potential future use
                }
            )

            return trajectory
        except Exception as e:
            print(f"Warning: Failed to parse example: {e}")
            return None

    def _load_task_file(self, json_path: Path) -> List[TrajectoryData]:
        """Load all episodes from a single task JSON file."""
        with open(json_path, 'r') as f:
            episodes = json.load(f)

        trajectories = []

        # Extract task_id from filename if not in episode data
        default_task_id = None
        if 'task_' in json_path.stem:
            try:
                default_task_id = int(json_path.stem.split('task_')[1])
            except (ValueError, IndexError):
                pass

        for episode in episodes:
            # Extract basic info
            episode_id = episode['episode_id']
            task_id = episode.get('task_id', default_task_id)

            if task_id is None:
                print(f"Warning: Could not determine task_id for episode {episode_id}")
                continue

            task_name = episode.get('task_name', '')
            init_scene_text = episode.get('init_scene_text', '')

            # Parse segmentation annotations (handle both 'label_info' and 'lable_info' typo)
            segments = []
            label_info = episode.get('label_info') or episode.get('lable_info')
            if label_info and 'action_config' in label_info:
                action_configs = label_info['action_config']

                for seg_id, action in enumerate(action_configs):
                    segment = SegmentAnnotation(
                        start_frame=action['start_frame'],
                        end_frame=action['end_frame'],
                        action_text=action['action_text'],
                        skill=action.get('skill'),
                        segment_id=seg_id
                    )
                    segments.append(segment)

            # Find data paths - handle both directory structures
            # Try Andrew's format first: task_*/observations/episode_id
            obs_dir = self.data_root / f'task_{task_id}' / 'observations' / str(episode_id)
            proprio_path = self.data_root / 'proprio_stats' / str(task_id) / str(episode_id) / 'proprio_stats.h5'

            # If not found, try HuggingFace format: observations/task_id/episode_id
            if not obs_dir.exists():
                obs_dir = self.data_root / 'observations' / str(task_id) / str(episode_id)

            # If proprio not found, try without the extra proprio_stats.h5 name
            if not proprio_path.exists():
                proprio_path = self.data_root / 'proprio_stats' / str(task_id) / f"{episode_id}.h5"

            # Find video files
            video_paths = {}
            if obs_dir.exists():
                # Look for video files in the observations directory
                for video_file in obs_dir.glob("*.mp4"):
                    # Extract camera name from filename (e.g., 'wrist.mp4' -> 'wrist')
                    camera_name = video_file.stem
                    video_paths[camera_name] = video_file

                # Also check for videos in a videos subdirectory
                videos_dir = obs_dir / 'videos'
                if videos_dir.exists():
                    for video_file in videos_dir.glob("*.mp4"):
                        camera_name = video_file.stem
                        video_paths[camera_name] = video_file

            # Create trajectory data
            trajectory = AgiBotTrajectoryData(
                trajectory_id=f"{task_id}_{episode_id}",
                task_id=task_id,  # Keep as int
                task_name=task_name,
                proprioception_path=proprio_path if proprio_path.exists() else None,
                video_paths=video_paths if video_paths else None,
                segments=segments,
                metadata={
                    'episode_id': episode_id,
                    'init_scene_text': init_scene_text
                }
            )

            trajectories.append(trajectory)

        return trajectories

    def get_skill_distribution(self) -> Dict[str, int]:
        """Get distribution of skills across the dataset."""
        skill_counts = {}

        for traj in self.trajectories:
            for seg in traj.segments:
                if seg.skill:
                    skill_counts[seg.skill] = skill_counts.get(seg.skill, 0) + 1

        return skill_counts

    def print_statistics(self):
        """Print dataset statistics including skill distribution."""
        super().print_statistics()

        # Print skill distribution
        skill_dist = self.get_skill_distribution()
        if skill_dist:
            print(f"Skill Distribution:")
            for skill, count in sorted(skill_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {skill}: {count}")
            print()


def download_agibot_dataset(save_dir: Path, task_ids: Optional[List[int]] = None):
    """
    Download AgiBotWorld dataset from HuggingFace.

    Args:
        save_dir: Directory to save the dataset
        task_ids: Optional list of specific task IDs to download

    Note:
        Requires git-lfs to be installed:
        $ git lfs install

        Full dataset download:
        $ git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

        Specific task download:
        $ git sparse-checkout set observations/327 task_info/task_327.json
    """
    import subprocess

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading AgiBotWorld dataset from HuggingFace...")
    print("Note: This requires git-lfs to be installed")

    # Check if git-lfs is installed
    try:
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: git-lfs is not installed. Please install it first:")
        print("  Ubuntu/Debian: sudo apt-get install git-lfs")
        print("  Mac: brew install git-lfs")
        print("  Then run: git lfs install")
        return False

    if task_ids:
        # Sparse checkout for specific tasks
        print(f"Downloading specific tasks: {task_ids}")

        # Clone with sparse checkout
        subprocess.run([
            'git', 'clone', '--no-checkout',
            'https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha',
            str(save_dir)
        ], check=True)

        # Configure sparse checkout
        subprocess.run(
            ['git', 'sparse-checkout', 'init', '--cone'],
            cwd=save_dir,
            check=True
        )

        # Set paths to checkout
        paths = ['task_info', 'parameters']
        for task_id in task_ids:
            paths.extend([
                f'observations/{task_id}',
                f'proprio_stats/{task_id}',
                f'task_info/task_{task_id}.json'
            ])

        subprocess.run(
            ['git', 'sparse-checkout', 'set'] + paths,
            cwd=save_dir,
            check=True
        )

        # Checkout
        subprocess.run(['git', 'checkout'], cwd=save_dir, check=True)
    else:
        # Full clone
        print("Downloading full dataset (this may take a while)...")
        subprocess.run([
            'git', 'clone',
            'https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha',
            str(save_dir)
        ], check=True)

    print(f"Dataset downloaded to {save_dir}")
    return True


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Test AgiBotWorld dataset loader')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to AgiBotWorld dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download the dataset first')
    parser.add_argument('--task_ids', type=int, nargs='+',
                       help='Specific task IDs to load')
    parser.add_argument('--max_trajectories', type=int,
                       help='Maximum number of trajectories to load')

    args = parser.parse_args()

    # Download if requested
    if args.download:
        download_agibot_dataset(args.data_root, args.task_ids)

    # Load dataset
    dataset = AgiBotWorldDataset(
        data_root=args.data_root,
        task_ids=args.task_ids,
        max_trajectories=args.max_trajectories
    )

    # Print statistics
    dataset.print_statistics()

    # Show a few examples
    print("Sample trajectories:")
    for i in range(min(3, len(dataset))):
        traj = dataset[i]
        print(f"\nTrajectory {i}: {traj.trajectory_id}")
        print(f"  Task: {traj.task_name}")
        print(f"  Segments: {traj.num_segments}")
        print(f"  Total frames: {traj.total_frames}")
        print(f"  Segments:")
        for seg in traj.segments:
            print(f"    [{seg.start_frame}-{seg.end_frame}] {seg.action_text} ({seg.skill})")
