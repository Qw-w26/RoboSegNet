"""Clean parquet data loader for processed trajectory data.

This module provides a simple, clean interface for loading trajectory data from
the parquet files created by prepare_data.py. It handles all the data format
conversions and provides an interface compatible with BoundaryDataset.

Example:
    >>> from data import load_parquet_split
    >>>
    >>> # Load training data
    >>> trajectories = load_parquet_split('train', splits_dir='./splits')
    >>> print(f"Loaded {len(trajectories)} trajectories")
    >>>
    >>> # Use with BoundaryDataset
    >>> from training import BoundaryDataset, TrainingDatasetConfig
    >>> config = TrainingDatasetConfig()
    >>> dataset = BoundaryDataset(trajectories, config)
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Segment:
    """Task segment annotation with frame boundaries.

    Attributes:
        start_frame: Start frame index
        end_frame: End frame index (exclusive)
        action_text: Natural language description of the action
        skill: Skill label (e.g., 'Pick', 'Place', 'Pull')
        segment_id: Unique segment identifier within trajectory
    """
    start_frame: int
    end_frame: int
    action_text: str
    skill: Optional[str] = None
    segment_id: Optional[int] = None

    @property
    def duration(self) -> int:
        """Duration of segment in frames."""
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start_frame,
            'end': self.end_frame,
            'action_text': self.action_text,
            'skill': self.skill,
            'segment_id': self.segment_id
        }


class ProprioData:
    """Wrapper for proprioceptive data to match BoundaryDataset interface."""

    def __init__(self, joint_positions: np.ndarray, timestamps: np.ndarray):
        """Initialize proprioceptive data.

        Args:
            joint_positions: Joint positions array of shape (T, D)
            timestamps: Timestamps array of shape (T,)
        """
        self.joint_position = joint_positions
        self.timestamps = timestamps


class ParquetTrajectory:
    """Trajectory wrapper for processed parquet data.

    This class wraps a row from a processed parquet file and provides the
    interface expected by BoundaryDataset._parse_trajectory().

    Attributes:
        task_id: Task ID
        episode_id: Episode ID
        trajectory_id: Unique trajectory identifier
        task_name: Task name/description
        videos_paths: Dictionary mapping camera name to video path
        proprio: ProprioData object with joint positions and timestamps
        intervals: List of Segment objects
        total_frames: Total number of frames in trajectory
        fps: Frames per second
        dino_features: Pre-extracted DINO features (if available)
        image_paths: Paths to extracted frame images (if available)
        metadata: Additional metadata dictionary
    """

    def __init__(self, parquet_row):
        """Initialize from parquet row.

        Args:
            parquet_row: Row from pandas DataFrame
        """
        # Basic identifiers
        self.task_id = int(parquet_row['task_id'])
        self.episode_id = int(parquet_row['episode_id'])
        self.trajectory_id = str(parquet_row['trajectory_id'])
        self.task_name = str(parquet_row['task_name'])

        # Parse video paths from JSON string to dict
        video_paths_str = parquet_row['video_paths']
        if isinstance(video_paths_str, str):
            # Handle empty strings
            if video_paths_str and video_paths_str.strip():
                self.videos_paths = ast.literal_eval(video_paths_str)
            else:
                self.videos_paths = {}
        else:
            self.videos_paths = video_paths_str if video_paths_str else {}

        # Process proprioception data
        proprio_array = parquet_row['proprioception']
        timestamps_array = parquet_row['proprio_timestamps']

        # Fix: Stack proprioception if it's an array of arrays (object dtype)
        if proprio_array.dtype == object:
            # Array of arrays - stack them
            proprio_array = np.stack(proprio_array)

        # Create ProprioData wrapper
        self.proprio = ProprioData(
            joint_positions=proprio_array,
            timestamps=timestamps_array
        )

        # Parse segments/intervals
        segments_array = parquet_row['segments']
        self.intervals = []

        for seg_dict in segments_array:
            segment = Segment(
                start_frame=seg_dict['start_frame'],
                end_frame=seg_dict['end_frame'],
                action_text=seg_dict['action_text'],
                skill=seg_dict.get('skill', None),
                segment_id=seg_dict.get('segment_id', None)
            )
            self.intervals.append(segment)

        # Metadata
        self.total_frames = int(parquet_row['total_frames'])
        self.fps = float(parquet_row['fps'])
        self.metadata = parquet_row.get('metadata', {})

        # Pre-extracted features (if available from prepare_data.py)
        self.dino_features = parquet_row.get('dino_features', None)
        self.image_paths = parquet_row.get('image_paths', None)

    def __repr__(self) -> str:
        return (f"ParquetTrajectory(trajectory_id='{self.trajectory_id}', "
                f"task_id={self.task_id}, episode_id={self.episode_id}, "
                f"segments={len(self.intervals)})")


def load_parquet_split(
    split_name: str,
    splits_dir: str = './splits',
    processed_dir: Optional[str] = None
) -> List[ParquetTrajectory]:
    """Load a data split from processed parquet files.

    Args:
        split_name: Name of split ('train', 'val', or 'test')
        splits_dir: Directory containing the splits
        processed_dir: Directory with processed parquet files
                      (defaults to splits_dir/processed_dataset)

    Returns:
        List of ParquetTrajectory objects

    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If split_name is invalid

    Example:
        >>> trajectories = load_parquet_split('train', splits_dir='./splits')
        >>> print(f"Loaded {len(trajectories)} training trajectories")
        >>>
        >>> # Access trajectory data
        >>> traj = trajectories[0]
        >>> print(f"Task: {traj.task_name}")
        >>> print(f"Proprio shape: {traj.proprio.joint_position.shape}")
        >>> print(f"Segments: {len(traj.intervals)}")
    """
    # Validate split name
    valid_splits = {'train', 'val', 'test'}
    if split_name not in valid_splits:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be one of {valid_splits}")

    # Determine processed directory
    if processed_dir is None:
        processed_dir = Path(splits_dir) / 'processed_dataset'
    else:
        processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found: {processed_dir}\n"
            f"Please run prepare_data.py with --build-dataset flag first."
        )

    # Construct parquet file path
    parquet_file = processed_dir / f"{split_name}_processed.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_file}\n"
            f"Please run prepare_data.py with --build-dataset flag first."
        )

    # Load parquet file
    print(f"Loading {split_name} split from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # Convert each row to ParquetTrajectory
    trajectories = []
    for _, row in df.iterrows():
        traj = ParquetTrajectory(row)
        trajectories.append(traj)

    print(f"✓ Loaded {len(trajectories)} trajectories from {split_name} split")

    return trajectories


def get_split_statistics(trajectories: List[ParquetTrajectory]) -> Dict[str, Any]:
    """Get statistics about a split.

    Args:
        trajectories: List of trajectories

    Returns:
        Dictionary with statistics

    Example:
        >>> trajectories = load_parquet_split('train')
        >>> stats = get_split_statistics(trajectories)
        >>> print(f"Total frames: {stats['total_frames']}")
    """
    total_frames = sum(t.total_frames for t in trajectories)
    total_segments = sum(len(t.intervals) for t in trajectories)

    task_ids = set(t.task_id for t in trajectories)

    segment_durations = []
    for t in trajectories:
        segment_durations.extend([s.duration for s in t.intervals])

    stats = {
        'num_trajectories': len(trajectories),
        'total_frames': total_frames,
        'total_segments': total_segments,
        'unique_tasks': len(task_ids),
        'task_ids': sorted(task_ids),
        'avg_frames_per_trajectory': total_frames / len(trajectories) if trajectories else 0,
        'avg_segments_per_trajectory': total_segments / len(trajectories) if trajectories else 0,
        'avg_segment_duration': np.mean(segment_durations) if segment_durations else 0,
        'min_segment_duration': np.min(segment_durations) if segment_durations else 0,
        'max_segment_duration': np.max(segment_durations) if segment_durations else 0,
    }

    return stats


def print_split_statistics(trajectories: List[ParquetTrajectory]) -> None:
    """Print formatted statistics about a split.

    Args:
        trajectories: List of trajectories

    Example:
        >>> trajectories = load_parquet_split('train')
        >>> print_split_statistics(trajectories)
    """
    stats = get_split_statistics(trajectories)

    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    print(f"Trajectories:              {stats['num_trajectories']}")
    print(f"Total frames:              {stats['total_frames']:,}")
    print(f"Total segments:            {stats['total_segments']}")
    print(f"Unique tasks:              {stats['unique_tasks']} {stats['task_ids']}")
    print(f"\nAvg frames/trajectory:     {stats['avg_frames_per_trajectory']:.1f}")
    print(f"Avg segments/trajectory:   {stats['avg_segments_per_trajectory']:.1f}")
    print(f"Avg segment duration:      {stats['avg_segment_duration']:.1f} frames")
    print(f"Segment duration range:    [{stats['min_segment_duration']}, {stats['max_segment_duration']}]")
    print("="*80 + "\n")


__all__ = [
    'Segment',
    'ProprioData',
    'ParquetTrajectory',
    'load_parquet_split',
    'get_split_statistics',
    'print_split_statistics',
]
