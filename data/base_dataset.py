"""
Base dataset interface for task segmentation benchmarks.

This module provides abstract base classes for datasets that support
task/action segmentation with ground truth annotations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path


@dataclass
class SegmentAnnotation:
    """Represents a single segment annotation."""
    start_frame: int
    end_frame: int
    action_text: str
    skill: Optional[str] = None
    segment_id: Optional[int] = None

    @property
    def duration(self) -> int:
        """Returns the duration of the segment in frames."""
        return self.end_frame - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'action_text': self.action_text,
            'skill': self.skill,
            'segment_id': self.segment_id
        }


@dataclass
class TrajectoryData:
    """Represents a single trajectory/episode with segmentation annotations."""
    trajectory_id: str
    task_id: Optional[str] = None
    task_name: Optional[str] = None

    # Raw data paths
    proprioception_path: Optional[Path] = None
    video_paths: Optional[Dict[str, Path]] = None  # e.g., {'wrist': path, 'exo': path}

    # Loaded data (lazy loaded)
    _proprioception: Optional[np.ndarray] = None
    _video_frames: Optional[Dict[str, np.ndarray]] = None

    # Segmentation annotations
    segments: List[SegmentAnnotation] = None

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = []

    @property
    def num_segments(self) -> int:
        """Returns number of segments in this trajectory."""
        return len(self.segments)

    @property
    def total_frames(self) -> int:
        """Returns total number of frames in the trajectory."""
        if self.segments:
            return max(seg.end_frame for seg in self.segments)
        return 0

    def load_proprioception(self) -> np.ndarray:
        """Lazy load proprioceptive data."""
        if self._proprioception is None and self.proprioception_path:
            self._proprioception = self._load_proprioception_impl()
        return self._proprioception

    def _load_proprioception_impl(self) -> np.ndarray:
        """Implementation-specific loading. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement proprioception loading")

    def get_segment_labels(self) -> np.ndarray:
        """
        Convert segment annotations to frame-level labels.

        Returns:
            Array of shape (total_frames,) with segment IDs for each frame.
        """
        if not self.segments:
            return np.array([])

        labels = np.zeros(self.total_frames, dtype=int)
        for seg in self.segments:
            labels[seg.start_frame:seg.end_frame] = seg.segment_id

        return labels

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'trajectory_id': self.trajectory_id,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'num_segments': self.num_segments,
            'total_frames': self.total_frames,
            'segments': [seg.to_dict() for seg in self.segments],
            'metadata': self.metadata
        }


class BaseSegmentationDataset(ABC):
    """
    Abstract base class for task segmentation datasets.

    This interface provides a common API for different segmentation datasets
    and enables consistent benchmarking across datasets.
    """

    def __init__(self, data_root: Path, split: str = 'train'):
        """
        Initialize the dataset.

        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_root = Path(data_root)
        self.split = split
        self._trajectories = None

    @abstractmethod
    def _load_trajectories(self) -> List[TrajectoryData]:
        """
        Load trajectory data and annotations.

        Returns:
            List of TrajectoryData objects
        """
        pass

    @property
    def trajectories(self) -> List[TrajectoryData]:
        """Get all trajectories (lazy loaded)."""
        if self._trajectories is None:
            self._trajectories = self._load_trajectories()
        return self._trajectories

    def __len__(self) -> int:
        """Returns number of trajectories in the dataset."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> TrajectoryData:
        """Get trajectory by index."""
        return self.trajectories[idx]

    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[TrajectoryData]:
        """Get trajectory by ID."""
        for traj in self.trajectories:
            if traj.trajectory_id == trajectory_id:
                return traj
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics (num_trajectories, avg_segments, etc.)
        """
        stats = {
            'num_trajectories': len(self),
            'total_segments': sum(traj.num_segments for traj in self.trajectories),
            'avg_segments_per_trajectory': np.mean([traj.num_segments for traj in self.trajectories]),
            'std_segments_per_trajectory': np.std([traj.num_segments for traj in self.trajectories]),
            'total_frames': sum(traj.total_frames for traj in self.trajectories),
            'avg_frames_per_trajectory': np.mean([traj.total_frames for traj in self.trajectories]),
        }

        # Segment duration statistics
        all_durations = []
        for traj in self.trajectories:
            all_durations.extend([seg.duration for seg in traj.segments])

        if all_durations:
            stats['avg_segment_duration'] = np.mean(all_durations)
            stats['std_segment_duration'] = np.std(all_durations)
            stats['min_segment_duration'] = np.min(all_durations)
            stats['max_segment_duration'] = np.max(all_durations)

        return stats

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"Dataset Statistics - {self.__class__.__name__} ({self.split})")
        print(f"{'='*60}")
        print(f"Number of trajectories: {stats['num_trajectories']}")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Avg segments per trajectory: {stats['avg_segments_per_trajectory']:.2f} ± {stats['std_segments_per_trajectory']:.2f}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Avg frames per trajectory: {stats['avg_frames_per_trajectory']:.2f}")
        if 'avg_segment_duration' in stats:
            print(f"\nSegment Duration Statistics:")
            print(f"  Average: {stats['avg_segment_duration']:.2f} frames")
            print(f"  Std Dev: {stats['std_segment_duration']:.2f} frames")
            print(f"  Min: {stats['min_segment_duration']} frames")
            print(f"  Max: {stats['max_segment_duration']} frames")
        print(f"{'='*60}\n")
