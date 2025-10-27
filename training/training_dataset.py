"""Training dataset for boundary prediction with pre-extracted features.

This module provides a PyTorch Dataset that:
- Loads pre-extracted DINO visual features from parquet files
- Tokenizes proprioceptive actions with Fast-DROID
- Applies label smoothing to boundary annotations
- Creates sliding windows for temporal context

Example:
    >>> from training import TrainingDatasetConfig, BoundaryDataset
    >>> from data import load_parquet_split
    >>>
    >>> # Load data split from parquet (must contain pre-extracted DINO features)
    >>> trajectories = load_parquet_split('train', splits_dir='./splits')
    >>>
    >>> # Create dataset
    >>> config = TrainingDatasetConfig(window_size=180, stride=10)
    >>> dataset = BoundaryDataset(trajectories, config)
    >>>
    >>> # Use with DataLoader
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=512, collate_fn=collate_dict)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from features.droid_extractor import DroidTokenEmbedder, FastTokenizerConfig
from training.label_smoothing import GaussianSmoother, create_smoother


@dataclass
class Window:
    """Sliding window for temporal context.

    Args:
        begin_frame: Start frame index
        middle_frame: Middle frame index
        end_frame: End frame index (exclusive)
    """
    begin_frame: int
    middle_frame: int
    end_frame: int


@dataclass
class TrainingDatasetConfig:
    """Configuration for boundary training dataset.

    Args:
        window_size: Window size for boundary prediction (in frames)
        stride: Stride for window sampling (in frames)
        vocab_size: Tokenizer vocabulary size
        action_dim: Action dimension for tokenizer
        device: Device for tokenization ('cuda' or 'cpu')
        label_smoother_type: Type of label smoother ('gaussian', 'triangular', etc.)
        label_smoother_sigma: Sigma parameter for Gaussian smoother
        label_smoother_peak_value: Peak value for smoother
        max_samples: Maximum number of samples to load (None for all)
        fps: Target FPS for data (should match preprocessing)

    Example:
        >>> config = TrainingDatasetConfig(
        ...     window_size=180,
        ...     stride=10,
        ...     vocab_size=2048
        ... )
    """

    # Window and stride settings
    window_size: int = 180
    stride: int = 10

    # Vocabulary and tokenization
    vocab_size: int = 2048
    action_dim: int = 14
    device: str = 'cuda'
    fps: float = 5.0  # Target FPS for data (should match preprocessing)

    # Label smoothing
    label_smoother_type: str = 'gaussian'
    label_smoother_sigma: float = 5.0
    label_smoother_peak_value: float = 1.0

    # Data limits
    max_samples: Optional[int] = None


class BoundaryDataset(Dataset):
    """PyTorch Dataset for boundary prediction training.

    Extracts and caches multi-modal features (DINO + proprioceptive) with
    sliding windows and label smoothing.

    Args:
        trajectories: List of trajectory data (from data loader or split manifest)
        config: Dataset configuration

    Example:
        >>> dataset = BoundaryDataset(trajectories, config)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """

    def __init__(
        self,
        trajectories: List[Any],
        config: TrainingDatasetConfig
    ):
        self.trajectories = trajectories
        self.config = config
        self.data_points = []

        # Initialize feature extractors
        self.proprio_embedder = DroidTokenEmbedder(
            config=FastTokenizerConfig(
                time_s=config.window_size / config.fps,
                frame_frequency=config.fps,
                action_dim=config.action_dim,
                vocab_size=config.vocab_size
            )
        )

        # Initialize label smoother
        self.label_smoother = create_smoother(
            config.label_smoother_type,
            sigma=config.label_smoother_sigma,
            peak_value=config.label_smoother_peak_value
        )

        # Process trajectories (load pre-extracted features from parquet)
        self._process_all_trajectories()

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data_points)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample."""
        return self.data_points[index]


    def _process_all_trajectories(self) -> None:
        """Process all trajectories loading pre-extracted features from parquet."""
        # Storage for pre-extracted DINO features
        self.dino_features = {}

        print("Loading pre-extracted features and assembling windows...")
        num_datapoints = 0

        for traj in tqdm(self.trajectories, desc="Processing trajectories"):
            # Check if we've hit max samples
            if self.config.max_samples and num_datapoints >= self.config.max_samples:
                break

            # Get trajectory metadata
            task_id, episode_id, video_path, proprio_data, intervals = self._parse_trajectory(traj)

            # Load pre-extracted DINO features from parquet
            if not hasattr(traj, 'dino_features') or traj.dino_features is None or len(traj.dino_features) == 0:
                raise ValueError(
                    f"Trajectory {task_id}/{episode_id} (trajectory_id: {traj.trajectory_id}) has no pre-extracted DINO features.\n"
                    f"This usually means the video files were missing during data preparation.\n"
                    f"Please regenerate the dataset with correct video paths:\n"
                    f"  python scripts/prepare_data.py --data-dir <path> --task-ids {task_id} --extract-dino\n"
                    f"\nMake sure the video files exist at the correct location before running prepare_data.py."
                )

            # Store DINO features for this trajectory
            if task_id not in self.dino_features:
                self.dino_features[task_id] = {}

            # DINO features are stored as an object array of (1024,) vectors
            # Convert to dict indexed by frame for easier lookup
            dino_array = traj.dino_features
            if dino_array.dtype == object:
                # Stack into 2D array (num_frames, 1024)
                if len(dino_array) == 0:
                    raise ValueError(
                        f"Trajectory {task_id}/{episode_id} has empty DINO features array. "
                        f"Please regenerate the dataset."
                    )
                dino_stacked = np.stack(dino_array)
            else:
                dino_stacked = dino_array

            # Store as dict: {frame_idx: feature_vector}
            self.dino_features[task_id][episode_id] = {
                i: dino_stacked[i] for i in range(len(dino_stacked))
            }

            missing_dino = sum(1 for i in range(len(dino_stacked)) if i not in self.dino_features[task_id][episode_id])
            if missing_dino > 0:
                print(f"Task {task_id}, Episode {episode_id}: Missing {missing_dino} DINO features")

            # Process trajectory into windows
            windows = self._preprocess_trajectory(
                task_id,
                episode_id,
                proprio_data,
                intervals
            )

            self.data_points.extend(windows)
            num_datapoints += len(windows)

            # Stop if we've hit max samples
            if self.config.max_samples and len(self.data_points) >= self.config.max_samples:
                self.data_points = self.data_points[:self.config.max_samples]
                break

        print(f"✓ Created dataset with {len(self.data_points)} samples from {len(self.trajectories)} trajectories")

    def _parse_trajectory(self, traj: Any) -> Tuple[int, int, str, Dict, List]:
        """Parse trajectory data into standard format.

        Supports ParquetTrajectory from parquet_loader.

        Returns:
            Tuple of (task_id, episode_id, video_path, proprio_data, intervals)
        """
        # Extract basic fields
        task_id = getattr(traj, 'task_id', 0)
        episode_id = getattr(traj, 'episode_id', 0)

        # Get video path - prefer head_color camera
        videos_paths = getattr(traj, 'videos_paths', {})
        video_path = videos_paths.get('head_color', '')

        # Extract proprioception
        proprio = getattr(traj, 'proprio', None)

        if proprio is None:
            raise ValueError(f"Trajectory {task_id}/{episode_id} has no proprioception data")

        # Convert proprio to dict format
        proprio_data = {
            'joint_positions': proprio.joint_position,
            'timestamps': proprio.timestamps
        }

        # Get intervals/segments
        intervals = getattr(traj, 'intervals', [])

        return task_id, episode_id, video_path, proprio_data, intervals

    def _make_windows(self, num_frames: int) -> List[Window]:
        """Create sliding windows over trajectory."""
        windows = []
        for i in range(0, num_frames - self.config.window_size, self.config.stride):
            windows.append(
                Window(
                    begin_frame=i,
                    middle_frame=i + self.config.window_size // 2,
                    end_frame=i + self.config.window_size
                )
            )
        return windows

    def _preprocess_trajectory(
        self,
        task_id: int,
        episode_id: int,
        proprio_data: Dict,
        intervals: List[Any]
    ) -> List[Dict[str, Any]]:
        """Preprocess trajectory into windowed samples."""
        # Get windows
        windows = self._make_windows(len(proprio_data['timestamps']))

        # Get DINO features
        dino_features = self.dino_features.get(task_id, {}).get(episode_id, {})

        # Detect DINO feature dimension from first available feature
        dino_dim = 1024  # Default to DINOv3
        if dino_features:
            first_feature = next(iter(dino_features.values()))
            dino_dim = len(first_feature)

        # Extract proprio features
        proprio_features = self._extract_proprio_features(proprio_data, windows)

        # Make training labels
        train_labels = self._make_train_labels(proprio_data['timestamps'], intervals, windows)

        # Create samples
        samples = []
        for window in windows:
            samples.append({
                'episode_id': episode_id,
                'task_id': task_id,
                'window': window,
                'dino_features': {
                    'begin_frame': torch.tensor(dino_features.get(window.begin_frame, np.zeros(dino_dim)), dtype=torch.float32),
                    'middle_frame': torch.tensor(dino_features.get(window.middle_frame, np.zeros(dino_dim)), dtype=torch.float32),
                    'end_frame': torch.tensor(dino_features.get(window.end_frame, np.zeros(dino_dim)), dtype=torch.float32),
                },
                'proprio_features': torch.tensor(proprio_features[window.begin_frame], dtype=torch.long),
                'train_labels': torch.tensor(train_labels[window.begin_frame], dtype=torch.float32),
                'intervals': intervals,
            })

        return samples

    def _extract_proprio_features(
        self,
        proprio_data: Dict,
        windows: List[Window]
    ) -> Dict[int, np.ndarray]:
        """Extract proprioceptive features for windows."""
        begin_frames = [window.begin_frame for window in windows]

        # FIXED: Pass joint_positions numpy array, not the full dict
        # embed_frames expects: (joint_positions: np.ndarray, frames: List[int], window_size: int)
        features = self.proprio_embedder.embed_frames(
            proprio_data['joint_positions'],  # Pass numpy array, not dict
            begin_frames,
            self.config.window_size
        )

        return features

    def _make_train_labels(
        self,
        timestamps: np.ndarray,
        intervals: List[Any],
        windows: List[Window]
    ) -> Dict[int, np.ndarray]:
        """Create training labels with smoothing."""
        # Extract boundary frame indices (start of each interval except first)
        transition_indices = [interval.start_frame for interval in intervals[1:]]

        # Handle case with no transitions
        if len(transition_indices) == 0:
            return {
                window.begin_frame: np.zeros(window.end_frame - window.begin_frame)
                for window in windows
            }

        # Apply label smoothing
        smoothed_labels = self.label_smoother.smooth(timestamps, transition_indices)

        # Extract labels for each window
        return {
            window.begin_frame: smoothed_labels[window.begin_frame:window.end_frame]
            for window in windows
        }


def collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for dict-based batches.

    Ported from andrew's code.
    """
    if len(batch) == 0:
        return {}

    # Get all keys from first sample
    keys = batch[0].keys()

    collated = {}
    for key in keys:
        # Handle nested dicts (like dino_features)
        if isinstance(batch[0][key], dict):
            collated[key] = {
                sub_key: torch.stack([sample[key][sub_key] for sample in batch])
                for sub_key in batch[0][key].keys()
            }
        # Handle tensors
        elif isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        # Handle lists
        else:
            collated[key] = [sample[key] for sample in batch]

    return collated
