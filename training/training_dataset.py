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
        num_dino_frames: Number of DINO frames to sample per window (evenly spaced)

    Example:
        >>> config = TrainingDatasetConfig(
        ...     window_size=180,
        ...     stride=10,
        ...     vocab_size=2048,
        ...     num_dino_frames=3
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

    # DINO features
    num_dino_frames: int = 3  # Number of DINO frames to sample per window

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

        # Detect actual proprioception frequency from first trajectory
        # config.fps is the window/DINO fps (e.g., 2 Hz)
        # actual_proprio_fps is the raw proprio frequency (e.g., 30 Hz)
        self.window_fps = config.fps  # For window creation and DINO indexing

        if len(trajectories) > 0:
            timestamps = trajectories[0].proprio.timestamps
            # Convert timestamps to seconds if needed
            if timestamps[0] > 1e12:  # Nanoseconds
                timestamps = timestamps / 1e9
            elif timestamps[0] > 1e9:  # Microseconds
                timestamps = timestamps / 1e6
            elif timestamps[0] > 1e6:  # Milliseconds
                timestamps = timestamps / 1e3

            dt = np.diff(timestamps)
            median_dt = np.median(dt)
            self.proprio_fps = float(1.0 / median_dt) if median_dt > 0 else 30.0
            print(f"Detected proprioception frequency: {self.proprio_fps:.2f} Hz")
        else:
            self.proprio_fps = config.fps  # Fallback
            print(f"Warning: No trajectories to detect frequency, using config fps: {self.proprio_fps} Hz")

        # Calculate frequency multiplier for mapping window indices to proprio indices
        self.freq_multiplier = int(round(self.proprio_fps / self.window_fps))
        print(f"Window fps: {self.window_fps} Hz, Proprio fps: {self.proprio_fps:.2f} Hz, Multiplier: {self.freq_multiplier}x")

        # Calculate actual time covered by window
        time_window_s = config.window_size / self.window_fps
        print(f"Time window: {config.window_size} frames at {self.window_fps} Hz = {time_window_s:.1f} seconds")

        # Calculate frame frequency for tokenizer (round to nearest integer)
        # The tokenizer requires that window_size = time_s * frame_frequency
        tokenizer_frame_freq = int(round(self.proprio_fps))
        print(f"Tokenizer frame frequency: {tokenizer_frame_freq} Hz (rounded from {self.proprio_fps:.2f} Hz)")

        # Calculate expected proprio samples per window that's compatible with tokenizer
        # Must be: window_size = time_s * frame_frequency (exactly)
        self.proprio_window_size = int(time_window_s * tokenizer_frame_freq)
        print(f"Proprio samples per window: {self.proprio_window_size} ({time_window_s:.1f}s * {tokenizer_frame_freq}Hz)")

        # Verify this is valid
        if self.proprio_window_size % tokenizer_frame_freq != 0:
            # Adjust to be a multiple
            self.proprio_window_size = (self.proprio_window_size // tokenizer_frame_freq) * tokenizer_frame_freq
            print(f"  Adjusted to nearest multiple: {self.proprio_window_size}")

        # Initialize feature extractors
        self.proprio_embedder = DroidTokenEmbedder(
            config=FastTokenizerConfig(
                time_s=time_window_s,  # Time in seconds
                frame_frequency=tokenizer_frame_freq,  # Use rounded frequency
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
        # Get DINO features first
        dino_features = self.dino_features.get(task_id, {}).get(episode_id, {})

        # Create windows at DINO fps (window_fps), not at proprio fps
        # The number of DINO samples determines how many windows we can create
        num_dino_samples = len(dino_features)
        windows = self._make_windows(num_dino_samples)

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
            # Extract evenly spaced DINO frames across the window
            # Note: window.end_frame is exclusive, so use end_frame - 1
            # For num_dino_frames=3 and window_size=18: [0, 8, 17]
            # For num_dino_frames=5 and window_size=18: [0, 4, 8, 13, 17]
            frame_indices = np.linspace(
                window.begin_frame,
                window.end_frame - 1,  # end_frame is exclusive
                self.config.num_dino_frames,
                dtype=int
            )

            # Extract DINO features for each sampled frame
            dino_frame_features = []
            for frame_idx in frame_indices:
                dino_feat = dino_features.get(frame_idx, np.zeros(dino_dim))
                dino_frame_features.append(torch.tensor(dino_feat, dtype=torch.float32))

            samples.append({
                'episode_id': episode_id,
                'task_id': task_id,
                'window': window,
                'dino_features': dino_frame_features,  # List of tensors
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
        """Extract proprioceptive features for windows.

        Windows are at window_fps (DINO fps, e.g., 2 Hz), but we need to extract
        proprio data at its native frequency (e.g., 30 Hz). We map window indices
        by multiplying by the frequency multiplier.
        """
        # Window begin_frames are at window_fps, map to proprio_fps
        begin_frames_at_window_fps = [window.begin_frame for window in windows]
        begin_frames_at_proprio_fps = [f * self.freq_multiplier for f in begin_frames_at_window_fps]

        # Extract proprio with mapped window size
        # self.proprio_window_size is window_size * freq_multiplier
        features = self.proprio_embedder.embed_frames(
            proprio_data['joint_positions'],  # Proprio at native frequency
            begin_frames_at_proprio_fps,  # Mapped start indices
            self.proprio_window_size  # Mapped window size
        )

        # Return with original window indices as keys (at window_fps)
        # This allows DINO features and proprio features to align
        return {
            begin_frames_at_window_fps[i]: features[begin_frames_at_proprio_fps[i]]
            for i in range(len(windows))
        }

    def _make_train_labels(
        self,
        timestamps: np.ndarray,
        intervals: List[Any],
        windows: List[Window]
    ) -> Dict[int, np.ndarray]:
        """Create training labels with smoothing.

        Labels should be at window_fps (DINO fps) to match model predictions,
        not at proprio_fps. Each label represents the boundary probability at
        a window frame (2 Hz), not at every proprio sample.
        """
        # Extract boundary frame indices (start of each interval except first)
        # These are at the original proprio frequency
        transition_indices = [interval.start_frame for interval in intervals[1:]]

        # Handle case with no transitions
        if len(transition_indices) == 0:
            return {
                window.begin_frame: np.zeros(self.config.window_size)
                for window in windows
            }

        # Apply label smoothing (labels are at proprio frequency)
        smoothed_labels = self.label_smoother.smooth(timestamps, transition_indices)

        # Extract labels for each window
        # Downsample from proprio_fps to window_fps
        labels_dict = {}
        for window in windows:
            # Map window frames to proprio frames
            begin_proprio = window.begin_frame * self.freq_multiplier
            end_proprio = begin_proprio + self.proprio_window_size

            # Extract labels at proprio frequency
            if end_proprio <= len(smoothed_labels):
                labels_at_proprio_fps = smoothed_labels[begin_proprio:end_proprio]
            else:
                # Pad if out of bounds
                available = smoothed_labels[begin_proprio:]
                padded = np.zeros(self.proprio_window_size)
                padded[:len(available)] = available
                labels_at_proprio_fps = padded

            # Downsample to window_fps by taking every freq_multiplier-th sample
            # This gives us window_size labels (one per window frame at 2 Hz)
            labels_at_window_fps = labels_at_proprio_fps[::self.freq_multiplier]

            # Ensure we have exactly window_size labels
            if len(labels_at_window_fps) < self.config.window_size:
                # Pad if needed
                padded = np.zeros(self.config.window_size)
                padded[:len(labels_at_window_fps)] = labels_at_window_fps
                labels_dict[window.begin_frame] = padded
            else:
                labels_dict[window.begin_frame] = labels_at_window_fps[:self.config.window_size]

        return labels_dict


def collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for dict-based batches.

    Handles dino_features as a list of tensors that get properly stacked and flattened.
    Also renames keys to match model expectations:
    - proprio_features -> tokenized_actions
    - train_labels -> label
    """
    if len(batch) == 0:
        return {}

    # Get all keys from first sample
    keys = batch[0].keys()

    collated = {}
    for key in keys:
        # Handle dino_features (list of tensors)
        if key == 'dino_features':
            # Stack each frame across batch, then concatenate along feature dim
            # Input: batch of [tensor(1024), tensor(1024), ...] (num_frames tensors)
            # Output: (batch_size, num_frames * 1024)
            num_frames = len(batch[0][key])
            # Stack each frame: [(batch, 1024), (batch, 1024), ...]
            stacked_frames = [
                torch.stack([sample[key][i] for sample in batch])
                for i in range(num_frames)
            ]
            # Concatenate along feature dimension: (batch, num_frames * 1024)
            collated[key] = torch.cat(stacked_frames, dim=1)
        # Handle tensors
        elif isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        # Handle lists
        else:
            collated[key] = [sample[key] for sample in batch]

    # Rename keys to match model expectations
    # Keep both names for compatibility with different parts of the codebase
    if 'proprio_features' in collated:
        collated['tokenized_actions'] = collated['proprio_features']
    if 'train_labels' in collated:
        collated['label'] = collated['train_labels']

    return collated
