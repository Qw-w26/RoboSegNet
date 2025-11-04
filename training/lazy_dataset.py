"""Lazy-loading dataset for faster initialization.

This is an optimized version of BoundaryDataset that:
- Only builds an index during __init__ (fast)
- Loads and processes data on-demand in __getitem__ (parallelized by workers)
- Caches tokenized features to avoid recomputation

Use this for large multi-task datasets where preprocessing everything upfront is slow.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import numpy as np

from features.droid_extractor import DroidTokenEmbedder, FastTokenizerConfig
from training.label_smoothing import create_smoother
from training.training_dataset import Window, TrainingDatasetConfig, collate_dict


class LazyBoundaryDataset(Dataset):
    """Lazy-loading dataset for boundary prediction.

    Only builds an index during init, loads data on-demand in __getitem__.
    Much faster initialization for large datasets.

    Args:
        trajectories: List of trajectory data
        config: Dataset configuration
    """

    def __init__(
        self,
        trajectories: List[Any],
        config: TrainingDatasetConfig
    ):
        self.trajectories = trajectories
        self.config = config

        # Detect frequencies from first trajectory
        self.window_fps = config.fps

        if len(trajectories) > 0:
            timestamps = trajectories[0].proprio.timestamps
            # Convert timestamps to seconds
            if timestamps[0] > 1e12:
                timestamps = timestamps / 1e9
            elif timestamps[0] > 1e9:
                timestamps = timestamps / 1e6
            elif timestamps[0] > 1e6:
                timestamps = timestamps / 1e3

            dt = np.diff(timestamps)
            median_dt = np.median(dt)
            self.proprio_fps = float(1.0 / median_dt) if median_dt > 0 else 30.0
        else:
            self.proprio_fps = config.fps

        self.freq_multiplier = int(round(self.proprio_fps / self.window_fps))

        # Calculate window parameters
        time_window_s = config.window_size / self.window_fps
        tokenizer_frame_freq = int(round(self.proprio_fps))
        self.proprio_window_size = int(time_window_s * tokenizer_frame_freq)

        if self.proprio_window_size % tokenizer_frame_freq != 0:
            self.proprio_window_size = (self.proprio_window_size // tokenizer_frame_freq) * tokenizer_frame_freq

        # Initialize feature extractors (lightweight)
        self.proprio_embedder = DroidTokenEmbedder(
            config=FastTokenizerConfig(
                time_s=time_window_s,
                frame_frequency=tokenizer_frame_freq,
                action_dim=config.action_dim,
                vocab_size=config.vocab_size
            )
        )

        self.label_smoother = create_smoother(
            config.label_smoother_type,
            sigma=config.label_smoother_sigma,
            peak_value=config.label_smoother_peak_value
        )

        # Build index only (fast - no feature extraction)
        self._build_index()

        print(f"✓ Created lazy dataset with {len(self.index)} samples from {len(self.trajectories)} trajectories")
        print(f"  Window fps: {self.window_fps} Hz, Proprio fps: {self.proprio_fps:.2f} Hz")
        print(f"  Features will be extracted on-demand by DataLoader workers")

    def _build_index(self) -> None:
        """Build index of (trajectory_idx, window) pairs without loading features."""
        self.index = []

        for traj_idx, traj in enumerate(self.trajectories):
            # Check max samples
            if self.config.max_samples and len(self.index) >= self.config.max_samples:
                break

            # Get DINO features to determine number of frames
            dino_features = traj.dino_features
            if dino_features is None or len(dino_features) == 0:
                continue

            num_frames = len(dino_features)

            # Create windows
            for i in range(0, num_frames - self.config.window_size, self.config.stride):
                window = Window(
                    begin_frame=i,
                    middle_frame=i + self.config.window_size // 2,
                    end_frame=i + self.config.window_size
                )

                self.index.append((traj_idx, window))

                if self.config.max_samples and len(self.index) >= self.config.max_samples:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and process sample on-demand."""
        traj_idx, window = self.index[idx]
        traj = self.trajectories[traj_idx]

        # Get metadata
        task_id = getattr(traj, 'task_id', 0)
        episode_id = getattr(traj, 'episode_id', 0)

        # Get DINO features
        dino_features_array = traj.dino_features
        if dino_features_array.dtype == object:
            dino_stacked = np.stack(dino_features_array)
        else:
            dino_stacked = dino_features_array

        # Sample DINO frames
        frame_indices = np.linspace(
            window.begin_frame,
            window.end_frame - 1,
            self.config.num_dino_frames,
            dtype=int
        )

        dino_dim = dino_stacked.shape[1] if len(dino_stacked.shape) > 1 else 1024
        dino_frame_features = []
        for frame_idx in frame_indices:
            if frame_idx < len(dino_stacked):
                dino_feat = dino_stacked[frame_idx]
            else:
                dino_feat = np.zeros(dino_dim)
            dino_frame_features.append(torch.tensor(dino_feat, dtype=torch.float32))

        # Extract proprio features
        proprio = traj.proprio
        begin_proprio = window.begin_frame * self.freq_multiplier

        # Use cached tokenization if available
        cache_key = (traj_idx, begin_proprio)
        if not hasattr(self, '_proprio_cache'):
            self._proprio_cache = {}

        if cache_key not in self._proprio_cache:
            proprio_tokens = self.proprio_embedder.embed_frames(
                proprio.joint_position,
                [begin_proprio],
                self.proprio_window_size
            )[begin_proprio]
            self._proprio_cache[cache_key] = proprio_tokens
        else:
            proprio_tokens = self._proprio_cache[cache_key]

        # Make labels
        intervals = getattr(traj, 'intervals', [])
        timestamps = proprio.timestamps

        # Convert timestamps to seconds
        if timestamps[0] > 1e12:
            timestamps = timestamps / 1e9
        elif timestamps[0] > 1e9:
            timestamps = timestamps / 1e6
        elif timestamps[0] > 1e6:
            timestamps = timestamps / 1e3

        transition_indices = [interval.start_frame for interval in intervals[1:]]

        if len(transition_indices) == 0:
            labels = np.zeros(self.config.window_size)
        else:
            smoothed_labels = self.label_smoother.smooth(timestamps, transition_indices)

            begin_proprio = window.begin_frame * self.freq_multiplier
            end_proprio = begin_proprio + self.proprio_window_size

            if end_proprio <= len(smoothed_labels):
                labels_at_proprio = smoothed_labels[begin_proprio:end_proprio]
            else:
                available = smoothed_labels[begin_proprio:]
                labels_at_proprio = np.zeros(self.proprio_window_size)
                labels_at_proprio[:len(available)] = available

            labels_at_window = labels_at_proprio[::self.freq_multiplier]

            if len(labels_at_window) < self.config.window_size:
                labels = np.zeros(self.config.window_size)
                labels[:len(labels_at_window)] = labels_at_window
            else:
                labels = labels_at_window[:self.config.window_size]

        return {
            'episode_id': episode_id,
            'task_id': task_id,
            'window': window,
            'dino_features': dino_frame_features,
            'proprio_features': torch.tensor(proprio_tokens, dtype=torch.long),
            'train_labels': torch.tensor(labels, dtype=torch.float32),
            'intervals': intervals,
        }


# Export collate function
__all__ = ['LazyBoundaryDataset', 'collate_dict']
