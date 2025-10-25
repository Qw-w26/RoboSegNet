"""Training dataset for boundary prediction with feature caching.

This module provides a PyTorch Dataset that:
- Extracts DINO visual features with Ray parallelization
- Tokenizes proprioceptive actions with Fast-DROID
- Applies label smoothing to boundary annotations
- Caches features for fast reloading
- Creates sliding windows for temporal context

Ported and refactored from andrew's excellent training_dataset.py.

Example:
    >>> from training import TrainingDatasetConfig, BoundaryDataset
    >>> from data.splits import load_split_manifest
    >>>
    >>> # Load data split
    >>> split = load_split_manifest('splits/task_327_train.json')
    >>>
    >>> # Create dataset
    >>> config = TrainingDatasetConfig(window_size=180, stride=10)
    >>> dataset = BoundaryDataset(split.trajectories, config)
    >>>
    >>> # Use with DataLoader
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=512, collate_fn=collate_dict)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import hashlib
import json
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import ray

from features.dino_extractor import DinoFeatureExtractor, RayDinoFeatureExtractor
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
        num_workers: Number of Ray workers for parallel DINO extraction
        device: Device for feature extraction ('cuda' or 'cpu')
        ray_worker_device: Device for Ray workers (defaults to device)
        cache_path: Path to cache directory for features (None to disable)
        label_smoother_type: Type of label smoother ('gaussian', 'triangular', etc.)
        label_smoother_sigma: Sigma parameter for Gaussian smoother
        label_smoother_peak_value: Peak value for smoother
        action_dim: Action dimension for tokenizer
        max_samples: Maximum number of samples to load (None for all)

    Example:
        >>> config = TrainingDatasetConfig(
        ...     window_size=180,
        ...     stride=10,
        ...     vocab_size=2048,
        ...     cache_path='./cache'
        ... )
    """

    # Window and stride settings
    window_size: int = 180
    stride: int = 10

    # Vocabulary and tokenization
    vocab_size: int = 2048
    action_dim: int = 14

    # Parallel processing
    num_workers: int = 4
    device: str = 'cuda'
    ray_worker_device: Optional[str] = None

    # Caching
    cache_path: Optional[str] = None

    # Label smoothing
    label_smoother_type: str = 'gaussian'
    label_smoother_sigma: float = 5.0
    label_smoother_peak_value: float = 1.0

    # Data limits
    max_samples: Optional[int] = None

    def __post_init__(self):
        """Set default values."""
        if self.ray_worker_device is None:
            self.ray_worker_device = self.device


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
                time_s=config.window_size / 30.0,  # Assuming 30 fps
                frame_frequency=30.0,
                action_dim=config.action_dim,
                vocab_size=config.vocab_size
            )
        )

        self.dino_embedder = DinoFeatureExtractor(device=config.device)

        # Initialize label smoother
        self.label_smoother = create_smoother(
            config.label_smoother_type,
            sigma=config.label_smoother_sigma,
            peak_value=config.label_smoother_peak_value
        )

        # Initialize cache directories if caching is enabled
        if config.cache_path:
            self.dino_cache_dir = Path(config.cache_path) / "dino_features"
            self.proprio_cache_dir = Path(config.cache_path) / "proprio_features"
            self.dino_cache_dir.mkdir(parents=True, exist_ok=True)
            # Disable proprio cache for now (same as andrew's code)
            # self.proprio_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.dino_cache_dir = None
            self.proprio_cache_dir = None

        # Initialize Ray and process trajectories
        self._initialize_ray()
        self._process_all_trajectories()
        self._cleanup_ray()

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data_points)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample."""
        return self.data_points[index]

    def _initialize_ray(self) -> None:
        """Initialize Ray for parallel DINO extraction."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create Ray actors for DINO extraction
        self.dino_actors = [
            RayDinoFeatureExtractor.remote(device=self.config.ray_worker_device)
            for _ in range(self.config.num_workers)
        ]

        print(f"Initialized {self.config.num_workers} Ray workers for DINO extraction")

    def _cleanup_ray(self) -> None:
        """Cleanup Ray actors and free GPU memory."""
        # Kill Ray actors
        for actor in self.dino_actors:
            ray.kill(actor)

        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _process_all_trajectories(self) -> None:
        """Process all trajectories with Ray parallelization and caching."""
        # Storage for DINO features
        self.dino_futures = {}
        self.dino_features = {}

        # Phase 1: Submit DINO extraction tasks (with caching)
        print("Phase 1: Submitting DINO extraction tasks...")
        num_datapoints = 0

        for i, traj in enumerate(tqdm(self.trajectories, desc="Assigning DINO tasks")):
            # Check if we've hit max samples
            if self.config.max_samples and num_datapoints >= self.config.max_samples:
                break

            # Get trajectory metadata
            task_id, episode_id, video_path, proprio_data, intervals = self._parse_trajectory(traj)

            # Count datapoints this trajectory will produce
            windows = self._make_windows(len(proprio_data['timestamps']))
            num_datapoints += len(windows)

            # Check cache for DINO features
            if self._is_dino_cached(video_path, windows):
                cached_features = self._load_dino_from_cache(video_path, windows)
                if cached_features is not None:
                    # Use cached features
                    if task_id not in self.dino_features:
                        self.dino_features[task_id] = {}
                    self.dino_features[task_id][episode_id] = cached_features
                    continue

            # Submit Ray task if not cached
            if task_id not in self.dino_futures:
                self.dino_futures[task_id] = {}

            future = self._submit_dino_extraction(
                video_path,
                windows,
                actor=self.dino_actors[i % self.config.num_workers]
            )
            self.dino_futures[task_id][episode_id] = future

        print(f"Submitted tasks for ~{num_datapoints} datapoints")

        # Phase 2: Collect DINO extraction results
        print("Phase 2: Collecting DINO extraction results...")
        num_datapoints = 0

        for traj in tqdm(self.trajectories, desc="Waiting for DINO tasks"):
            # Check if we've hit max samples
            if self.config.max_samples and num_datapoints >= self.config.max_samples:
                break

            task_id, episode_id, video_path, proprio_data, intervals = self._parse_trajectory(traj)

            windows = self._make_windows(len(proprio_data['timestamps']))
            num_datapoints += len(windows)

            # Skip if already cached
            if task_id in self.dino_features and episode_id in self.dino_features[task_id]:
                continue

            # Wait for Ray result
            if task_id in self.dino_futures and episode_id in self.dino_futures[task_id]:
                future = self.dino_futures[task_id][episode_id]
                dino_features = ray.get(future)

                if task_id not in self.dino_features:
                    self.dino_features[task_id] = {}
                self.dino_features[task_id][episode_id] = dino_features

                # Save to cache
                self._save_dino_to_cache(video_path, windows, dino_features)

        # Phase 3: Assemble windows with all features
        print("Phase 3: Assembling windows with features...")
        num_datapoints = 0

        for traj in tqdm(self.trajectories, desc="Assembling windows"):
            # Check if we've hit max samples
            if self.config.max_samples and num_datapoints >= self.config.max_samples:
                break

            task_id, episode_id, video_path, proprio_data, intervals = self._parse_trajectory(traj)

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

        print(f"Created dataset with {len(self.data_points)} samples")

    def _parse_trajectory(self, traj: Any) -> Tuple[int, int, str, Dict, List]:
        """Parse trajectory data into standard format.

        Override this method to support different trajectory data formats.

        Returns:
            Tuple of (task_id, episode_id, video_path, proprio_data, intervals)
        """
        # Default implementation - assumes andrew's AgibotDatapoint format
        # Override for different formats
        task_id = getattr(traj, 'task_id', 0)
        episode_id = getattr(traj, 'episode_id', 0)
        video_path = getattr(traj, 'videos_paths', {}).get('head_color', '')
        proprio = getattr(traj, 'proprio', None)
        intervals = getattr(traj, 'intervals', [])

        # Convert proprio to dict format
        proprio_data = {
            'joint_positions': proprio.joint_position if proprio else None,
            'timestamps': proprio.timestamps if proprio else None
        }

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

    def _is_dino_cached(self, video_path: str, windows: List[Window]) -> bool:
        """Check if DINO features are cached."""
        if not self.dino_cache_dir or not video_path:
            return False

        if not Path(video_path).exists():
            return False

        cache_file = self._get_dino_cache_file(video_path, windows)
        return cache_file.exists()

    def _load_dino_from_cache(self, video_path: str, windows: List[Window]) -> Optional[Dict]:
        """Load DINO features from cache."""
        cache_file = self._get_dino_cache_file(video_path, windows)
        return self._load_from_cache(cache_file)

    def _save_dino_to_cache(
        self,
        video_path: str,
        windows: List[Window],
        features: Dict
    ) -> None:
        """Save DINO features to cache."""
        if self.dino_cache_dir and video_path and Path(video_path).exists():
            cache_file = self._get_dino_cache_file(video_path, windows)
            self._save_to_cache(cache_file, features)

    def _get_dino_cache_file(self, video_path: str, windows: List[Window]) -> Path:
        """Get cache file path for DINO features."""
        # Extract unique frame indices from windows
        interest_frames = set()
        for window in windows:
            interest_frames.update([window.begin_frame, window.middle_frame, window.end_frame])

        cache_key = self._get_dino_cache_key(video_path, sorted(list(interest_frames)))
        return self.dino_cache_dir / f"{cache_key}.pkl"

    def _get_dino_cache_key(self, video_path: str, frame_indices: List[int]) -> str:
        """Generate cache key for DINO features."""
        key_data = {
            'video_path': str(video_path),
            'frame_indices': sorted(frame_indices),
            'window_size': self.config.window_size,
            'stride': self.config.stride
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load data from cache file."""
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError, FileNotFoundError):
                cache_file.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, cache_file: Path, data: Dict) -> None:
        """Save data to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except (OSError, IOError) as e:
            warnings.warn(f"Could not save to cache {cache_file}: {e}")

    def _submit_dino_extraction(
        self,
        video_path: str,
        windows: List[Window],
        actor: Any
    ) -> Any:
        """Submit DINO extraction task to Ray actor."""
        # Extract unique frame indices
        interest_frames = set()
        for window in windows:
            interest_frames.update([window.begin_frame, window.middle_frame, window.end_frame])

        interest_frames = sorted(list(interest_frames))

        # Submit task
        return actor.extract_batch.remote(video_path, interest_frames)

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
                    'begin_frame': torch.tensor(dino_features.get(window.begin_frame, np.zeros(384)), dtype=torch.float32),
                    'middle_frame': torch.tensor(dino_features.get(window.middle_frame, np.zeros(384)), dtype=torch.float32),
                    'end_frame': torch.tensor(dino_features.get(window.end_frame, np.zeros(384)), dtype=torch.float32),
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

        # Convert to format expected by embedder
        # This assumes proprio_data has 'joint_positions' and 'timestamps'
        features = self.proprio_embedder.embed_frames(
            proprio_data,
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
