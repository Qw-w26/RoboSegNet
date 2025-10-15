"""
Base class for task segmentation methods.

This module provides an abstract interface that all segmentation methods must implement
to be compatible with the benchmark framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path


class BaseSegmentationMethod(ABC):
    """
    Abstract base class for task segmentation methods.

    All segmentation methods should inherit from this class and implement
    the required methods to be compatible with the benchmark framework.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the segmentation method.

        Args:
            name: Human-readable name for the method
            **kwargs: Method-specific hyperparameters
        """
        self.name = name
        self.hyperparameters = kwargs

    @abstractmethod
    def fit(self, trajectories: list, embeddings: Optional[np.ndarray] = None):
        """
        Fit the segmentation method on training data.

        Args:
            trajectories: List of TrajectoryData objects
            embeddings: Optional pre-computed embeddings (N_windows, embedding_dim)

        Note:
            Some methods may not require training, in which case this can be a no-op.
        """
        pass

    @abstractmethod
    def predict(self, trajectory_data: Any, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict segment labels for a single trajectory.

        Args:
            trajectory_data: TrajectoryData object or raw trajectory data
            embeddings: Optional pre-computed embeddings for this trajectory

        Returns:
            Frame-level segment labels (num_frames,) where each frame gets a segment ID
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get method configuration.

        Returns:
            Dictionary with method name and hyperparameters
        """
        return {
            'method_name': self.name,
            'hyperparameters': self.hyperparameters
        }

    def save(self, save_path: Path):
        """
        Save method state (weights, models, etc.) to disk.

        Args:
            save_path: Path to save the method
        """
        # Default implementation: save config only
        import json
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    def load(self, load_path: Path):
        """
        Load method state from disk.

        Args:
            load_path: Path to load the method from
        """
        # Default implementation: load config only
        import json
        load_path = Path(load_path)
        config_path = load_path / 'config.json'

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.hyperparameters = config.get('hyperparameters', {})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class WindowBasedSegmentationMethod(BaseSegmentationMethod):
    """
    Base class for window-based segmentation methods.

    These methods operate on sliding windows of trajectory data and require
    embeddings to be pre-computed.
    """

    def __init__(self, name: str, window_size: int = 15, stride: int = 1, **kwargs):
        """
        Initialize window-based segmentation method.

        Args:
            name: Method name
            window_size: Size of the sliding window
            stride: Stride for the sliding window
            **kwargs: Additional hyperparameters
        """
        super().__init__(name, window_size=window_size, stride=stride, **kwargs)
        self.window_size = window_size
        self.stride = stride

    def create_windows(self, trajectory: np.ndarray) -> list:
        """
        Create sliding windows from a trajectory.

        Args:
            trajectory: Trajectory data (num_timesteps, feature_dim)

        Returns:
            List of windows
        """
        windows = []
        n_timesteps = len(trajectory)

        for start_idx in range(0, n_timesteps - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = trajectory[start_idx:end_idx]
            windows.append(window)

        # Handle last window if necessary
        if n_timesteps > self.window_size and (n_timesteps - self.window_size) % self.stride != 0:
            window = trajectory[-self.window_size:]
            windows.append(window)

        return windows

    def window_labels_to_frame_labels(
        self,
        window_labels: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Convert window-level labels to frame-level labels.

        Args:
            window_labels: Labels for each window (num_windows,)
            num_frames: Total number of frames in trajectory

        Returns:
            Frame-level labels (num_frames,)
        """
        frame_labels = np.zeros(num_frames, dtype=int)
        vote_counts = np.zeros((num_frames, len(np.unique(window_labels))), dtype=int)

        window_idx = 0
        for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            if window_idx < len(window_labels):
                label = window_labels[window_idx]
                vote_counts[start_idx:end_idx, label] += 1
            window_idx += 1

        # Handle last window if necessary
        if num_frames > self.window_size and (num_frames - self.window_size) % self.stride != 0:
            if window_idx < len(window_labels):
                label = window_labels[window_idx]
                vote_counts[-self.window_size:, label] += 1

        # Assign labels based on majority voting
        frame_labels = np.argmax(vote_counts, axis=1)

        return frame_labels
