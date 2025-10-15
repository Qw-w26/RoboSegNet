"""
Label smoothing strategies for segment boundary prediction.

This module provides extensible label smoothing methods to convert
binary boundary labels into smooth probability distributions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LabelSmoother(ABC):
    """Abstract base class for label smoothing strategies."""

    @abstractmethod
    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """
        Smooth binary labels into probability distributions.

        Args:
            labels: Binary labels (0 or 1) of shape (num_samples,)
            boundary_indices: Indices of ground truth boundaries

        Returns:
            Smoothed probability labels of shape (num_samples,)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        pass


class GaussianSmoother(LabelSmoother):
    """
    Gaussian smoothing around boundaries.

    Creates a smooth Gaussian distribution centered at each boundary.
    """

    def __init__(self, sigma: float = 2.0, peak_value: float = 1.0):
        """
        Initialize Gaussian smoother.

        Args:
            sigma: Standard deviation of Gaussian kernel
            peak_value: Maximum probability value at boundary center
        """
        self.sigma = sigma
        self.peak_value = peak_value

    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing around boundaries."""
        smoothed = np.zeros_like(labels, dtype=np.float32)

        for boundary_idx in boundary_indices:
            # Create Gaussian weights for all positions
            distances = np.abs(np.arange(len(labels)) - boundary_idx)
            weights = self.peak_value * np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

            # Add to smoothed labels (allowing overlap)
            smoothed = np.maximum(smoothed, weights)

        return smoothed

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'gaussian',
            'sigma': self.sigma,
            'peak_value': self.peak_value
        }


class TriangularSmoother(LabelSmoother):
    """
    Triangular (linear) smoothing around boundaries.

    Creates a triangular distribution centered at each boundary.
    """

    def __init__(self, width: int = 5, peak_value: float = 1.0):
        """
        Initialize triangular smoother.

        Args:
            width: Half-width of triangle (distance from center to zero)
            peak_value: Maximum probability value at boundary center
        """
        self.width = width
        self.peak_value = peak_value

    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """Apply triangular smoothing around boundaries."""
        smoothed = np.zeros_like(labels, dtype=np.float32)

        for boundary_idx in boundary_indices:
            # Create triangular weights
            distances = np.abs(np.arange(len(labels)) - boundary_idx)
            weights = np.maximum(
                0,
                self.peak_value * (1 - distances / self.width)
            )

            # Add to smoothed labels (allowing overlap)
            smoothed = np.maximum(smoothed, weights)

        return smoothed

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'triangular',
            'width': self.width,
            'peak_value': self.peak_value
        }


class UniformWindowSmoother(LabelSmoother):
    """
    Uniform smoothing within a window around boundaries.

    Creates a flat probability distribution within a window.
    """

    def __init__(self, window_size: int = 5, value: float = 1.0):
        """
        Initialize uniform window smoother.

        Args:
            window_size: Size of window (±window_size frames)
            value: Probability value within window
        """
        self.window_size = window_size
        self.value = value

    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """Apply uniform smoothing around boundaries."""
        smoothed = np.zeros_like(labels, dtype=np.float32)

        for boundary_idx in boundary_indices:
            # Set uniform value within window
            start = max(0, boundary_idx - self.window_size)
            end = min(len(labels), boundary_idx + self.window_size + 1)
            smoothed[start:end] = np.maximum(smoothed[start:end], self.value)

        return smoothed

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'uniform',
            'window_size': self.window_size,
            'value': self.value
        }


class ExponentialSmoother(LabelSmoother):
    """
    Exponential decay smoothing around boundaries.

    Creates exponentially decaying probabilities around boundaries.
    """

    def __init__(self, decay_rate: float = 0.5, peak_value: float = 1.0):
        """
        Initialize exponential smoother.

        Args:
            decay_rate: Rate of exponential decay
            peak_value: Maximum probability value at boundary center
        """
        self.decay_rate = decay_rate
        self.peak_value = peak_value

    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing around boundaries."""
        smoothed = np.zeros_like(labels, dtype=np.float32)

        for boundary_idx in boundary_indices:
            # Create exponential weights
            distances = np.abs(np.arange(len(labels)) - boundary_idx)
            weights = self.peak_value * np.exp(-self.decay_rate * distances)

            # Add to smoothed labels (allowing overlap)
            smoothed = np.maximum(smoothed, weights)

        return smoothed

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'exponential',
            'decay_rate': self.decay_rate,
            'peak_value': self.peak_value
        }


class NoSmoother(LabelSmoother):
    """
    No smoothing - keeps binary labels as is.

    Useful for comparison or disabling smoothing.
    """

    def smooth(self, labels: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
        """Return labels without smoothing."""
        return labels.astype(np.float32)

    def get_config(self) -> Dict[str, Any]:
        return {'type': 'none'}


def create_smoother(smoother_type: str, **kwargs) -> LabelSmoother:
    """
    Factory function to create label smoothers.

    Args:
        smoother_type: Type of smoother ('gaussian', 'triangular', 'uniform', 'exponential', 'none')
        **kwargs: Additional arguments for the smoother

    Returns:
        LabelSmoother instance

    Example:
        smoother = create_smoother('gaussian', sigma=2.0, peak_value=1.0)
        smoother = create_smoother('triangular', width=5)
        smoother = create_smoother('none')
    """
    smoothers = {
        'gaussian': GaussianSmoother,
        'triangular': TriangularSmoother,
        'uniform': UniformWindowSmoother,
        'exponential': ExponentialSmoother,
        'none': NoSmoother
    }

    if smoother_type not in smoothers:
        raise ValueError(
            f"Unknown smoother type: {smoother_type}. "
            f"Available types: {list(smoothers.keys())}"
        )

    return smoothers[smoother_type](**kwargs)


if __name__ == "__main__":
    # Test different smoothing strategies
    import matplotlib.pyplot as plt

    # Create synthetic labels
    num_frames = 100
    boundary_indices = np.array([20, 50, 80])

    # Test all smoothers
    smoothers = [
        ('None', create_smoother('none')),
        ('Gaussian (σ=2)', create_smoother('gaussian', sigma=2.0)),
        ('Gaussian (σ=5)', create_smoother('gaussian', sigma=5.0)),
        ('Triangular (w=5)', create_smoother('triangular', width=5)),
        ('Triangular (w=10)', create_smoother('triangular', width=10)),
        ('Uniform (w=3)', create_smoother('uniform', window_size=3)),
        ('Exponential (r=0.3)', create_smoother('exponential', decay_rate=0.3)),
    ]

    fig, axes = plt.subplots(len(smoothers), 1, figsize=(12, len(smoothers) * 2))

    for ax, (name, smoother) in zip(axes, smoothers):
        # Create binary labels
        binary_labels = np.zeros(num_frames)
        for b in boundary_indices:
            if b < num_frames:
                binary_labels[b] = 1

        # Apply smoothing
        smoothed = smoother.smooth(binary_labels, boundary_indices)

        # Plot
        ax.plot(smoothed, label='Smoothed', linewidth=2)
        ax.scatter(boundary_indices, [smoothed[b] for b in boundary_indices],
                  color='red', s=100, zorder=5, label='GT Boundaries')
        ax.set_title(name)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('label_smoothing_comparison.png', dpi=150)
    print("Saved comparison plot to label_smoothing_comparison.png")
