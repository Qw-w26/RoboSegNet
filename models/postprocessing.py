"""
Post-processing utilities for boundary predictions.

This module provides methods to post-process raw model predictions
to ensure non-overlapping boundaries and improve detection quality.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import find_peaks


def non_maximum_suppression(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply non-maximum suppression to boundary predictions.

    Finds local maxima in the probability curve and suppresses nearby
    predictions to ensure non-overlapping boundaries.

    Args:
        probabilities: Array of boundary probabilities (num_frames,)
        threshold: Minimum probability threshold for a boundary
        min_distance: Minimum distance between adjacent boundaries

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)
    """
    if len(probabilities) == 0:
        return np.array([]), np.array([])

    # Find peaks in the probability curve
    peaks, properties = find_peaks(
        probabilities,
        height=threshold,
        distance=min_distance
    )

    # Get the probabilities at peak locations
    peak_probs = probabilities[peaks]

    return peaks, peak_probs


def threshold_detection(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_distance: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple threshold-based boundary detection.

    Args:
        probabilities: Array of boundary probabilities (num_frames,)
        threshold: Minimum probability threshold
        min_distance: If specified, enforce minimum distance between boundaries

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)
    """
    # Find all frames above threshold
    candidates = np.where(probabilities >= threshold)[0]

    if len(candidates) == 0:
        return np.array([]), np.array([])

    if min_distance is None:
        return candidates, probabilities[candidates]

    # Enforce minimum distance by greedy selection
    selected = [candidates[0]]
    selected_probs = [probabilities[candidates[0]]]

    for idx in candidates[1:]:
        if idx - selected[-1] >= min_distance:
            selected.append(idx)
            selected_probs.append(probabilities[idx])

    return np.array(selected), np.array(selected_probs)


def adaptive_threshold_detection(
    probabilities: np.ndarray,
    percentile: float = 90,
    min_distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive threshold based on probability distribution.

    Uses a percentile of the probability distribution as threshold,
    then applies non-maximum suppression.

    Args:
        probabilities: Array of boundary probabilities (num_frames,)
        percentile: Percentile for adaptive threshold (0-100)
        min_distance: Minimum distance between boundaries

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)
    """
    if len(probabilities) == 0:
        return np.array([]), np.array([])

    # Compute adaptive threshold
    threshold = np.percentile(probabilities, percentile)

    # Apply NMS with adaptive threshold
    return non_maximum_suppression(probabilities, threshold, min_distance)


def top_k_boundaries(
    probabilities: np.ndarray,
    k: int,
    min_distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-k boundaries by probability, enforcing minimum distance.

    Args:
        probabilities: Array of boundary probabilities (num_frames,)
        k: Number of boundaries to select
        min_distance: Minimum distance between boundaries

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)
    """
    if len(probabilities) == 0 or k == 0:
        return np.array([]), np.array([])

    # Sort all indices by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]

    # Greedily select top-k with distance constraint
    selected = []
    selected_probs = []

    for idx in sorted_indices:
        # Check if this index is far enough from all selected
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)
            selected_probs.append(probabilities[idx])

            if len(selected) >= k:
                break

    # Sort by frame index for output
    if len(selected) > 0:
        order = np.argsort(selected)
        selected = np.array(selected)[order]
        selected_probs = np.array(selected_probs)[order]
    else:
        selected = np.array([])
        selected_probs = np.array([])

    return selected, selected_probs


def expected_num_boundaries(
    probabilities: np.ndarray,
    expected_segments: int,
    min_distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select boundaries to match expected number of segments.

    If expected_segments is N, selects N-1 boundaries.

    Args:
        probabilities: Array of boundary probabilities (num_frames,)
        expected_segments: Expected number of segments
        min_distance: Minimum distance between boundaries

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)
    """
    num_boundaries = max(0, expected_segments - 1)
    return top_k_boundaries(probabilities, num_boundaries, min_distance)


def merge_nearby_boundaries(
    boundary_indices: np.ndarray,
    boundary_probs: np.ndarray,
    merge_distance: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge boundaries that are too close together.

    When boundaries are within merge_distance, keeps the one with
    higher probability and discards the others.

    Args:
        boundary_indices: Indices of detected boundaries
        boundary_probs: Probabilities of detected boundaries
        merge_distance: Maximum distance for merging

    Returns:
        Tuple of (merged_indices, merged_probabilities)
    """
    if len(boundary_indices) <= 1:
        return boundary_indices, boundary_probs

    # Sort by index
    order = np.argsort(boundary_indices)
    sorted_indices = boundary_indices[order]
    sorted_probs = boundary_probs[order]

    merged_indices = [sorted_indices[0]]
    merged_probs = [sorted_probs[0]]

    for i in range(1, len(sorted_indices)):
        # Check distance to last kept boundary
        if sorted_indices[i] - merged_indices[-1] < merge_distance:
            # Replace with higher probability boundary
            if sorted_probs[i] > merged_probs[-1]:
                merged_indices[-1] = sorted_indices[i]
                merged_probs[-1] = sorted_probs[i]
        else:
            merged_indices.append(sorted_indices[i])
            merged_probs.append(sorted_probs[i])

    return np.array(merged_indices), np.array(merged_probs)


def smooth_probabilities(
    probabilities: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Smooth probability curve with a moving average.

    Args:
        probabilities: Array of boundary probabilities
        window_size: Size of smoothing window

    Returns:
        Smoothed probabilities
    """
    if window_size <= 1:
        return probabilities

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(probabilities, kernel, mode='same')

    return smoothed


def detect_boundaries(
    probabilities: np.ndarray,
    method: str = 'nms',
    threshold: float = 0.5,
    min_distance: int = 5,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified interface for boundary detection with various methods.

    Args:
        probabilities: Array of boundary probabilities
        method: Detection method:
            - 'nms': Non-maximum suppression (default)
            - 'threshold': Simple thresholding
            - 'adaptive': Adaptive threshold based on percentile
            - 'topk': Select top-k boundaries
            - 'expected': Match expected number of segments
        threshold: Probability threshold (for 'nms' and 'threshold')
        min_distance: Minimum distance between boundaries
        **kwargs: Additional method-specific arguments:
            - percentile: For 'adaptive' (default: 90)
            - k: For 'topk' (required)
            - expected_segments: For 'expected' (required)
            - merge_distance: For post-merging (optional)
            - smooth_window: For pre-smoothing (optional)

    Returns:
        Tuple of (boundary_indices, boundary_probabilities)

    Example:
        # Non-maximum suppression
        boundaries, probs = detect_boundaries(probs, method='nms', threshold=0.5)

        # Top-k detection
        boundaries, probs = detect_boundaries(probs, method='topk', k=5)

        # With smoothing and merging
        boundaries, probs = detect_boundaries(
            probs, method='nms',
            smooth_window=5, merge_distance=3
        )
    """
    # Optional pre-smoothing
    smooth_window = kwargs.get('smooth_window', None)
    if smooth_window:
        probabilities = smooth_probabilities(probabilities, smooth_window)

    # Detect boundaries based on method
    if method == 'nms':
        boundaries, probs = non_maximum_suppression(
            probabilities, threshold, min_distance
        )
    elif method == 'threshold':
        boundaries, probs = threshold_detection(
            probabilities, threshold, min_distance
        )
    elif method == 'adaptive':
        percentile = kwargs.get('percentile', 90)
        boundaries, probs = adaptive_threshold_detection(
            probabilities, percentile, min_distance
        )
    elif method == 'topk':
        k = kwargs.get('k')
        if k is None:
            raise ValueError("'k' must be specified for 'topk' method")
        boundaries, probs = top_k_boundaries(probabilities, k, min_distance)
    elif method == 'expected':
        expected_segments = kwargs.get('expected_segments')
        if expected_segments is None:
            raise ValueError("'expected_segments' must be specified for 'expected' method")
        boundaries, probs = expected_num_boundaries(
            probabilities, expected_segments, min_distance
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: 'nms', 'threshold', 'adaptive', 'topk', 'expected'"
        )

    # Optional post-merging
    merge_distance = kwargs.get('merge_distance', None)
    if merge_distance and len(boundaries) > 0:
        boundaries, probs = merge_nearby_boundaries(
            boundaries, probs, merge_distance
        )

    return boundaries, probs


if __name__ == "__main__":
    # Test boundary detection methods
    import matplotlib.pyplot as plt

    # Create synthetic probability curve with multiple peaks
    np.random.seed(42)
    num_frames = 200

    # Create smooth peaks at known boundaries
    gt_boundaries = [40, 80, 120, 160]
    probs = np.zeros(num_frames)

    for b in gt_boundaries:
        # Add Gaussian peak
        distances = np.abs(np.arange(num_frames) - b)
        probs += 0.8 * np.exp(-(distances ** 2) / (2 * 5 ** 2))

    # Add noise
    probs += np.random.normal(0, 0.1, num_frames)
    probs = np.clip(probs, 0, 1)

    # Test different methods
    methods = [
        ('NMS', 'nms', {'threshold': 0.3, 'min_distance': 10}),
        ('Threshold', 'threshold', {'threshold': 0.5, 'min_distance': 10}),
        ('Adaptive', 'adaptive', {'percentile': 85, 'min_distance': 10}),
        ('Top-K', 'topk', {'k': 4, 'min_distance': 10}),
    ]

    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(12, len(methods) * 2 + 2))

    # Plot original probabilities
    axes[0].plot(probs, label='Probabilities', linewidth=2)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold=0.5')
    for b in gt_boundaries:
        axes[0].axvline(b, color='green', linestyle='--', alpha=0.5)
    axes[0].set_title('Original Probabilities (Green = Ground Truth)')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test each method
    for i, (name, method, params) in enumerate(methods, 1):
        boundaries, boundary_probs = detect_boundaries(probs, method=method, **params)

        axes[i].plot(probs, label='Probabilities', linewidth=2, alpha=0.5)
        axes[i].scatter(boundaries, boundary_probs, color='red', s=100, zorder=5,
                       label=f'Detected ({len(boundaries)} boundaries)')
        for b in gt_boundaries:
            axes[i].axvline(b, color='green', linestyle='--', alpha=0.5)

        axes[i].set_title(f'{name}: {params}')
        axes[i].set_ylabel('Probability')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig('boundary_detection_methods.png', dpi=150)
    print("Saved comparison plot to boundary_detection_methods.png")
