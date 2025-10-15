"""
Evaluation metrics for task segmentation.

This module provides standard metrics for evaluating segmentation quality
including frame-level accuracy, boundary detection, and temporal metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)


@dataclass
class SegmentationMetrics:
    """Container for segmentation evaluation metrics."""

    # Frame-level metrics
    frame_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    # Clustering metrics
    adjusted_rand_index: float
    normalized_mutual_info: float

    # Boundary detection metrics
    boundary_precision: Optional[float] = None
    boundary_recall: Optional[float] = None
    boundary_f1: Optional[float] = None
    mean_boundary_error: Optional[float] = None  # frames

    # Temporal metrics
    over_segmentation_score: Optional[float] = None
    under_segmentation_score: Optional[float] = None

    # Additional info
    num_predicted_segments: Optional[int] = None
    num_ground_truth_segments: Optional[int] = None
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame_accuracy': self.frame_accuracy,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'weighted_precision': self.weighted_precision,
            'weighted_recall': self.weighted_recall,
            'weighted_f1': self.weighted_f1,
            'adjusted_rand_index': self.adjusted_rand_index,
            'normalized_mutual_info': self.normalized_mutual_info,
            'boundary_precision': self.boundary_precision,
            'boundary_recall': self.boundary_recall,
            'boundary_f1': self.boundary_f1,
            'mean_boundary_error': self.mean_boundary_error,
            'over_segmentation_score': self.over_segmentation_score,
            'under_segmentation_score': self.under_segmentation_score,
            'num_predicted_segments': self.num_predicted_segments,
            'num_ground_truth_segments': self.num_ground_truth_segments,
        }

    def print_summary(self, method_name: str = ""):
        """Print a formatted summary of metrics."""
        print(f"\n{'='*70}")
        if method_name:
            print(f"Segmentation Metrics - {method_name}")
        else:
            print("Segmentation Metrics")
        print(f"{'='*70}")

        print(f"\nFrame-Level Metrics:")
        print(f"  Accuracy:           {self.frame_accuracy:.4f}")
        print(f"  Macro Precision:    {self.macro_precision:.4f}")
        print(f"  Macro Recall:       {self.macro_recall:.4f}")
        print(f"  Macro F1:           {self.macro_f1:.4f}")
        print(f"  Weighted Precision: {self.weighted_precision:.4f}")
        print(f"  Weighted Recall:    {self.weighted_recall:.4f}")
        print(f"  Weighted F1:        {self.weighted_f1:.4f}")

        print(f"\nClustering Metrics:")
        print(f"  Adjusted Rand Index: {self.adjusted_rand_index:.4f}")
        print(f"  Normalized Mutual Info: {self.normalized_mutual_info:.4f}")

        if self.boundary_precision is not None:
            print(f"\nBoundary Detection Metrics:")
            print(f"  Precision:          {self.boundary_precision:.4f}")
            print(f"  Recall:             {self.boundary_recall:.4f}")
            print(f"  F1:                 {self.boundary_f1:.4f}")
            print(f"  Mean Error (frames): {self.mean_boundary_error:.2f}")

        if self.over_segmentation_score is not None:
            print(f"\nTemporal Metrics:")
            print(f"  Over-segmentation:  {self.over_segmentation_score:.4f}")
            print(f"  Under-segmentation: {self.under_segmentation_score:.4f}")

        print(f"\nSegment Counts:")
        print(f"  Predicted:          {self.num_predicted_segments}")
        print(f"  Ground Truth:       {self.num_ground_truth_segments}")

        print(f"{'='*70}\n")


def compute_frame_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute frame-level classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        Dictionary of frame-level metrics
    """
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Clustering metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'frame_accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'confusion_matrix': cm
    }


def detect_boundaries(labels: np.ndarray) -> np.ndarray:
    """
    Detect segment boundaries from frame labels.

    Args:
        labels: Frame-level segment labels (N,)

    Returns:
        Array of boundary frame indices
    """
    if len(labels) < 2:
        return np.array([])

    # Find where labels change
    transitions = np.where(np.diff(labels) != 0)[0]
    # Boundary is at the transition point
    boundaries = transitions + 1

    return boundaries


def compute_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 5
) -> Dict[str, float]:
    """
    Compute boundary detection metrics.

    A predicted boundary is considered correct if it's within `tolerance` frames
    of a ground truth boundary.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        tolerance: Tolerance window for boundary matching (frames)

    Returns:
        Dictionary of boundary metrics
    """
    # Detect boundaries
    true_boundaries = detect_boundaries(y_true)
    pred_boundaries = detect_boundaries(y_pred)

    if len(true_boundaries) == 0 or len(pred_boundaries) == 0:
        return {
            'boundary_precision': 0.0,
            'boundary_recall': 0.0,
            'boundary_f1': 0.0,
            'mean_boundary_error': float('inf')
        }

    # Match predicted boundaries to ground truth
    matched_pred = []
    boundary_errors = []

    for pred_b in pred_boundaries:
        # Find closest ground truth boundary
        distances = np.abs(true_boundaries - pred_b)
        min_dist = np.min(distances)

        if min_dist <= tolerance:
            matched_pred.append(pred_b)
            boundary_errors.append(min_dist)

    # Match ground truth boundaries to predictions
    matched_true = []
    for true_b in true_boundaries:
        distances = np.abs(pred_boundaries - true_b)
        min_dist = np.min(distances)

        if min_dist <= tolerance:
            matched_true.append(true_b)

    # Compute metrics
    precision = len(matched_pred) / len(pred_boundaries) if len(pred_boundaries) > 0 else 0
    recall = len(matched_true) / len(true_boundaries) if len(true_boundaries) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_error = np.mean(boundary_errors) if boundary_errors else float('inf')

    return {
        'boundary_precision': precision,
        'boundary_recall': recall,
        'boundary_f1': f1,
        'mean_boundary_error': mean_error
    }


def compute_temporal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute temporal segmentation quality metrics.

    Over-segmentation: How much the prediction splits ground truth segments
    Under-segmentation: How much the prediction merges ground truth segments

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        Dictionary of temporal metrics
    """
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    # Over-segmentation: for each true segment, how many pred segments does it map to?
    over_seg_scores = []
    for true_label in unique_true:
        true_mask = y_true == true_label
        pred_labels_in_true = y_pred[true_mask]
        n_pred_segments = len(np.unique(pred_labels_in_true))
        over_seg_scores.append(n_pred_segments)

    over_segmentation = np.mean(over_seg_scores)

    # Under-segmentation: for each pred segment, how many true segments does it map to?
    under_seg_scores = []
    for pred_label in unique_pred:
        pred_mask = y_pred == pred_label
        true_labels_in_pred = y_true[pred_mask]
        n_true_segments = len(np.unique(true_labels_in_pred))
        under_seg_scores.append(n_true_segments)

    under_segmentation = np.mean(under_seg_scores)

    return {
        'over_segmentation_score': over_segmentation,
        'under_segmentation_score': under_segmentation
    }


def evaluate_segmentation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    boundary_tolerance: int = 5,
    compute_boundary: bool = True,
    compute_temporal: bool = True
) -> SegmentationMetrics:
    """
    Comprehensive evaluation of segmentation quality.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        boundary_tolerance: Tolerance for boundary detection (frames)
        compute_boundary: Whether to compute boundary metrics
        compute_temporal: Whether to compute temporal metrics

    Returns:
        SegmentationMetrics object with all computed metrics
    """
    # Frame-level metrics
    frame_metrics = compute_frame_metrics(y_true, y_pred)

    # Boundary metrics
    boundary_metrics = {}
    if compute_boundary:
        boundary_metrics = compute_boundary_metrics(y_true, y_pred, boundary_tolerance)

    # Temporal metrics
    temporal_metrics = {}
    if compute_temporal:
        temporal_metrics = compute_temporal_metrics(y_true, y_pred)

    # Combine all metrics
    metrics = SegmentationMetrics(
        frame_accuracy=frame_metrics['frame_accuracy'],
        macro_precision=frame_metrics['macro_precision'],
        macro_recall=frame_metrics['macro_recall'],
        macro_f1=frame_metrics['macro_f1'],
        weighted_precision=frame_metrics['weighted_precision'],
        weighted_recall=frame_metrics['weighted_recall'],
        weighted_f1=frame_metrics['weighted_f1'],
        adjusted_rand_index=frame_metrics['adjusted_rand_index'],
        normalized_mutual_info=frame_metrics['normalized_mutual_info'],
        boundary_precision=boundary_metrics.get('boundary_precision'),
        boundary_recall=boundary_metrics.get('boundary_recall'),
        boundary_f1=boundary_metrics.get('boundary_f1'),
        mean_boundary_error=boundary_metrics.get('mean_boundary_error'),
        over_segmentation_score=temporal_metrics.get('over_segmentation_score'),
        under_segmentation_score=temporal_metrics.get('under_segmentation_score'),
        num_predicted_segments=len(np.unique(y_pred)),
        num_ground_truth_segments=len(np.unique(y_true)),
        confusion_matrix=frame_metrics['confusion_matrix']
    )

    return metrics


def aggregate_metrics(metrics_list: List[SegmentationMetrics]) -> SegmentationMetrics:
    """
    Aggregate metrics across multiple trajectories.

    Args:
        metrics_list: List of SegmentationMetrics objects

    Returns:
        Aggregated SegmentationMetrics
    """
    if not metrics_list:
        raise ValueError("Empty metrics list")

    # Compute means for all numeric fields
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return np.mean(values) if values else None

    aggregated = SegmentationMetrics(
        frame_accuracy=np.mean([m.frame_accuracy for m in metrics_list]),
        macro_precision=np.mean([m.macro_precision for m in metrics_list]),
        macro_recall=np.mean([m.macro_recall for m in metrics_list]),
        macro_f1=np.mean([m.macro_f1 for m in metrics_list]),
        weighted_precision=np.mean([m.weighted_precision for m in metrics_list]),
        weighted_recall=np.mean([m.weighted_recall for m in metrics_list]),
        weighted_f1=np.mean([m.weighted_f1 for m in metrics_list]),
        adjusted_rand_index=np.mean([m.adjusted_rand_index for m in metrics_list]),
        normalized_mutual_info=np.mean([m.normalized_mutual_info for m in metrics_list]),
        boundary_precision=safe_mean([m.boundary_precision for m in metrics_list]),
        boundary_recall=safe_mean([m.boundary_recall for m in metrics_list]),
        boundary_f1=safe_mean([m.boundary_f1 for m in metrics_list]),
        mean_boundary_error=safe_mean([m.mean_boundary_error for m in metrics_list]),
        over_segmentation_score=safe_mean([m.over_segmentation_score for m in metrics_list]),
        under_segmentation_score=safe_mean([m.under_segmentation_score for m in metrics_list]),
        num_predicted_segments=int(np.sum([m.num_predicted_segments or 0 for m in metrics_list])),
        num_ground_truth_segments=int(np.sum([m.num_ground_truth_segments or 0 for m in metrics_list]))
    )

    return aggregated


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic ground truth (5 segments)
    y_true = np.concatenate([
        np.full(100, 0),
        np.full(150, 1),
        np.full(80, 2),
        np.full(120, 3),
        np.full(90, 4)
    ])

    # Generate synthetic predictions (with some errors)
    y_pred = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(len(y_pred), size=50, replace=False)
    y_pred[noise_indices] = (y_pred[noise_indices] + 1) % 5
    # Shift some boundaries
    y_pred[95:105] = 0  # Shift first boundary
    y_pred[245:255] = 1  # Shift second boundary

    # Evaluate
    metrics = evaluate_segmentation(y_true, y_pred)
    metrics.print_summary("Example Method")
