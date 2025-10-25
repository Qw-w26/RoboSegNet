"""Comprehensive evaluation metrics for boundary prediction.

This module provides a complete suite of metrics for evaluating task
segmentation models, including:
- Frame-level classification metrics
- F1@IoU with Hungarian matching
- Temporal boundary detection metrics
- Confidence-based metrics (AUC-ROC, AUC-PR)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .f1_iou import f1_at_iou, extract_intervals_from_binary


class BoundaryMetrics:
    """
    Comprehensive metrics calculator for boundary prediction evaluation.

    This class computes various metrics for evaluating segment boundary
    predictions, including frame-level classification metrics, F1@IoU
    scores at multiple thresholds, temporal accuracy, and confidence-based
    metrics.

    Attributes:
        iou_thresholds: List of IoU thresholds for F1@IoU calculation

    Example:
        >>> metrics = BoundaryMetrics(iou_thresholds=[0.3, 0.5, 0.7])
        >>> results = metrics.calculate_all_metrics(
        ...     predictions=pred_probs,
        ...     ground_truth=gt_labels,
        ...     confidence_scores=pred_probs,
        ...     timestamps=frame_times
        ... )
        >>> print(f"F1@IoU(0.5): {results['f1_iou_0.5']['f1']:.3f}")
    """

    def __init__(self, iou_thresholds: List[float] = [0.3, 0.5, 0.7]):
        """
        Initialize metrics calculator.

        Args:
            iou_thresholds: List of IoU thresholds for F1@IoU calculation
        """
        self.iou_thresholds = iou_thresholds

    def calculate_all_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.

        Args:
            predictions: Binary predictions (0/1) or probabilities ([0, 1]) for each timestep
            ground_truth: Ground truth binary labels (0/1) or soft labels ([0, 1])
            confidence_scores: Confidence scores for predictions (optional, uses predictions if None)
            timestamps: Timestamps for temporal analysis (optional)

        Returns:
            Dictionary containing all calculated metrics with keys:
                - precision, recall, f1_score, accuracy: Frame-level metrics
                - f1_iou_{threshold}: F1@IoU metrics for each threshold
                - temporal_{metric}: Temporal boundary detection metrics (if timestamps provided)
                - auc_roc, auc_pr: Confidence metrics (if confidence_scores provided)

        Example:
            >>> predictions = (model_output > 0.5).astype(int)
            >>> results = metrics.calculate_all_metrics(predictions, ground_truth)
        """
        metrics = {}

        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(predictions, ground_truth))

        # F1@IoU metrics at different thresholds
        for threshold in self.iou_thresholds:
            f1_metrics = self._calculate_f1_iou_metrics(predictions, ground_truth, threshold)
            metrics[f'f1_iou_{threshold}'] = f1_metrics

        # Temporal accuracy metrics
        if timestamps is not None:
            metrics.update(self._calculate_temporal_metrics(
                predictions, ground_truth, timestamps
            ))

        # Confidence-based metrics
        if confidence_scores is not None:
            metrics.update(self._calculate_confidence_metrics(
                confidence_scores, ground_truth
            ))

        return metrics

    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic frame-level classification metrics.

        Args:
            predictions: Binary predictions or probabilities
            ground_truth: Ground truth labels

        Returns:
            Dictionary with precision, recall, f1_score, accuracy, TP, FP, FN, TN
        """
        # Ensure arrays have the same shape
        if predictions.shape != ground_truth.shape:
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]

        # Convert to binary
        binary_predictions = (predictions > 0.5).astype(int)
        binary_ground_truth = (ground_truth > 0.5).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((binary_predictions == 1) & (binary_ground_truth == 1))
        fp = np.sum((binary_predictions == 1) & (binary_ground_truth == 0))
        fn = np.sum((binary_predictions == 0) & (binary_ground_truth == 1))
        tn = np.sum((binary_predictions == 0) & (binary_ground_truth == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

    def _calculate_f1_iou_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        iou_threshold: float
    ) -> Dict[str, float]:
        """
        Calculate F1@IoU metrics using Hungarian matching.

        Args:
            predictions: Binary predictions or probabilities
            ground_truth: Ground truth labels
            iou_threshold: IoU threshold for matching

        Returns:
            Dictionary with F1@IoU metrics
        """
        # Ensure arrays have the same shape
        if predictions.shape != ground_truth.shape:
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]

        # Convert to binary
        binary_predictions = (predictions > 0.5).astype(int)
        binary_ground_truth = (ground_truth > 0.5).astype(int)

        # Extract intervals
        pred_intervals = extract_intervals_from_binary(binary_predictions)
        gt_intervals = extract_intervals_from_binary(binary_ground_truth)

        # Calculate F1@IoU
        return f1_at_iou(pred_intervals, gt_intervals, iou_threshold)

    def _calculate_temporal_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate temporal boundary detection accuracy metrics.

        Measures how accurately predicted boundaries align with ground truth
        boundaries in time, using a tolerance window (default: 0.5 seconds).

        Args:
            predictions: Binary predictions or probabilities
            ground_truth: Ground truth labels
            timestamps: Timestamp for each frame

        Returns:
            Dictionary with temporal metrics
        """
        # Ensure arrays have the same shape
        if predictions.shape != ground_truth.shape:
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]
            timestamps = timestamps[:min_len]

        # Convert to binary
        binary_predictions = (predictions > 0.5).astype(int)
        binary_ground_truth = (ground_truth > 0.5).astype(int)

        # Find boundary timestamps
        pred_boundaries = self._find_boundaries(binary_predictions, timestamps)
        gt_boundaries = self._find_boundaries(binary_ground_truth, timestamps)

        # Handle edge cases
        if len(pred_boundaries) == 0 and len(gt_boundaries) == 0:
            return {
                'temporal_precision': 1.0,
                'temporal_recall': 1.0,
                'temporal_f1': 1.0,
                'mean_temporal_error': 0.0,
                'std_temporal_error': 0.0
            }

        if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
            return {
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'mean_temporal_error': float('inf'),
                'std_temporal_error': 0.0
            }

        # Calculate temporal distances
        temporal_distances = self._calculate_temporal_distances(
            pred_boundaries, gt_boundaries
        )

        # Temporal precision: fraction of predicted boundaries close to ground truth
        temporal_tolerance = 0.5  # 0.5 second tolerance
        temporal_precision = np.mean(
            temporal_distances['pred_to_gt'] < temporal_tolerance
        )

        # Temporal recall: fraction of ground truth boundaries close to predictions
        temporal_recall = np.mean(
            temporal_distances['gt_to_pred'] < temporal_tolerance
        )

        # Temporal F1
        temporal_f1 = (
            2 * temporal_precision * temporal_recall /
            (temporal_precision + temporal_recall)
            if (temporal_precision + temporal_recall) > 0 else 0.0
        )

        return {
            'temporal_precision': float(temporal_precision),
            'temporal_recall': float(temporal_recall),
            'temporal_f1': float(temporal_f1),
            'mean_temporal_error': float(np.mean(temporal_distances['pred_to_gt'])),
            'std_temporal_error': float(np.std(temporal_distances['pred_to_gt']))
        }

    def _calculate_confidence_metrics(
        self,
        confidence_scores: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate confidence-based metrics (AUC-ROC, AUC-PR, optimal threshold).

        Args:
            confidence_scores: Confidence/probability scores
            ground_truth: Ground truth labels

        Returns:
            Dictionary with AUC metrics and optimal threshold
        """
        # Convert ground truth to binary
        binary_ground_truth = (ground_truth > 0.5).astype(int)

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(binary_ground_truth, confidence_scores)
        except ValueError:
            # Happens when all labels are same class
            auc_roc = 0.0

        # Precision-Recall curve
        try:
            precision, recall, thresholds = precision_recall_curve(
                binary_ground_truth, confidence_scores
            )
        except ValueError:
            return {
                'auc_roc': auc_roc,
                'auc_pr': 0.0,
                'optimal_threshold': 0.5,
                'max_f1_at_optimal': 0.0
            }

        # AUC-PR (area under precision-recall curve)
        auc_pr = np.trapz(precision, recall)

        # Find optimal threshold (maximizing F1)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        return {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'optimal_threshold': float(optimal_threshold),
            'max_f1_at_optimal': float(f1_scores[optimal_idx]) if len(f1_scores) > 0 else 0.0
        }

    @staticmethod
    def _find_boundaries(binary_sequence: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Find boundary timestamps from binary sequence (rising edges).

        Args:
            binary_sequence: Binary sequence
            timestamps: Corresponding timestamps

        Returns:
            Array of boundary timestamps
        """
        boundaries = []
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] == 1 and binary_sequence[i-1] == 0:
                boundaries.append(timestamps[i])
        return np.array(boundaries)

    @staticmethod
    def _calculate_temporal_distances(
        pred_boundaries: np.ndarray,
        gt_boundaries: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate temporal distances between predicted and ground truth boundaries.

        Args:
            pred_boundaries: Predicted boundary timestamps
            gt_boundaries: Ground truth boundary timestamps

        Returns:
            Dictionary with 'pred_to_gt' and 'gt_to_pred' distance arrays
        """
        if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
            return {'pred_to_gt': np.array([]), 'gt_to_pred': np.array([])}

        # Distance from each prediction to nearest ground truth
        pred_to_gt = []
        for pred_boundary in pred_boundaries:
            distances = np.abs(gt_boundaries - pred_boundary)
            pred_to_gt.append(np.min(distances))

        # Distance from each ground truth to nearest prediction
        gt_to_pred = []
        for gt_boundary in gt_boundaries:
            distances = np.abs(pred_boundaries - gt_boundary)
            gt_to_pred.append(np.min(distances))

        return {
            'pred_to_gt': np.array(pred_to_gt),
            'gt_to_pred': np.array(gt_to_pred)
        }


def calculate_segment_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate segment-level metrics using Hungarian matching.

    This function extracts segments from binary sequences and matches them
    using the Hungarian algorithm based on IoU, then computes precision,
    recall, and F1 for the matched segments.

    Args:
        predictions: Binary predictions (0/1) for each timestep
        ground_truth: Ground truth binary labels (0/1) for each timestep
        iou_threshold: IoU threshold for considering a match (default: 0.5)

    Returns:
        Dictionary with segment-level metrics

    Example:
        >>> pred = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        >>> gt = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> metrics = calculate_segment_metrics(pred, gt)
        >>> print(f"Segment F1: {metrics['segment_f1']:.3f}")
    """
    # Ensure arrays have the same shape
    if predictions.shape != ground_truth.shape:
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

    # Convert to binary
    binary_predictions = (predictions > 0.5).astype(int)
    binary_ground_truth = (ground_truth > 0.5).astype(int)

    # Extract segments
    pred_segments = _extract_segments(binary_predictions)
    gt_segments = _extract_segments(binary_ground_truth)

    # Handle edge cases
    if len(pred_segments) == 0 and len(gt_segments) == 0:
        return {
            'segment_precision': 1.0,
            'segment_recall': 1.0,
            'segment_f1': 1.0,
            'num_predicted_segments': 0,
            'num_ground_truth_segments': 0,
            'num_matched_segments': 0
        }

    if len(pred_segments) == 0:
        return {
            'segment_precision': 0.0,
            'segment_recall': 0.0,
            'segment_f1': 0.0,
            'num_predicted_segments': 0,
            'num_ground_truth_segments': len(gt_segments),
            'num_matched_segments': 0
        }

    if len(gt_segments) == 0:
        return {
            'segment_precision': 0.0,
            'segment_recall': 0.0,
            'segment_f1': 0.0,
            'num_predicted_segments': len(pred_segments),
            'num_ground_truth_segments': 0,
            'num_matched_segments': 0
        }

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_segments), len(gt_segments)))
    for i, pred_seg in enumerate(pred_segments):
        for j, gt_seg in enumerate(gt_segments):
            iou_matrix[i, j] = _calculate_iou(pred_seg, gt_seg)

    # Use Hungarian algorithm for optimal matching
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Count matches above threshold
    matches = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches += 1

    # Calculate metrics
    precision = matches / len(pred_segments)
    recall = matches / len(gt_segments)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'segment_precision': float(precision),
        'segment_recall': float(recall),
        'segment_f1': float(f1),
        'num_predicted_segments': len(pred_segments),
        'num_ground_truth_segments': len(gt_segments),
        'num_matched_segments': int(matches)
    }


def _extract_segments(binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
    """Extract segments (continuous 1s) from binary sequence."""
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_segment:
            start = i
            in_segment = True
        elif val == 0 and in_segment:
            segments.append((start, i))
            in_segment = False

    # Handle case where sequence ends with 1
    if in_segment:
        segments.append((start, len(binary_sequence)))

    return segments


def _calculate_iou(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> float:
    """Calculate IoU between two segments."""
    start1, end1 = segment1
    start2, end2 = segment2

    # Intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start

    return intersection / union if union > 0 else 0.0
