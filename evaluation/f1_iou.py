"""F1@IoU metric with Hungarian algorithm matching.

This module implements the F1@IoU metric for evaluating segment boundary
predictions. It uses the Hungarian algorithm for optimal interval matching
and supports multiple IoU thresholds.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def f1_at_iou(
    pred_intervals: np.ndarray,
    gt_intervals: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute F1 score at a given IoU threshold using Hungarian matching.

    This function finds the optimal matching between predicted and ground truth
    intervals using the Hungarian algorithm, then counts matches above the
    IoU threshold to compute precision, recall, and F1.

    Args:
        pred_intervals: Predicted intervals of shape (N1, 2) with [start, end]
        gt_intervals: Ground truth intervals of shape (N2, 2) with [start, end]
        iou_threshold: IoU threshold to consider a match (default: 0.5)

    Returns:
        Dictionary with keys:
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'tp': Number of true positives
            - 'fp': Number of false positives
            - 'fn': Number of false negatives

    Example:
        >>> pred = np.array([[0, 10], [20, 30]])
        >>> gt = np.array([[0, 12], [18, 28]])
        >>> metrics = f1_at_iou(pred, gt, iou_threshold=0.5)
        >>> print(f"F1@0.5: {metrics['f1']:.3f}")
    """
    N_pred = len(pred_intervals)
    N_gt = len(gt_intervals)

    # Handle edge cases
    if N_pred == 0 and N_gt == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    elif N_pred == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': N_gt
        }
    elif N_gt == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': N_pred,
            'fn': 0
        }

    # Compute IoU matrix
    iou_matrix = calculate_iou_matrix(pred_intervals, gt_intervals)

    # Convert IoU to cost for Hungarian algorithm (maximize IoU → minimize cost)
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Count matches above threshold
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1

    fp = N_pred - tp
    fn = N_gt - tp

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def calculate_iou_matrix(
    intervals1: np.ndarray,
    intervals2: np.ndarray
) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of intervals.

    Args:
        intervals1: First set of intervals, shape (N1, 2)
        intervals2: Second set of intervals, shape (N2, 2)

    Returns:
        IoU matrix of shape (N1, N2)

    Example:
        >>> intervals1 = np.array([[0, 10], [20, 30]])
        >>> intervals2 = np.array([[0, 12], [18, 28]])
        >>> iou_mat = calculate_iou_matrix(intervals1, intervals2)
        >>> print(iou_mat.shape)  # (2, 2)
    """
    N1 = len(intervals1)
    N2 = len(intervals2)

    iou_matrix = np.zeros((N1, N2))

    for i, (start1, end1) in enumerate(intervals1):
        for j, (start2, end2) in enumerate(intervals2):
            # Calculate intersection
            inter_start = max(start1, start2)
            inter_end = min(end1, end2)
            intersection = max(0, inter_end - inter_start)

            # Calculate union
            union_start = min(start1, start2)
            union_end = max(end1, end2)
            union = union_end - union_start

            # IoU
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    return iou_matrix


def calculate_iou(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float]
) -> float:
    """
    Calculate IoU between two intervals.

    Args:
        interval1: First interval as (start, end)
        interval2: Second interval as (start, end)

    Returns:
        IoU score between 0 and 1

    Example:
        >>> iou = calculate_iou((0, 10), (5, 15))
        >>> print(f"IoU: {iou:.3f}")  # 0.333
    """
    start1, end1 = interval1
    start2, end2 = interval2

    # Intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)

    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start

    return intersection / union if union > 0 else 0.0


def extract_intervals_from_binary(binary_sequence: np.ndarray) -> np.ndarray:
    """
    Extract continuous intervals where binary_sequence is 1.

    Args:
        binary_sequence: Binary sequence of 0s and 1s

    Returns:
        Array of intervals with shape (N, 2) where each row is [start, end]

    Example:
        >>> seq = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])
        >>> intervals = extract_intervals_from_binary(seq)
        >>> print(intervals)  # [[2, 5], [6, 8]]
    """
    intervals = []
    in_interval = False
    start = 0

    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_interval:
            start = i
            in_interval = True
        elif val == 0 and in_interval:
            intervals.append([start, i])
            in_interval = False

    # Handle case where sequence ends with 1
    if in_interval:
        intervals.append([start, len(binary_sequence)])

    return np.array(intervals) if intervals else np.array([]).reshape(0, 2)


def f1_at_multiple_ious(
    pred_intervals: np.ndarray,
    gt_intervals: np.ndarray,
    iou_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[float, Dict[str, float]]:
    """
    Compute F1@IoU for multiple IoU thresholds.

    Args:
        pred_intervals: Predicted intervals of shape (N1, 2)
        gt_intervals: Ground truth intervals of shape (N2, 2)
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary mapping threshold to metrics dict

    Example:
        >>> pred = np.array([[0, 10], [20, 30]])
        >>> gt = np.array([[0, 12], [18, 28]])
        >>> results = f1_at_multiple_ious(pred, gt)
        >>> for thresh, metrics in results.items():
        ...     print(f"F1@{thresh}: {metrics['f1']:.3f}")
    """
    results = {}
    for threshold in iou_thresholds:
        results[threshold] = f1_at_iou(pred_intervals, gt_intervals, threshold)
    return results
