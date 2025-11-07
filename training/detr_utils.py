"""Utilities for DETR-style boundary detection training.

This module provides helper functions for converting between dense labels
and set-based targets used by DETR models.
"""

import torch
import numpy as np
from typing import List, Dict, Any
from scipy.signal import find_peaks
from evaluation.f1_iou import f1_at_iou


def boundaries_to_segments(boundary_frames: np.ndarray, window_size: int) -> np.ndarray:
    """Convert boundary positions to segment intervals.

    Args:
        boundary_frames: Array of boundary frame positions
        window_size: Total window size in frames

    Returns:
        Array of segment intervals with shape (N, 2) where each row is [start, end]

    Example:
        >>> boundaries = np.array([5, 10, 15])
        >>> segments = boundaries_to_segments(boundaries, window_size=20)
        >>> print(segments)  # [[0, 5], [5, 10], [10, 15], [15, 20]]
    """
    if len(boundary_frames) == 0:
        # No boundaries means one segment covering entire window
        return np.array([[0, window_size]])

    # Sort boundaries
    boundaries = np.sort(boundary_frames)

    # Create segments between consecutive boundaries
    segments = []

    # First segment: from start to first boundary
    segments.append([0, boundaries[0]])

    # Middle segments: between consecutive boundaries
    for i in range(len(boundaries) - 1):
        segments.append([boundaries[i], boundaries[i + 1]])

    # Last segment: from last boundary to end
    segments.append([boundaries[-1], window_size])

    return np.array(segments)


def dense_labels_to_set_targets(
    labels: np.ndarray,
    window_size: int,
    threshold: float = 0.5,
    min_distance: int = 1
) -> Dict[str, torch.Tensor]:
    """Convert dense smoothed labels to set-based targets for DETR.

    Args:
        labels: (window_size,) array of smoothed boundary labels [0, 1]
        window_size: Size of the window (for normalization)
        threshold: Minimum peak height to consider as boundary
        min_distance: Minimum distance between boundaries (frames)

    Returns:
        Dictionary with:
            - 'positions': (num_boundaries,) normalized positions [0, 1]
            - 'labels': (num_boundaries,) all ones (for compatibility)
            - 'num_boundaries': int, number of boundaries

    Example:
        >>> labels = np.array([0.0, 0.1, 0.9, 1.0, 0.9, 0.1, 0.0, ...])
        >>> targets = dense_labels_to_set_targets(labels, window_size=18)
        >>> print(targets)
        {'positions': tensor([0.1667, 0.5000, 0.8333]),
         'labels': tensor([1., 1., 1.]),
         'num_boundaries': 3}
    """
    # Find peaks in the label curve
    peaks, properties = find_peaks(
        labels,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) == 0:
        # No boundaries detected
        return {
            'positions': torch.tensor([], dtype=torch.float32),
            'labels': torch.tensor([], dtype=torch.float32),
            'num_boundaries': 0
        }

    # Normalize positions to [0, 1]
    normalized_positions = peaks / float(window_size)

    return {
        'positions': torch.tensor(normalized_positions, dtype=torch.float32),
        'labels': torch.ones(len(peaks), dtype=torch.float32),
        'num_boundaries': len(peaks)
    }


def collate_detr_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DETR training batches.

    Converts dense labels to set-based targets and handles variable-length
    boundary sets using padding.

    Args:
        batch: List of samples, each containing:
            - 'dino_features': List of tensors or tensor
            - 'proprio_features': (seq_len,) tensor
            - 'train_labels': (window_size,) tensor of dense labels
            - 'episode_id', 'task_id', etc.

    Returns:
        Batched dictionary with:
            - 'dino_features': (batch, num_frames * 1024) stacked tensor
            - 'tokenized_actions': (batch, seq_len) stacked tensor
            - 'targets': List of dicts with set-based targets
            - 'episode_ids', 'task_ids', etc.
    """
    # Extract fields
    batch_size = len(batch)
    window_size = len(batch[0]['train_labels'])

    # Stack DINO features
    dino_features_list = []
    for sample in batch:
        dino_feats = sample['dino_features']
        if isinstance(dino_feats, list):
            # List of tensors -> concatenate
            dino_feats = torch.cat(dino_feats)
        dino_features_list.append(dino_feats)

    dino_features_batch = torch.stack(dino_features_list)

    # Stack proprioceptive features
    proprio_batch = torch.stack([sample['proprio_features'] for sample in batch])

    # Convert dense labels to set-based targets
    targets = []
    for sample in batch:
        dense_labels = sample['train_labels'].numpy()
        set_target = dense_labels_to_set_targets(
            dense_labels,
            window_size=window_size,
            threshold=0.5,  # Peak threshold
            min_distance=1  # Minimum 1 frame between boundaries
        )
        targets.append(set_target)

    # Collect metadata
    episode_ids = [sample.get('episode_id', 0) for sample in batch]
    task_ids = [sample.get('task_id', 0) for sample in batch]

    return {
        'dino_features': dino_features_batch,
        'tokenized_actions': proprio_batch,
        'targets': targets,
        'episode_ids': episode_ids,
        'task_ids': task_ids,
        # Keep original dense labels for debugging
        'dense_labels': torch.stack([sample['train_labels'] for sample in batch])
    }


def compute_detr_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    window_size: int,
    confidence_threshold: float = 0.5,
    tolerances: List[float] = None,
    confidence_thresholds_for_map: List[float] = None,
    compute_map: bool = True
) -> Dict[str, float]:
    """Compute evaluation metrics for DETR predictions at multiple tolerances.

    Args:
        predictions: Model output with 'pred_positions' and 'pred_confidences'
        targets: List of ground truth targets
        window_size: Window size in frames
        confidence_threshold: Minimum confidence to count as detection
        tolerances: List of matching tolerances in frames
        confidence_thresholds_for_map: List of confidence thresholds for mAP calculation
        compute_map: Whether to compute mAP (expensive, disable during training for speed)

    Returns:
        Dictionary with precision, recall, F1 at each tolerance, counts, and optionally mAP
    """
    if tolerances is None:
        tolerances = [2.0]
    if confidence_thresholds_for_map is None:
        confidence_thresholds_for_map = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    pred_positions = predictions['pred_positions']  # (batch, num_queries)
    pred_confidences = predictions['pred_confidences']

    batch_size = pred_positions.shape[0]

    # Compute metrics for each tolerance
    results = {}

    for tolerance in tolerances:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for b in range(batch_size):
            # Get predictions above threshold
            valid_mask = pred_confidences[b] > confidence_threshold
            pred_pos = pred_positions[b][valid_mask]

            if len(pred_pos) == 0:
                pred_frames = []
            else:
                # Convert to frame indices
                pred_frames = (pred_pos * window_size).detach().cpu().numpy()

            # Get ground truth
            gt_pos = targets[b]['positions']
            if len(gt_pos) == 0:
                gt_frames = []
            else:
                gt_frames = (gt_pos * window_size).detach().cpu().numpy()

            # Match predictions to GT
            matched_pred = set()
            matched_gt = set()

            for i, pred_frame in enumerate(pred_frames):
                for j, gt_frame in enumerate(gt_frames):
                    if j not in matched_gt and abs(pred_frame - gt_frame) <= tolerance:
                        matched_pred.add(i)
                        matched_gt.add(j)
                        break

            tp = len(matched_pred)
            fp = len(pred_frames) - tp
            fn = len(gt_frames) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Compute metrics for this tolerance
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f'precision@{tolerance}'] = precision
        results[f'recall@{tolerance}'] = recall
        results[f'f1@{tolerance}'] = f1
        results[f'true_positives@{tolerance}'] = total_tp
        results[f'false_positives@{tolerance}'] = total_fp
        results[f'false_negatives@{tolerance}'] = total_fn

    # Also add simple metrics for first tolerance (for backward compat)
    first_tol = tolerances[0]
    results['precision'] = results[f'precision@{first_tol}']
    results['recall'] = results[f'recall@{first_tol}']
    results['f1'] = results[f'f1@{first_tol}']

    # Skip expensive mAP computation during training for speed
    if not compute_map:
        return results

    # Compute mAP (mean Average Precision) across tolerances and confidence thresholds
    ap_per_tolerance = {}

    for tolerance in tolerances:
        # Collect precision-recall pairs at different confidence thresholds
        pr_pairs = []

        for conf_thresh in sorted(confidence_thresholds_for_map):
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for b in range(batch_size):
                # Get predictions above threshold
                valid_mask = pred_confidences[b] > conf_thresh
                pred_pos = pred_positions[b][valid_mask]
                pred_conf = pred_confidences[b][valid_mask]

                if len(pred_pos) == 0:
                    pred_frames = []
                else:
                    # Convert to frame indices
                    pred_frames = (pred_pos * window_size).detach().cpu().numpy()

                # Get ground truth
                gt_pos = targets[b]['positions']
                if len(gt_pos) == 0:
                    gt_frames = []
                else:
                    gt_frames = (gt_pos * window_size).detach().cpu().numpy()

                # Match predictions to GT
                matched_pred = set()
                matched_gt = set()

                for i, pred_frame in enumerate(pred_frames):
                    for j, gt_frame in enumerate(gt_frames):
                        if j not in matched_gt and abs(pred_frame - gt_frame) <= tolerance:
                            matched_pred.add(i)
                            matched_gt.add(j)
                            break

                tp = len(matched_pred)
                fp = len(pred_frames) - tp
                fn = len(gt_frames) - len(matched_gt)

                total_tp += tp
                total_fp += fp
                total_fn += fn

            # Compute precision and recall at this confidence threshold
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

            pr_pairs.append((recall, precision))

        # Sort by recall (ascending)
        pr_pairs = sorted(pr_pairs, key=lambda x: x[0])

        # Compute Average Precision using 11-point interpolation
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            # Find maximum precision for recall >= recall_threshold
            precisions_above_threshold = [p for r, p in pr_pairs if r >= recall_threshold]
            if len(precisions_above_threshold) > 0:
                ap += max(precisions_above_threshold)
        ap /= 11.0

        ap_per_tolerance[tolerance] = ap
        results[f'ap@{tolerance}'] = ap

    # Compute mAP as average of APs across tolerances
    if len(ap_per_tolerance) > 0:
        results['map'] = sum(ap_per_tolerance.values()) / len(ap_per_tolerance)
    else:
        results['map'] = 0.0

    # Compute edit distance (Levenshtein distance) for boundary sequences
    total_edit_distance = 0
    total_sequences = 0

    for b in range(batch_size):
        # Get predictions above threshold
        valid_mask = pred_confidences[b] > confidence_threshold
        pred_pos = pred_positions[b][valid_mask]

        if len(pred_pos) == 0:
            pred_frames = []
        else:
            # Convert to frame indices and sort
            pred_frames = sorted((pred_pos * window_size).detach().cpu().numpy().tolist())

        # Get ground truth
        gt_pos = targets[b]['positions']
        if len(gt_pos) == 0:
            gt_frames = []
        else:
            gt_frames = sorted((gt_pos * window_size).detach().cpu().numpy().tolist())

        # Compute Levenshtein distance
        # Using dynamic programming
        m, n = len(pred_frames), len(gt_frames)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if abs(pred_frames[i-1] - gt_frames[j-1]) <= 1.0:  # Consider same if within 1 frame
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        total_edit_distance += dp[m][n]
        total_sequences += 1

    # Average edit distance across batch
    results['edit_distance'] = total_edit_distance / total_sequences if total_sequences > 0 else 0.0

    # Normalized edit distance (0 to 1, where 0 is perfect match)
    max_possible_distance = sum(max(len(pred_positions[b][pred_confidences[b] > confidence_threshold]),
                                     len(targets[b]['positions'])) for b in range(batch_size))
    results['normalized_edit_distance'] = (total_edit_distance / max_possible_distance
                                           if max_possible_distance > 0 else 0.0)

    # Compute Segmental F1@IoU at standard action segmentation thresholds (10%, 25%, 50%)
    iou_thresholds = [0.1, 0.25, 0.5]

    for iou_threshold in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for b in range(batch_size):
            # Get predictions above threshold
            valid_mask = pred_confidences[b] > confidence_threshold
            pred_pos = pred_positions[b][valid_mask]

            if len(pred_pos) == 0:
                pred_frames = np.array([])
            else:
                # Convert to frame indices
                pred_frames = (pred_pos * window_size).detach().cpu().numpy()

            # Get ground truth
            gt_pos = targets[b]['positions']
            if len(gt_pos) == 0:
                gt_frames = np.array([])
            else:
                gt_frames = (gt_pos * window_size).detach().cpu().numpy()

            # Convert boundaries to segments
            pred_segments = boundaries_to_segments(pred_frames, window_size)
            gt_segments = boundaries_to_segments(gt_frames, window_size)

            # Compute F1@IoU for this sample
            sample_metrics = f1_at_iou(pred_segments, gt_segments, iou_threshold)

            total_tp += sample_metrics['tp']
            total_fp += sample_metrics['fp']
            total_fn += sample_metrics['fn']

        # Compute aggregate metrics
        seg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        seg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if (seg_precision + seg_recall) > 0 else 0.0

        # Store with clear naming to distinguish from boundary-level F1
        iou_pct = int(iou_threshold * 100)
        results[f'seg_f1@{iou_pct}'] = seg_f1
        results[f'seg_precision@{iou_pct}'] = seg_precision
        results[f'seg_recall@{iou_pct}'] = seg_recall

    # Also report framewise accuracy (for compatibility with action segmentation literature)
    # Note: DETR doesn't directly predict frame-level labels, so we approximate this
    # by converting predictions and targets back to dense sequences
    total_correct = 0
    total_frames = 0

    for b in range(batch_size):
        # Get predictions above threshold
        valid_mask = pred_confidences[b] > confidence_threshold
        pred_pos = pred_positions[b][valid_mask]

        if len(pred_pos) == 0:
            pred_frames = np.array([])
        else:
            pred_frames = (pred_pos * window_size).detach().cpu().numpy()

        # Get ground truth
        gt_pos = targets[b]['positions']
        if len(gt_pos) == 0:
            gt_frames = np.array([])
        else:
            gt_frames = (gt_pos * window_size).detach().cpu().numpy()

        # Convert to dense binary sequences (1 at boundaries, 0 elsewhere)
        pred_dense = np.zeros(window_size)
        gt_dense = np.zeros(window_size)

        for frame in pred_frames:
            frame_idx = int(np.round(frame))
            if 0 <= frame_idx < window_size:
                pred_dense[frame_idx] = 1

        for frame in gt_frames:
            frame_idx = int(np.round(frame))
            if 0 <= frame_idx < window_size:
                gt_dense[frame_idx] = 1

        # Count correct predictions
        total_correct += np.sum(pred_dense == gt_dense)
        total_frames += window_size

    # Framewise accuracy
    results['framewise_accuracy'] = total_correct / total_frames if total_frames > 0 else 0.0

    return results
