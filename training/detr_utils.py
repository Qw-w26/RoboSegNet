"""Utilities for DETR-style boundary detection training.

This module provides helper functions for converting between dense labels
and set-based targets used by DETR models.
"""

import torch
import numpy as np
from typing import List, Dict, Any
from scipy.signal import find_peaks


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
    confidence_thresholds_for_map: List[float] = None
) -> Dict[str, float]:
    """Compute evaluation metrics for DETR predictions at multiple tolerances.

    Args:
        predictions: Model output with 'pred_positions' and 'pred_confidences'
        targets: List of ground truth targets
        window_size: Window size in frames
        confidence_threshold: Minimum confidence to count as detection
        tolerances: List of matching tolerances in frames
        confidence_thresholds_for_map: List of confidence thresholds for mAP calculation

    Returns:
        Dictionary with precision, recall, F1 at each tolerance, counts, and mAP
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

    return results
