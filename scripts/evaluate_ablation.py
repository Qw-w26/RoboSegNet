#!/usr/bin/env python3
"""Evaluation script for ablation studies.

This script evaluates a trained model checkpoint on specified tasks and
computes F1 scores at different granularities (tolerances).

Usage:
    python scripts/evaluate_ablation.py \
        --checkpoint ./checkpoints/best_model.pt \
        --config ./configs/ablations/ablation_pure_proprio.yaml \
        --task-ids 327 352 356 \
        --output-dir ./ablation_results/evaluations/pure_proprio

Author: Claude Code
Date: November 2025
"""

import argparse
import sys
import os
from pathlib import Path
import json
import yaml
from typing import List, Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models import create_model
from training import LazyBoundaryDataset, TrainingDatasetConfig, collate_dict
from training.detr_utils import collate_detr_batch, compute_detr_metrics
from data import load_multi_task_split


def evaluate_detr_model(
    model,
    dataloader,
    device,
    window_size: int,
    metric_tolerances: List[float],
    confidence_threshold: float = 0.5,
    confidence_thresholds_for_map: List[float] = None
) -> Dict[str, float]:
    """Evaluate DETR model on dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        window_size: Window size in frames
        metric_tolerances: List of tolerances for F1 calculation
        confidence_threshold: Confidence threshold for predictions

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            tokenized_actions = batch.get('tokenized_actions')
            dino_features = batch.get('dino_features')
            targets = batch.get('targets')

            if tokenized_actions is not None:
                tokenized_actions = tokenized_actions.to(device)
            if dino_features is not None:
                dino_features = dino_features.to(device)

            # Forward pass
            outputs = model(tokenized_actions, None, dino_features)

            # Collect predictions and targets (keep on CPU)
            all_predictions.append({
                'pred_positions': outputs['pred_positions'].cpu(),
                'pred_confidences': outputs['pred_confidences'].cpu()
            })
            all_targets.extend(targets)

    # Concatenate all predictions
    pred_positions = torch.cat([p['pred_positions'] for p in all_predictions], dim=0)
    pred_confidences = torch.cat([p['pred_confidences'] for p in all_predictions], dim=0)

    predictions = {
        'pred_positions': pred_positions,
        'pred_confidences': pred_confidences
    }

    # Compute metrics
    metrics = compute_detr_metrics(
        predictions=predictions,
        targets=all_targets,
        window_size=window_size,
        confidence_threshold=confidence_threshold,
        tolerances=metric_tolerances,
        confidence_thresholds_for_map=confidence_thresholds_for_map
    )

    return metrics


def evaluate_mlp_model(
    model,
    dataloader,
    device,
    metric_tolerances: List[float],
    classification_threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate MLP model on dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        metric_tolerances: List of tolerances for F1 calculation
        classification_threshold: Threshold for binary classification

    Returns:
        Dictionary of metrics
    """
    from scipy.signal import find_peaks
    from evaluation.metrics import BoundaryMetrics

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            tokenized_actions = batch.get('tokenized_actions')
            dino_features = batch.get('dino_features')
            labels = batch.get('label')

            if tokenized_actions is not None:
                tokenized_actions = tokenized_actions.to(device)
            if dino_features is not None:
                dino_features = dino_features.to(device)
            if labels is not None:
                labels = labels.to(device)

            # Forward pass
            logits = model(tokenized_actions, None, dino_features)

            # Convert to probabilities
            probs = torch.sigmoid(logits)

            # Collect predictions and targets
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics using BoundaryMetrics
    metrics_calculator = BoundaryMetrics(iou_thresholds=[])

    # Frame-level metrics
    frame_metrics = metrics_calculator._calculate_classification_metrics(
        predictions=all_predictions.flatten(),
        ground_truth=all_targets.flatten()
    )

    # Compute boundary-level F1 at different tolerances
    boundary_metrics = {}
    for tol in metric_tolerances:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i in range(len(all_predictions)):
            # Extract boundaries from predictions (peaks)
            pred_peaks, _ = find_peaks(all_predictions[i], height=classification_threshold, distance=1)

            # Extract boundaries from targets (peaks)
            target_peaks, _ = find_peaks(all_targets[i], height=0.5, distance=1)

            # Match predictions to targets
            matched_pred = set()
            matched_target = set()

            for pred_idx in pred_peaks:
                for target_idx in target_peaks:
                    if target_idx not in matched_target and abs(pred_idx - target_idx) <= tol:
                        matched_pred.add(pred_idx)
                        matched_target.add(target_idx)
                        break

            tp = len(matched_pred)
            fp = len(pred_peaks) - tp
            fn = len(target_peaks) - len(matched_target)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        boundary_metrics[f'precision@{tol}'] = precision
        boundary_metrics[f'recall@{tol}'] = recall
        boundary_metrics[f'f1@{tol}'] = f1

    # Combine metrics
    metrics = {**frame_metrics, **boundary_metrics}

    return metrics


def evaluate_model_on_tasks(
    checkpoint_path: str,
    config_path: str,
    task_ids: List[int],
    output_dir: str,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Evaluate model on specified tasks.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        task_ids: List of task IDs to evaluate
        output_dir: Output directory for results
        device: Device to run on

    Returns:
        Dictionary of results per task
    """
    # Load config
    config = load_config(config_path)

    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = create_model(config.model, device=device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP wrapper
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"Model loaded successfully")

    # Determine architecture
    is_detr = getattr(config.model, 'architecture', 'mlp') == 'boundary_detr'

    # Evaluate on each task
    results = {}
    for task_id in task_ids:
        print(f"\n{'='*80}")
        print(f"Evaluating Task {task_id}")
        print(f"{'='*80}\n")

        try:
            # Load task data
            print(f"Loading data for task {task_id}...")
            val_trajectories = load_multi_task_split(
                'val',
                task_ids=[task_id],
                meta_dir=config.data.meta_dir
            )

            # Create dataset
            dataset_config = TrainingDatasetConfig(
                window_size=config.data.window_size,
                stride=config.data.stride,
                vocab_size=config.data.vocab_size,
                fps=config.data.fps,
                device=str(device),
                label_smoother_type=getattr(config.data, 'label_smoother_type', 'gaussian'),
                label_smoother_sigma=getattr(config.data, 'label_smoother_sigma', 1.0),
                label_smoother_peak_value=getattr(config.data, 'label_smoother_peak_value', 1.0),
                action_dim=config.data.action_dim,
                num_dino_frames=getattr(config.data, 'num_dino_frames', getattr(config.model, 'num_dino_frames', 3)),
                max_samples=None
            )

            val_dataset = LazyBoundaryDataset(val_trajectories, dataset_config)

            print(f"Dataset size: {len(val_dataset)}")

            # Create dataloader
            collate_fn = collate_detr_batch if is_detr else collate_dict
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )

            # Evaluate
            print("Running evaluation...")
            if is_detr:
                task_metrics = evaluate_detr_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    window_size=config.model.prediction_window_size,
                    metric_tolerances=config.training.metric_tolerances,
                    confidence_threshold=getattr(config.training, 'confidence_threshold', 0.5),
                    confidence_thresholds_for_map=getattr(config.training, 'confidence_thresholds_for_map', None)
                )
            else:
                task_metrics = evaluate_mlp_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    metric_tolerances=config.training.metric_tolerances,
                    classification_threshold=getattr(config.training, 'classification_threshold', 0.5)
                )

            results[f'task_{task_id}'] = task_metrics

            # Print metrics
            print(f"\nResults for Task {task_id}:")
            for key, value in task_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        except Exception as e:
            print(f"ERROR evaluating task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            results[f'task_{task_id}'] = {'status': 'failed', 'error': str(e)}

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate ablation study model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        required=True,
        help='Task IDs to evaluate'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    results = evaluate_model_on_tasks(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        task_ids=args.task_ids,
        output_dir=args.output_dir,
        device=args.device
    )

    print("\nFinal Results Summary:")
    print("=" * 80)
    for task_key, metrics in results.items():
        print(f"\n{task_key}:")
        if isinstance(metrics, dict) and 'status' not in metrics:
            # Print F1 scores at different tolerances
            for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
                key = f'f1@{tol}'
                if key in metrics:
                    print(f"  F1@{tol}: {metrics[key]:.4f}")


if __name__ == '__main__':
    main()
