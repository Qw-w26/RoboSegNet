#!/usr/bin/env python3
"""Evaluate ablation study checkpoints across different classification thresholds.

This script evaluates trained ablation baseline models (pure proprio, pure visual,
MLP + MSE) with various classification/confidence thresholds to study the impact
on all metrics.

Similar to evaluate_threshold_sensitivity.py but specifically for ablation baselines
trained by run_ablation_study.py.

Usage:
    # Evaluate all ablation baselines with thresholds [0.5, 0.6, 0.7, 0.8, 0.9]
    python scripts/evaluate_ablation_threshold_sensitivity.py \
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --output-dir ./ablation_threshold_sensitivity_results

    # Evaluate specific ablations
    python scripts/evaluate_ablation_threshold_sensitivity.py \
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --ablations pure_proprio pure_visual mlp_mse \
        --output-dir ./ablation_threshold_sensitivity_results

    # Evaluate with specific thresholds
    python scripts/evaluate_ablation_threshold_sensitivity.py \
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --thresholds 0.8 \
        --output-dir ./ablation_threshold_sensitivity_results

    # Evaluate specific tasks only
    python scripts/evaluate_ablation_threshold_sensitivity.py \
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --task-ids 327 352 356 \
        --output-dir ./ablation_threshold_sensitivity_results

Author: Claude Code
Date: November 2025
"""

import argparse
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import ray

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from models import create_model
from training import LazyBoundaryDataset, TrainingDatasetConfig
from training.detr_utils import collate_detr_batch, compute_detr_metrics
from training.training_dataset import collate_dict
from data import load_multi_task_split


# Ablation configurations (matching run_ablation_study.py)
ABLATION_CONFIGS = {
    'pure_proprio': {
        'name': 'Pure Proprioceptive',
        'base_exp_name': 'ablation_pure_proprio'
    },
    'pure_visual': {
        'name': 'Pure Visual',
        'base_exp_name': 'ablation_pure_visual'
    },
    'mlp_mse': {
        'name': 'MLP + MSE',
        'base_exp_name': 'ablation_mlp_mse'
    },
    'single_task': {
        'name': 'Single Task Baseline',
        'base_exp_name': 'ablation_single_task'
    }
}


def run_inference_on_task(
    checkpoint_path: str,
    config_path: str,
    task_id: int,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Run inference on a single task and collect predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        task_id: Task ID to evaluate
        device: Device to run on

    Returns:
        Dictionary containing predictions and targets
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

    print(f"\n{'='*80}")
    print(f"Running inference on Task {task_id}")
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
            max_samples=None,
            proprio_subsample_rate=getattr(config.data, 'proprio_subsample_rate', None)
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

        # Run inference
        print("Running inference...")
        if is_detr:
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Task {task_id}"):
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

            task_data = {
                'architecture': 'detr',
                'pred_positions': pred_positions.numpy(),
                'pred_confidences': pred_confidences.numpy(),
                'targets': all_targets,
                'window_size': config.model.prediction_window_size,
                'metric_tolerances': config.training.metric_tolerances,
                'confidence_thresholds_for_map': getattr(config.training, 'confidence_thresholds_for_map', None)
            }

        else:  # MLP architecture
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Task {task_id}"):
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

            task_data = {
                'architecture': 'mlp',
                'predictions': all_predictions,
                'targets': all_targets,
                'metric_tolerances': config.training.metric_tolerances
            }

        print(f"✓ Inference complete for task {task_id}")
        return task_data

    except Exception as e:
        print(f"ERROR running inference on task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


@ray.remote
def compute_metrics_for_threshold_detr(threshold, task_data):
    """Compute DETR metrics for a specific threshold (Ray remote function).

    Args:
        threshold: Threshold value
        task_data: Dictionary containing task data

    Returns:
        Tuple of (threshold, metrics_dict)
    """
    import sys
    from pathlib import Path
    import torch

    # Add project root to path (needed in Ray worker)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.detr_utils import compute_detr_metrics

    if task_data.get('architecture') != 'detr' or 'status' in task_data:
        return threshold, {'status': 'failed'}

    # Convert numpy back to torch tensors
    predictions = {
        'pred_positions': torch.from_numpy(task_data['pred_positions']),
        'pred_confidences': torch.from_numpy(task_data['pred_confidences'])
    }

    # Compute metrics with specified threshold
    metrics = compute_detr_metrics(
        predictions=predictions,
        targets=task_data['targets'],
        window_size=task_data['window_size'],
        confidence_threshold=threshold,
        tolerances=task_data['metric_tolerances'],
        confidence_thresholds_for_map=task_data['confidence_thresholds_for_map']
    )

    return threshold, metrics


@ray.remote
def compute_metrics_for_threshold_mlp(threshold, task_data):
    """Compute MLP metrics for a specific threshold (Ray remote function).

    Args:
        threshold: Threshold value
        task_data: Dictionary containing task data

    Returns:
        Tuple of (threshold, metrics_dict)
    """
    import sys
    from pathlib import Path
    from scipy.signal import find_peaks

    # Add project root to path (needed in Ray worker)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.metrics import BoundaryMetrics
    from evaluation.f1_iou import f1_at_iou
    from training.detr_utils import boundaries_to_segments

    if task_data.get('architecture') != 'mlp' or 'status' in task_data:
        return threshold, {'status': 'failed'}

    all_predictions = task_data['predictions']
    all_targets = task_data['targets']
    metric_tolerances = task_data['metric_tolerances']

    # Compute metrics using BoundaryMetrics
    metrics_calculator = BoundaryMetrics(iou_thresholds=[])

    # Frame-level metrics with specified threshold
    preds_binary = (all_predictions.flatten() >= threshold).astype(float)
    frame_metrics = metrics_calculator._calculate_classification_metrics(
        predictions=preds_binary,
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
            pred_peaks, _ = find_peaks(all_predictions[i], height=threshold, distance=1)

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

    # Compute edit distance for boundary sequences
    total_edit_distance = 0
    total_sequences = 0

    for i in range(len(all_predictions)):
        # Extract boundaries from predictions (peaks)
        pred_peaks, _ = find_peaks(all_predictions[i], height=threshold, distance=1)
        pred_frames = sorted(pred_peaks.tolist())

        # Extract boundaries from targets (peaks)
        target_peaks, _ = find_peaks(all_targets[i], height=0.5, distance=1)
        gt_frames = sorted(target_peaks.tolist())

        # Compute Levenshtein distance using dynamic programming
        m, n = len(pred_frames), len(gt_frames)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for j in range(m + 1):
            dp[j][0] = j
        for k in range(n + 1):
            dp[0][k] = k

        # Fill DP table
        for j in range(1, m + 1):
            for k in range(1, n + 1):
                if abs(pred_frames[j-1] - gt_frames[k-1]) <= 1.0:
                    dp[j][k] = dp[j-1][k-1]
                else:
                    dp[j][k] = 1 + min(
                        dp[j-1][k],    # deletion
                        dp[j][k-1],    # insertion
                        dp[j-1][k-1]   # substitution
                    )

        total_edit_distance += dp[m][n]
        total_sequences += 1

    # Average edit distance
    edit_distance = total_edit_distance / total_sequences if total_sequences > 0 else 0.0

    # Normalized edit distance
    max_possible_distance = sum(max(len(find_peaks(all_predictions[i], height=threshold, distance=1)[0]),
                                     len(find_peaks(all_targets[i], height=0.5, distance=1)[0]))
                                for i in range(len(all_predictions)))
    normalized_edit_distance = (total_edit_distance / max_possible_distance
                                 if max_possible_distance > 0 else 0.0)

    # Compute Segmental F1@IoU at standard thresholds (10%, 25%, 50%)
    iou_thresholds = [0.1, 0.25, 0.5]
    window_size = all_predictions.shape[1]

    segmental_metrics = {}
    for iou_threshold in iou_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i in range(len(all_predictions)):
            # Extract boundaries from predictions (peaks)
            pred_peaks, _ = find_peaks(all_predictions[i], height=threshold, distance=1)
            pred_frames = pred_peaks.astype(float)

            # Extract boundaries from targets (peaks)
            target_peaks, _ = find_peaks(all_targets[i], height=0.5, distance=1)
            gt_frames = target_peaks.astype(float)

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

        iou_pct = int(iou_threshold * 100)
        segmental_metrics[f'seg_f1@{iou_pct}'] = seg_f1
        segmental_metrics[f'seg_precision@{iou_pct}'] = seg_precision
        segmental_metrics[f'seg_recall@{iou_pct}'] = seg_recall

    # Combine metrics
    metrics = {
        **frame_metrics,
        **boundary_metrics,
        **segmental_metrics,
        'edit_distance': edit_distance,
        'normalized_edit_distance': normalized_edit_distance,
        'framewise_accuracy': frame_metrics['accuracy']
    }

    return threshold, metrics


def evaluate_ablation_with_thresholds(
    ablation_name: str,
    task_id: int,
    thresholds: List[float],
    ablation_root: str = './ablation_results',
    output_root: str = './ablation_threshold_sensitivity_results',
    num_parallel: int = 4
) -> Dict[str, Any]:
    """Evaluate a single ablation checkpoint across different thresholds.

    Args:
        ablation_name: Name of ablation (e.g., 'pure_proprio')
        task_id: Task ID to evaluate
        thresholds: List of thresholds to evaluate
        ablation_root: Root directory of ablation results
        output_root: Root directory for threshold sensitivity results
        num_parallel: Number of parallel workers

    Returns:
        Dictionary containing results for all thresholds
    """
    ablation_root = Path(ablation_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Get ablation config
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    ablation_config = ABLATION_CONFIGS[ablation_name]
    base_exp_name = ablation_config['base_exp_name']
    exp_name = f'{base_exp_name}_{task_id}'

    print(f"\n{'='*80}")
    print(f"EVALUATING: {ablation_config['name']} - Task {task_id}")
    print(f"{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Thresholds: {thresholds}")
    print(f"{'='*80}\n")

    # Find checkpoint
    checkpoint_dir = Path(f'./checkpoints/{exp_name}')
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return {'status': 'checkpoint_not_found', 'error': f'Directory not found: {checkpoint_dir}'}

    checkpoints = list(checkpoint_dir.glob('best_*.pt'))
    if not checkpoints:
        print(f"ERROR: No checkpoint found in {checkpoint_dir}")
        return {'status': 'checkpoint_not_found', 'error': f'No checkpoint in {checkpoint_dir}'}

    checkpoint_path = checkpoints[0]
    print(f"Using checkpoint: {checkpoint_path}")

    # Find config
    config_path = ablation_root / 'configs' / f'{exp_name}.yaml'
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return {'status': 'config_not_found', 'error': f'Config not found: {config_path}'}

    print(f"Using config: {config_path}")

    # Run inference
    print("\nRunning inference...")
    task_data = run_inference_on_task(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        task_id=task_id
    )

    if 'status' in task_data and task_data['status'] == 'failed':
        return task_data

    # Compute metrics for all thresholds
    print(f"\nComputing metrics for {len(thresholds)} thresholds using Ray...")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(num_cpus=num_parallel)
        print(f"Ray initialized with {num_parallel} CPUs\n")

    # Determine architecture and select appropriate metric computation function
    architecture = task_data.get('architecture', 'mlp')
    print(f"Architecture: {architecture}")

    if architecture == 'detr':
        compute_fn = compute_metrics_for_threshold_detr
    else:
        compute_fn = compute_metrics_for_threshold_mlp

    # Submit all tasks to Ray
    futures = [compute_fn.remote(threshold, task_data) for threshold in thresholds]

    # Collect results as they complete
    all_results = {}
    with tqdm(total=len(thresholds), desc="Computing metrics") as pbar:
        while futures:
            # Wait for at least one task to complete
            done, futures = ray.wait(futures, num_returns=1)

            for future in done:
                try:
                    threshold, metrics = ray.get(future)
                    all_results[threshold] = metrics
                    pbar.update(1)
                except Exception as e:
                    print(f"Error computing metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)

    return {
        'ablation': ablation_name,
        'ablation_name': ablation_config['name'],
        'task_id': task_id,
        'threshold_results': all_results,
        'status': 'completed'
    }


def run_full_evaluation(
    ablation_names: List[str],
    task_ids: List[int],
    thresholds: List[float],
    ablation_root: str = './ablation_results',
    output_root: str = './ablation_threshold_sensitivity_results',
    num_parallel: int = 4
):
    """Run threshold sensitivity evaluation for all ablations and tasks.

    Args:
        ablation_names: List of ablation names to evaluate
        task_ids: List of task IDs to evaluate
        thresholds: List of thresholds to evaluate
        ablation_root: Root directory of ablation results
        output_root: Root directory for threshold sensitivity results
        num_parallel: Number of parallel workers
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ABLATION THRESHOLD SENSITIVITY EVALUATION")
    print(f"{'='*80}")
    print(f"Ablations: {ablation_names}")
    print(f"Tasks: {task_ids}")
    print(f"Thresholds: {thresholds}")
    print(f"Output: {output_root}")
    print(f"{'='*80}\n")

    # Initialize Ray
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(num_cpus=num_parallel)
        print(f"Ray initialized with {num_parallel} CPUs\n")

    all_results = []

    # Evaluate each ablation-task combination
    for ablation_name in ablation_names:
        for task_id in task_ids:
            result = evaluate_ablation_with_thresholds(
                ablation_name=ablation_name,
                task_id=task_id,
                thresholds=thresholds,
                ablation_root=ablation_root,
                output_root=output_root,
                num_parallel=num_parallel
            )
            all_results.append(result)

            # Save incremental results
            results_file = output_root / 'threshold_sensitivity_raw_results.json'
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Generate summary CSV
    print(f"\n{'='*80}")
    print(f"GENERATING SUMMARY")
    print(f"{'='*80}\n")

    summary_rows = []
    for result in all_results:
        if result.get('status') != 'completed':
            continue

        ablation_name = result['ablation_name']
        task_id = result['task_id']
        threshold_results = result['threshold_results']

        for threshold, metrics in threshold_results.items():
            if isinstance(metrics, dict) and 'status' not in metrics:
                row = {
                    'ablation': ablation_name,
                    'task_id': task_id,
                    'threshold': threshold
                }

                # Add all metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[metric_name] = value

                summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)

        # Save summary CSV
        summary_file = output_root / 'ablation_threshold_sensitivity_summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")

        # Print key metrics
        print(f"\n{'='*80}")
        print(f"KEY METRICS SUMMARY")
        print(f"{'='*80}\n")

        # Group by ablation and threshold, show average across tasks
        key_metrics = ['f1@2.0', 'seg_f1@50', 'framewise_accuracy', 'edit_distance']
        for ablation in ablation_names:
            ablation_display = ABLATION_CONFIGS[ablation]['name']
            print(f"\n{ablation_display}:")
            print("-" * 80)

            df_ablation = df[df['ablation'] == ablation_display]
            if not df_ablation.empty:
                # Group by threshold and compute averages
                grouped = df_ablation.groupby('threshold')[key_metrics].mean()
                print(grouped.to_string())

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results directory: {output_root}")
    print(f"{'='*80}\n")

    # Shutdown Ray
    if ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate ablation baselines with threshold sensitivity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-config',
        type=str,
        required=True,
        help='Path to base configuration file'
    )

    parser.add_argument(
        '--ablation-root',
        type=str,
        default='./ablation_results',
        help='Root directory of ablation results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ablation_threshold_sensitivity_results',
        help='Output directory for threshold sensitivity results'
    )

    parser.add_argument(
        '--ablations',
        type=str,
        nargs='+',
        choices=list(ABLATION_CONFIGS.keys()) + ['all'],
        default=['all'],
        help='Ablations to evaluate'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        help='Task IDs to evaluate (default: use all from base config)'
    )

    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help='Thresholds to evaluate'
    )

    parser.add_argument(
        '--num-parallel',
        type=int,
        default=4,
        help='Number of parallel workers for metric computation'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load base config to get task IDs if not specified
    base_config = load_config(args.base_config)

    # Get task IDs
    task_ids = args.task_ids
    if task_ids is None:
        # Use training task IDs from base config
        task_ids = base_config.data.train_task_ids

    # Get ablation names
    ablation_names = args.ablations
    if 'all' in ablation_names:
        ablation_names = list(ABLATION_CONFIGS.keys())

    # Run evaluation
    run_full_evaluation(
        ablation_names=ablation_names,
        task_ids=task_ids,
        thresholds=args.thresholds,
        ablation_root=args.ablation_root,
        output_root=args.output_dir,
        num_parallel=args.num_parallel
    )


if __name__ == '__main__':
    main()
