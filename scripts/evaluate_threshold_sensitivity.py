#!/usr/bin/env python3
"""Evaluate model performance across different classification thresholds.

This script evaluates a trained model with various classification/confidence
thresholds (0.1 to 0.9) to study the impact on all metrics for both
in-distribution and out-of-distribution tasks.

The script runs inference once, then computes metrics for all thresholds
in parallel for efficiency.

Usage:
    python scripts/evaluate_threshold_sensitivity.py \
        --checkpoint ./checkpoints/multi_task_ablation_baseline/best_model.pt \
        --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --output-dir ./threshold_sensitivity_results

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


def run_inference_on_tasks(
    checkpoint_path: str,
    config_path: str,
    task_ids: List[int],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Run inference on all tasks and collect predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        task_ids: List of task IDs to evaluate
        device: Device to run on

    Returns:
        Dictionary containing predictions and targets for each task
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

    # Collect predictions for all tasks
    all_task_data = {}

    for task_id in task_ids:
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

                all_task_data[f'task_{task_id}'] = {
                    'architecture': 'detr',
                    'pred_positions': pred_positions.numpy(),  # Convert to numpy to avoid shared memory issues
                    'pred_confidences': pred_confidences.numpy(),  # Convert to numpy
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

                all_task_data[f'task_{task_id}'] = {
                    'architecture': 'mlp',
                    'predictions': all_predictions,
                    'targets': all_targets,
                    'metric_tolerances': config.training.metric_tolerances
                }

            print(f"✓ Inference complete for task {task_id}")

        except Exception as e:
            print(f"ERROR running inference on task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            all_task_data[f'task_{task_id}'] = {'status': 'failed', 'error': str(e)}

    return all_task_data


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

    results = {}
    for task_key, task_info in task_data.items():
        if task_info.get('architecture') != 'detr' or 'status' in task_info:
            continue

        # Convert numpy back to torch tensors
        predictions = {
            'pred_positions': torch.from_numpy(task_info['pred_positions']),
            'pred_confidences': torch.from_numpy(task_info['pred_confidences'])
        }

        # Compute metrics with specified threshold
        metrics = compute_detr_metrics(
            predictions=predictions,
            targets=task_info['targets'],
            window_size=task_info['window_size'],
            confidence_threshold=threshold,
            tolerances=task_info['metric_tolerances'],
            confidence_thresholds_for_map=task_info['confidence_thresholds_for_map']
        )

        results[task_key] = metrics

    return threshold, results


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

    results = {}
    for task_key, task_info in task_data.items():
        if task_info.get('architecture') != 'mlp' or 'status' in task_info:
            continue

        all_predictions = task_info['predictions']
        all_targets = task_info['targets']
        metric_tolerances = task_info['metric_tolerances']

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

        results[task_key] = metrics

    return threshold, results


def compute_metrics_for_all_thresholds(
    task_data: Dict[str, Any],
    thresholds: List[float],
    num_parallel: int = 4
) -> Dict[float, Dict[str, Any]]:
    """Compute metrics for all thresholds in parallel using Ray.

    Args:
        task_data: Dictionary containing predictions and targets for all tasks
        thresholds: List of thresholds to evaluate
        num_parallel: Number of parallel workers (used for Ray resource allocation)

    Returns:
        Dictionary mapping threshold to metrics for all tasks
    """
    # Determine architecture from first task
    architecture = None
    for task_key, task_info in task_data.items():
        if 'architecture' in task_info:
            architecture = task_info['architecture']
            break

    if architecture is None:
        print("ERROR: Could not determine architecture")
        return {}

    print(f"\nComputing metrics for {len(thresholds)} thresholds using Ray with {num_parallel} workers...")
    print(f"Architecture: {architecture}")

    # Select appropriate Ray remote function
    compute_fn = compute_metrics_for_threshold_detr if architecture == 'detr' else compute_metrics_for_threshold_mlp

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
                    threshold, results = ray.get(future)
                    all_results[threshold] = results
                    pbar.update(1)
                except Exception as e:
                    print(f"Error computing metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)

    return all_results


def run_threshold_sensitivity_study(
    checkpoint_path: str,
    config_path: str,
    indist_task_ids: List[int],
    ood_task_ids: List[int],
    thresholds: List[float],
    output_dir: str,
    num_parallel: int = 4
):
    """Run threshold sensitivity study.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        indist_task_ids: In-distribution task IDs
        ood_task_ids: Out-of-distribution task IDs
        thresholds: List of thresholds to evaluate
        output_dir: Output directory for results
        num_parallel: Number of parallel jobs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Thresholds: {thresholds}")
    print(f"In-dist tasks ({len(indist_task_ids)}): {indist_task_ids}")
    print(f"OOD tasks ({len(ood_task_ids)}): {ood_task_ids}")
    print(f"Output: {output_dir}")
    print(f"Parallel workers: {num_parallel}")
    print(f"{'='*80}\n")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(num_cpus=num_parallel)
        print(f"Ray initialized with {num_parallel} CPUs\n")

    # Step 1: Run inference on all in-distribution tasks
    print("\n" + "="*80)
    print("STEP 1: Running inference on IN-DISTRIBUTION tasks")
    print("="*80)
    indist_data = run_inference_on_tasks(checkpoint_path, config_path, indist_task_ids)

    # Step 2: Run inference on all OOD tasks
    print("\n" + "="*80)
    print("STEP 2: Running inference on OUT-OF-DISTRIBUTION tasks")
    print("="*80)
    ood_data = run_inference_on_tasks(checkpoint_path, config_path, ood_task_ids)

    # Step 3: Compute metrics for all thresholds (in-distribution)
    print("\n" + "="*80)
    print("STEP 3: Computing metrics for all thresholds (IN-DISTRIBUTION)")
    print("="*80)
    indist_results = compute_metrics_for_all_thresholds(indist_data, thresholds, num_parallel)

    # Step 4: Compute metrics for all thresholds (OOD)
    print("\n" + "="*80)
    print("STEP 4: Computing metrics for all thresholds (OUT-OF-DISTRIBUTION)")
    print("="*80)
    ood_results = compute_metrics_for_all_thresholds(ood_data, thresholds, num_parallel)

    # Step 5: Organize and save results
    print("\n" + "="*80)
    print("STEP 5: Organizing and saving results")
    print("="*80)

    # Combine results
    organized_results = {}
    for threshold in thresholds:
        organized_results[f"threshold_{threshold:.1f}_indist"] = {
            'threshold': threshold,
            'eval_type': 'indist',
            'results': indist_results.get(threshold, {})
        }
        organized_results[f"threshold_{threshold:.1f}_ood"] = {
            'threshold': threshold,
            'eval_type': 'ood',
            'results': ood_results.get(threshold, {})
        }

    # Save raw results
    raw_results_file = output_dir / 'threshold_sensitivity_raw_results.json'

    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in organized_results.items():
        serializable_results[key] = {
            'threshold': value['threshold'],
            'eval_type': value['eval_type'],
            'results': value['results']
        }

    with open(raw_results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Raw results saved to: {raw_results_file}")

    # Create summary dataframes
    summary_data = []

    for threshold in thresholds:
        for eval_type in ['indist', 'ood']:
            key = f"threshold_{threshold:.1f}_{eval_type}"
            if key in organized_results:
                result = organized_results[key]
                task_results = result['results']

                # Aggregate metrics across tasks
                all_metrics = {}
                num_tasks = 0

                for task_key, task_metrics in task_results.items():
                    if isinstance(task_metrics, dict) and 'status' not in task_metrics:
                        num_tasks += 1
                        for metric_name, value in task_metrics.items():
                            if isinstance(value, (int, float)):
                                if metric_name not in all_metrics:
                                    all_metrics[metric_name] = []
                                all_metrics[metric_name].append(value)

                # Compute averages
                if num_tasks > 0:
                    row = {
                        'threshold': threshold,
                        'eval_type': eval_type,
                        'num_tasks': num_tasks
                    }

                    for metric_name, values in all_metrics.items():
                        row[f'avg_{metric_name}'] = np.mean(values)
                        row[f'std_{metric_name}'] = np.std(values)

                    summary_data.append(row)

    # Create summary dataframe
    if summary_data:
        df = pd.DataFrame(summary_data)

        # Save summary CSV
        summary_file = output_dir / 'threshold_sensitivity_summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")

        # Create separate CSVs for in-dist and OOD
        for eval_type in ['indist', 'ood']:
            df_filtered = df[df['eval_type'] == eval_type]
            if not df_filtered.empty:
                csv_file = output_dir / f'threshold_sensitivity_{eval_type}.csv'
                df_filtered.to_csv(csv_file, index=False)
                print(f"{eval_type.upper()} results saved to: {csv_file}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"THRESHOLD SENSITIVITY SUMMARY")
        print(f"{'='*80}\n")

        # Print key metrics at each threshold
        key_metrics = ['avg_f1@2.0', 'avg_seg_f1@50', 'avg_framewise_accuracy', 'avg_edit_distance']
        for eval_type in ['indist', 'ood']:
            print(f"\n{eval_type.upper()} Performance:")
            print("-" * 80)
            df_filtered = df[df['eval_type'] == eval_type]

            # Create a display with key metrics
            display_cols = ['threshold'] + [col for col in key_metrics if col in df_filtered.columns]
            if len(display_cols) > 1:
                print(df_filtered[display_cols].to_string(index=False))

    print(f"\n{'='*80}")
    print(f"Threshold sensitivity analysis complete!")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")

    # Shutdown Ray
    if ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate model threshold sensitivity',
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
        help='Path to model configuration file'
    )

    parser.add_argument(
        '--indist-task-ids',
        type=int,
        nargs='+',
        default=[374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354, 360, 372, 359, 378, 390, 377, 384],
        help='In-distribution task IDs'
    )

    parser.add_argument(
        '--ood-task-ids',
        type=int,
        nargs='+',
        default=[380, 424, 392],
        help='Out-of-distribution task IDs'
    )

    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='Thresholds to evaluate'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./threshold_sensitivity_results',
        help='Output directory for results'
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

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Verify config exists
    if not Path(args.config).exists():
        print(f"ERROR: Config not found: {args.config}")
        return

    # Run threshold sensitivity study
    run_threshold_sensitivity_study(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        indist_task_ids=args.indist_task_ids,
        ood_task_ids=args.ood_task_ids,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
        num_parallel=args.num_parallel
    )


if __name__ == '__main__':
    main()
