#!/usr/bin/env python3
"""Run multi-task MS-TCN baseline experiment.

This script trains and evaluates a pure visual MS-TCN model in a multi-task setting.
Trains ONE model on ALL training tasks simultaneously using torchrun with 8 GPUs,
then evaluates on both in-distribution and out-of-distribution tasks in parallel.

Training set: 374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354,
              360, 372, 359, 378, 390, 377, 384 (18 tasks)
OOD test set: 380, 424, 392 (3 tasks)

MS-TCN variant:
- Pure visual MS-TCN (2 stages, 5 layers, 64 feature maps)
- Uses only DINO visual features (no proprioception)
- Trained with 8 GPUs using torchrun
- Evaluated in parallel across 8 GPUs

Output metrics:
Boundary Detection Metrics:
- F1@1.0, F1@2.0, F1@3.0, etc. (Boundary F1 at different frame tolerances)
- AP@1.0, AP@2.0, AP@3.0, etc. (Average Precision at different tolerances)
- mAP (mean Average Precision)

Action Segmentation Metrics (Standard):
- Seg-F1@10%, Seg-F1@25%, Seg-F1@50% (Segmental F1 at IoU thresholds)
- Edit-Dist (Segmental edit distance / Levenshtein distance)
- Acc (Framewise accuracy)

Usage:
    # Run with default settings (8 GPUs)
    python scripts/run_multi_task_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Quick test run (fewer epochs)
    python scripts/run_multi_task_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --quick-test

    # Custom number of GPUs
    python scripts/run_multi_task_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-gpus 4

Author: Claude Code
Date: November 2025
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import yaml
import json
from typing import List, Dict, Any
import pandas as pd
import multiprocessing as mp
from functools import partial

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config


class MultiTaskMSTCNBaselineStudy:
    """Manages multi-task MS-TCN baseline experiment."""

    # Single variant: pure visual MS-TCN
    VARIANT_CONFIG = {
        'name': 'MS-TCN Visual Only',
        'description': 'MS-TCN with only visual features (no proprioception)',
        'modifications': {
            'model.architecture': 'ms_tcn',
            'ms_tcn.num_stages': 2,
            'ms_tcn.num_layers': 5,
            'ms_tcn.num_f_maps': 64,
            'model.use_proprio': False,  # MS-TCN uses only visual features
            'model.use_dino': True,
            'training.loss_type': 'bce',  # Use BCE with sigmoid for proper convergence
            'training.loss_pos_weight': 10.0,
            'training.optimizer': 'adam',
            'experiment_name': 'multi_task_mstcn_visual_only'
        }
    }

    def __init__(
        self,
        base_config_path: str,
        train_task_ids: List[int],
        ood_task_ids: List[int] = None,
        output_root: str = './multi_task_ms_tcn_results',
        quick_test: bool = False,
        num_gpus: int = 8
    ):
        """Initialize multi-task MS-TCN baseline study.

        Args:
            base_config_path: Path to base configuration file
            train_task_ids: List of task IDs to train on
            ood_task_ids: List of OOD task IDs to evaluate on
            output_root: Root directory for results
            quick_test: If True, use reduced epochs for testing
            num_gpus: Number of GPUs to use for training
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        self.train_task_ids = train_task_ids
        self.num_gpus = num_gpus
        self.output_root = Path(output_root)
        self.quick_test = quick_test

        # Auto-detect OOD tasks if not specified
        if ood_task_ids is None:
            meta_dir = Path(self.base_config.data.meta_dir)
            all_tasks = []
            for task_dir in sorted(meta_dir.glob('task_*_processed')):
                task_id = int(task_dir.name.split('_')[1])
                all_tasks.append(task_id)

            self.ood_task_ids = [tid for tid in all_tasks if tid >= 400]
            print(f"Auto-detected OOD tasks (>= 400): {self.ood_task_ids}")
        else:
            self.ood_task_ids = ood_task_ids

        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = {}

    def create_config(self) -> Dict[str, Any]:
        """Create configuration for MS-TCN model.

        Returns:
            Modified configuration dictionary
        """
        # Load base config
        config_dict = yaml.safe_load(open(self.base_config_path))

        # Get modifications
        modifications = self.VARIANT_CONFIG['modifications'].copy()

        # Multi-task training: use experiment name as-is (no task ID suffix)
        exp_name = modifications['experiment_name']

        # Set checkpoint directory
        modifications['training.checkpoint_dir'] = f'./checkpoints/{exp_name}'

        # Disable wandb
        modifications['wandb.enabled'] = False

        # Set task IDs for multi-task training
        modifications['data.train_task_ids'] = self.train_task_ids
        modifications['data.eval_task_ids'] = self.train_task_ids  # Evaluate on training tasks during training

        # Quick test modifications
        if self.quick_test:
            modifications['training.num_epochs'] = 5
            modifications['training.early_stopping'] = True
            modifications['training.patience'] = 3
            modifications['data.max_samples'] = 1000

        # Apply modifications
        for key, value in modifications.items():
            keys = key.split('.')
            target = config_dict
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

        return config_dict

    def save_config(self, config: Dict[str, Any], output_path: Path):
        """Save configuration to YAML file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _check_training_status(self, checkpoint_dir: Path) -> Dict[str, Any]:
        """Check if training is complete or can be resumed.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Dictionary with status information
        """
        if not checkpoint_dir.exists():
            return {
                'status': 'not_started',
                'checkpoint_path': None,
                'message': 'No checkpoint directory found'
            }

        # Check if training completed (final_model.pt exists)
        final_checkpoint = checkpoint_dir / 'final_model.pt'
        if final_checkpoint.exists():
            return {
                'status': 'complete',
                'checkpoint_path': str(final_checkpoint),
                'message': 'Training already completed'
            }

        # Find latest checkpoint to resume from
        latest_checkpoint = self._find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            return {
                'status': 'resumable',
                'checkpoint_path': str(latest_checkpoint),
                'message': f'Can resume from {latest_checkpoint.name}'
            }

        return {
            'status': 'not_started',
            'checkpoint_path': None,
            'message': 'No valid checkpoints found'
        }

    def _find_latest_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Find the latest checkpoint in a directory."""
        if not checkpoint_dir.exists():
            return None

        # Look for periodic checkpoints
        periodic_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if periodic_checkpoints:
            def get_epoch(path):
                try:
                    return int(path.stem.split('_')[-1])
                except (ValueError, IndexError):
                    return -1

            periodic_checkpoints.sort(key=get_epoch, reverse=True)
            return periodic_checkpoints[0]

        # If no periodic checkpoints, check for best_model.pt
        best_checkpoint = checkpoint_dir / 'best_model.pt'
        if best_checkpoint.exists():
            return best_checkpoint

        return None

    def run_training(self, config_path: Path, checkpoint_dir: Path) -> Dict[str, Any]:
        """Run training using torchrun with multiple GPUs."""
        print(f"\n{'='*80}")
        print(f"Training: {config_path.stem}")
        print(f"Config: {config_path}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"GPUs: {self.num_gpus}")
        print(f"{'='*80}\n")

        # Check training status
        status_info = self._check_training_status(checkpoint_dir)
        print(f"Training status: {status_info['message']}")

        # If training is complete, skip
        if status_info['status'] == 'complete':
            print("Training already completed, skipping...")
            best_checkpoint = checkpoint_dir / 'best_model.pt'
            final_checkpoint = checkpoint_dir / 'final_model.pt'

            checkpoint_path = best_checkpoint if best_checkpoint.exists() else final_checkpoint

            return {
                'status': 'completed',
                'checkpoint': str(checkpoint_path)
            }

        # Training command using torchrun
        cmd = [
            # 'torchrun',
            # f'--nproc_per_node={self.num_gpus}',
            # '--standalone',
            "python3",
            'scripts/train.py',
            '--config', str(config_path)
        ]

        # Add resume flag if we can resume
        if status_info['status'] == 'resumable':
            resume_checkpoint = status_info['checkpoint_path']
            print(f"Resuming training from: {resume_checkpoint}")
            cmd.extend(['--resume', resume_checkpoint])

        # Run training
        print(f"Running command: {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"WARNING: Training failed")
            return {'status': 'failed'}

        # Find best checkpoint
        best_checkpoint = None
        if checkpoint_dir.exists():
            best_model = checkpoint_dir / 'best_model.pt'
            if best_model.exists():
                best_checkpoint = best_model
            else:
                checkpoints = list(checkpoint_dir.glob('best_*.pt'))
                if checkpoints:
                    best_checkpoint = checkpoints[0]
                else:
                    final_model = checkpoint_dir / 'final_model.pt'
                    if final_model.exists():
                        best_checkpoint = final_model

        return {
            'status': 'completed',
            'checkpoint': str(best_checkpoint) if best_checkpoint else None
        }

    def run_evaluation(
        self,
        checkpoint_path: Path,
        config_path: Path,
        task_ids: List[int],
        output_dir: Path,
        eval_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """Run evaluation on checkpoint."""
        print(f"\n{'='*80}")
        print(f"Evaluating ({eval_name}): {checkpoint_path.stem}")
        print(f"Config: {config_path}")
        print(f"Tasks: {task_ids}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")

        cmd = [
            'python', 'scripts/evaluate_ablation.py',
            '--checkpoint', str(checkpoint_path),
            '--config', str(config_path),
            '--task-ids'] + [str(tid) for tid in task_ids] + [
            '--output-dir', str(output_dir)
        ]

        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"WARNING: Evaluation failed")
            return {'status': 'failed'}

        # Load results
        results_file = output_dir / 'evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            return {'status': 'failed'}

    def _evaluate_task_subset(self, checkpoint_path, config_path, task_ids, output_dir, gpu_id):
        """Evaluate a subset of tasks on a specific GPU."""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        result = self.run_evaluation(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            task_ids=task_ids,
            output_dir=output_dir,
            eval_name=f"GPU-{gpu_id}"
        )

        return result

    def run_parallel_evaluation(
        self,
        checkpoint_path: Path,
        config_path: Path,
        task_ids: List[int],
        output_dir: Path,
        eval_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """Run evaluation in parallel across multiple GPUs."""
        print(f"\n{'='*80}")
        print(f"Parallel Evaluation ({eval_name}): {checkpoint_path.stem}")
        print(f"Tasks: {task_ids} ({len(task_ids)} tasks)")
        print(f"GPUs: {self.num_gpus}")
        print(f"{'='*80}\n")

        # Split tasks across GPUs
        tasks_per_gpu = (len(task_ids) + self.num_gpus - 1) // self.num_gpus
        task_chunks = [task_ids[i:i+tasks_per_gpu] for i in range(0, len(task_ids), tasks_per_gpu)]

        # Evaluate each chunk sequentially (one GPU at a time to avoid conflicts)
        all_results = {}
        for gpu_id, task_chunk in enumerate(task_chunks):
            if not task_chunk:
                continue

            print(f"Evaluating tasks {task_chunk} on GPU {gpu_id}...")
            chunk_output_dir = output_dir / f'gpu_{gpu_id}'

            result = self.run_evaluation(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                task_ids=task_chunk,
                output_dir=chunk_output_dir,
                eval_name=f"{eval_name}_gpu{gpu_id}"
            )

            # Merge results
            if result and 'status' not in result:
                all_results.update(result)

        # Save combined results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        return all_results

    def run(self):
        """Run the multi-task MS-TCN baseline experiment."""
        # Create config
        config = self.create_config()

        # Setup paths
        exp_name = config['experiment_name']
        config_path = self.output_root / 'configs' / f'{exp_name}.yaml'
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        eval_dir = self.output_root / 'evaluations' / exp_name

        # Save config
        self.save_config(config, config_path)

        # Run training
        train_result = self.run_training(config_path, checkpoint_dir)
        if train_result['status'] != 'completed':
            print("Training failed!")
            return

        # Run evaluation on both in-distribution and OOD tasks
        checkpoint = train_result.get('checkpoint')

        if checkpoint is None and checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('best_*.pt'))
            if checkpoints:
                checkpoint = checkpoints[0]

        if checkpoint:
            # Evaluate on in-distribution tasks in parallel
            print(f"\n{'='*80}")
            print("Evaluating on IN-DISTRIBUTION tasks")
            print(f"{'='*80}")
            indist_eval_dir = eval_dir / 'indist'
            indist_result = self.run_parallel_evaluation(
                checkpoint_path=Path(checkpoint),
                config_path=config_path,
                task_ids=self.train_task_ids,
                output_dir=indist_eval_dir,
                eval_name="in-distribution"
            )

            # Save in-dist results
            indist_results_file = eval_dir / 'indist_evaluation_results.json'
            indist_results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(indist_results_file, 'w') as f:
                json.dump(indist_result, f, indent=2)

            # Evaluate on OOD tasks in parallel
            if self.ood_task_ids:
                print(f"\n{'='*80}")
                print("Evaluating on OUT-OF-DISTRIBUTION tasks")
                print(f"{'='*80}")
                ood_eval_dir = eval_dir / 'ood'
                ood_result = self.run_parallel_evaluation(
                    checkpoint_path=Path(checkpoint),
                    config_path=config_path,
                    task_ids=self.ood_task_ids,
                    output_dir=ood_eval_dir,
                    eval_name="out-of-distribution"
                )

                # Save OOD results
                ood_results_file = eval_dir / 'ood_evaluation_results.json'
                with open(ood_results_file, 'w') as f:
                    json.dump(ood_result, f, indent=2)
            else:
                ood_result = None

            # Compile results
            self.results = {
                'name': self.VARIANT_CONFIG['name'],
                'description': self.VARIANT_CONFIG['description'],
                'config_path': str(config_path),
                'checkpoint_dir': str(checkpoint_dir),
                'training': train_result,
                'evaluation_indist': indist_result,
                'evaluation_ood': ood_result,
                'status': 'completed'
            }

            # Generate report
            self.generate_summary_report()
        else:
            print("No checkpoint found for evaluation!")


    def _print_aggregate_metrics(self, evaluation: Dict[str, Any]):
        """Print aggregate metrics across tasks."""
        all_metrics = {}
        num_tasks = 0

        for task_key, task_eval in evaluation.items():
            if isinstance(task_eval, dict) and 'status' not in task_eval:
                num_tasks += 1
                for metric_name, value in task_eval.items():
                    if isinstance(value, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)

        # Print averages
        if num_tasks > 0:
            print(f"  Tasks evaluated: {num_tasks}")
            for metric_name in ['f1@2.0', 'seg_f1@50', 'framewise_accuracy']:
                if metric_name in all_metrics:
                    avg_value = sum(all_metrics[metric_name]) / len(all_metrics[metric_name])
                    print(f"  avg_{metric_name}: {avg_value:.4f}")

    def _update_summary_csvs(self):
        """Update summary CSV files."""
        for eval_type in ['indist', 'ood']:
            eval_key = f'evaluation_{eval_type}'

            if eval_key not in self.results or not self.results[eval_key]:
                continue

            summary_rows = []
            for task_key, task_eval in self.results[eval_key].items():
                if isinstance(task_eval, dict) and 'status' not in task_eval:
                    row = {
                        'method': self.results['name'],
                        'task': task_key,
                        'num_stages': 2,
                        'num_layers': 5,
                        'status': self.results.get('status', 'unknown')
                    }

                    # Add metrics
                    for tol in [1.0, 2.0, 3.0]:
                        if f'f1@{tol}' in task_eval:
                            row[f'F1@{tol}'] = task_eval[f'f1@{tol}']
                        if f'ap@{tol}' in task_eval:
                            row[f'AP@{tol}'] = task_eval[f'ap@{tol}']

                    if 'map' in task_eval:
                        row['mAP'] = task_eval['map']

                    for iou_pct in [10, 25, 50]:
                        if f'seg_f1@{iou_pct}' in task_eval:
                            row[f'Seg-F1@{iou_pct}%'] = task_eval[f'seg_f1@{iou_pct}']

                    if 'edit_distance' in task_eval:
                        row['Edit-Dist'] = task_eval['edit_distance']
                    if 'framewise_accuracy' in task_eval:
                        row['Acc'] = task_eval['framewise_accuracy']

                    summary_rows.append(row)

            if summary_rows:
                df = pd.DataFrame(summary_rows)
                csv_file = self.output_root / f'multi_task_ms_tcn_summary_{eval_type}.csv'
                df.to_csv(csv_file, index=False)

    def generate_summary_report(self):
        """Generate final summary report."""
        print(f"\n{'='*80}")
        print(f"MULTI-TASK MS-TCN BASELINE FINAL SUMMARY")
        print(f"{'='*80}\n")

        # Save JSON
        results_file = self.output_root / 'multi_task_ms_tcn_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results: {results_file}")

        # Update CSV
        self._update_summary_csvs()

        # Print aggregate metrics
        if 'evaluation_indist' in self.results and self.results['evaluation_indist']:
            print(f"\nIn-Distribution Performance:")
            self._print_aggregate_metrics(self.results['evaluation_indist'])

        if 'evaluation_ood' in self.results and self.results['evaluation_ood']:
            print(f"\nOut-of-Distribution Performance:")
            self._print_aggregate_metrics(self.results['evaluation_ood'])

        # Display summaries
        for eval_type in ['indist', 'ood']:
            csv_file = self.output_root / f'multi_task_ms_tcn_summary_{eval_type}.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                print(f"\n{eval_type.upper()} Summary ({len(df)} task evaluations):")
                print(df.to_string(index=False))
                print(f"\nCSV: {csv_file}")

        print(f"\n{'='*80}")
        print(f"Multi-task MS-TCN baseline study complete!")
        print(f"{'='*80}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run multi-task MS-TCN baseline experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-config',
        type=str,
        required=True,
        help='Path to base configuration file'
    )

    parser.add_argument(
        '--output-root',
        type=str,
        default='./multi_task_ms_tcn_results',
        help='Root directory for results'
    )

    parser.add_argument(
        '--train-task-ids',
        type=int,
        nargs='+',
        help='Task IDs to train on (default: 18 tasks)'
    )

    parser.add_argument(
        '--ood-task-ids',
        type=int,
        nargs='+',
        help='OOD task IDs to evaluate on (default: [380, 424, 392])'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (fewer epochs)'
    )

    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for training and evaluation'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print(f"\n{'='*80}")
    print("MULTI-TASK MS-TCN BASELINE")
    print(f"{'='*80}\n")

    # Get training task IDs
    if args.train_task_ids:
        train_task_ids = args.train_task_ids
    else:
        train_task_ids = [374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354,
                         360, 372, 359, 378, 390, 377, 384]

    # Get OOD task IDs
    if args.ood_task_ids:
        ood_task_ids = args.ood_task_ids
    else:
        ood_task_ids = [380, 424, 392]

    print(f"Training tasks: {train_task_ids}")
    print(f"OOD tasks: {ood_task_ids}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {args.output_root}\n")

    # Initialize study
    study = MultiTaskMSTCNBaselineStudy(
        base_config_path=args.base_config,
        train_task_ids=train_task_ids,
        ood_task_ids=ood_task_ids,
        output_root=args.output_root,
        quick_test=args.quick_test,
        num_gpus=args.num_gpus
    )

    # Run the experiment
    study.run()


if __name__ == '__main__':
    main()
