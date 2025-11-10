#!/usr/bin/env python3
"""Run frequency sensitivity ablation study for multi-task boundary detection.

This script studies how the frequency/sampling rate of different modalities affects
model performance:

1. Visual frequency (num_dino_frames): How many DINO frames to sample per window
   - Ablation values: 6, 9, 12, 15, 18 frames
   - Controls temporal resolution of visual features

2. Proprioceptive frequency: How frequently to sample robot proprioceptive data
   - Current: 30Hz (detected from data)
   - Ablation values: 30Hz, 15Hz, 5Hz, 2Hz
   - Controls temporal resolution of proprioceptive features

Key differences from multi-task ablation:
- All models use the same architecture (multi-task baseline)
- Only frequency/sampling parameters vary
- No checkpoints saved (config: save_best_only=False, save_every=None)

Data alignment:
- Window-level operations: 2 Hz (18 frames = 9 seconds)
- DINO sampling: num_dino_frames evenly spaced across 18 frames using np.linspace
  - Example (num_dino_frames=9): indices [0, 2, 4, 6, 9, 11, 13, 15, 17]
  - Example (num_dino_frames=6): indices [0, 3, 7, 10, 14, 17]
- Proprioceptive sampling: native frequency (30Hz) mapped to window via freq_multiplier
  - freq_multiplier = proprio_fps / window_fps = 30 / 2 = 15x
  - Each window extracts: window_size * freq_multiplier = 18 * 15 = 270 samples
  - For subsampling, we reduce the effective frequency before tokenization

Usage:
    # Run all ablations
    python scripts/run_frequency_sensitivity_ablation.py \\
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Run only visual frequency ablations
    python scripts/run_frequency_sensitivity_ablation.py \\
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \\
        --ablation-type visual

    # Run only proprioceptive frequency ablations
    python scripts/run_frequency_sensitivity_ablation.py \\
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \\
        --ablation-type proprio

    # Quick test
    python scripts/run_frequency_sensitivity_ablation.py \\
        --base-config configs/experiments/multi_task_327_352_356_detr.yaml \\
        --quick-test

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
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.callback import Callback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config


class IncrementalResultsCallback(Callback):
    """Ray Tune callback to update results incrementally as trials complete."""

    def __init__(self, ablation_study: 'FrequencySensitivityStudy'):
        """Initialize callback.

        Args:
            ablation_study: Reference to FrequencySensitivityStudy instance
        """
        self.ablation_study = ablation_study
        self.processed_trials = set()

    def on_trial_complete(self, iteration: int, trials, trial, **info):
        """Called when a trial completes.

        Args:
            iteration: Current iteration
            trials: List of all trials
            trial: The trial that just completed
            **info: Additional info
        """
        trial_id = trial.trial_id
        if trial_id in self.processed_trials:
            return

        # Extract trial config
        ablation_name = trial.config.get('ablation_name')

        if not ablation_name:
            return

        # Reconstruct experiment name
        exp_name = self.ablation_study.ABLATION_CONFIGS[ablation_name]['modifications']['experiment_name']

        # Load evaluation results from filesystem
        eval_dir = self.ablation_study.output_root / 'evaluations' / exp_name

        # Load both in-distribution and OOD results
        indist_results_file = eval_dir / 'indist_evaluation_results.json'
        ood_results_file = eval_dir / 'ood_evaluation_results.json'

        if indist_results_file.exists() and ood_results_file.exists():
            with open(indist_results_file, 'r') as f:
                indist_results = json.load(f)
            with open(ood_results_file, 'r') as f:
                ood_results = json.load(f)

            result = {
                'ablation': ablation_name,
                'ablation_name': self.ablation_study.ABLATION_CONFIGS[ablation_name]['name'],
                'description': self.ablation_study.ABLATION_CONFIGS[ablation_name]['description'],
                'config_path': str(self.ablation_study.output_root / 'configs' / f'{exp_name}.yaml'),
                'training': {'status': 'completed'},
                'evaluation_indist': indist_results,
                'evaluation_ood': ood_results,
                'status': 'completed'
            }

            # Update incrementally
            self.ablation_study._update_incremental_results(result)
            self.processed_trials.add(trial_id)


class FrequencySensitivityStudy:
    """Manages frequency sensitivity ablation study experiments."""

    # Multi-modal (baseline) visual frequency ablations
    VISUAL_FREQ_ABLATIONS = {
        f'visual_freq_{n}frames': {
            'name': f'Visual {n} Frames',
            'description': f'Multi-modal: Sample {n} DINO frames per 18-frame window',
            'modifications': {
                'model.num_dino_frames': n,
                'data.num_dino_frames': n,
                'experiment_name': f'freq_ablation_visual_{n}frames'
            }
        }
        for n in [6, 9, 12, 15, 18]
    }

    # Multi-modal (baseline) proprioceptive frequency ablations
    PROPRIO_FREQ_ABLATIONS = {
        f'proprio_freq_{freq}hz': {
            'name': f'Proprio {freq}Hz',
            'description': f'Multi-modal: Sample proprioceptive data at {freq}Hz (from 30Hz)',
            'modifications': {
                'data.proprio_subsample_rate': freq,  # Target frequency
                'experiment_name': f'freq_ablation_proprio_{freq}hz'
            }
        }
        for freq in [30, 15, 5, 2]
    }

    # Pure proprioceptive frequency ablations (no visual)
    PURE_PROPRIO_FREQ_ABLATIONS = {
        f'pure_proprio_freq_{freq}hz': {
            'name': f'Pure Proprio {freq}Hz',
            'description': f'Pure proprioceptive (no visual): Sample at {freq}Hz',
            'modifications': {
                'model.use_proprio': True,
                'model.use_dino': False,
                'model.architecture': 'mlp',
                'training.loss_type': 'mse_raw',
                'training.optimizer': 'adam',
                'data.proprio_subsample_rate': freq,
                'experiment_name': f'freq_ablation_pure_proprio_{freq}hz'
            }
        }
        for freq in [30, 15, 5, 2]
    }

    # Pure visual frequency ablations (no proprioceptive)
    PURE_VISUAL_FREQ_ABLATIONS = {
        f'pure_visual_freq_{n}frames': {
            'name': f'Pure Visual {n} Frames',
            'description': f'Pure visual (no proprio): Sample {n} DINO frames per window',
            'modifications': {
                'model.use_proprio': False,
                'model.use_dino': True,
                'model.architecture': 'mlp',
                'training.loss_type': 'mse_raw',
                'training.optimizer': 'adam',
                'model.num_dino_frames': n,
                'data.num_dino_frames': n,
                'experiment_name': f'freq_ablation_pure_visual_{n}frames'
            }
        }
        for n in [6, 9, 12, 15, 18]
    }

    def __init__(
        self,
        base_config_path: str,
        train_task_ids: List[int],
        ood_task_ids: List[int] = None,
        output_root: str = './frequency_sensitivity_results',
        quick_test: bool = False,
        num_gpus: int = 1,
        ablation_type: str = 'all'  # 'all', 'visual', or 'proprio'
    ):
        """Initialize frequency sensitivity study.

        Args:
            base_config_path: Path to base configuration file
            train_task_ids: List of task IDs to train on
            ood_task_ids: List of OOD task IDs to evaluate on
            output_root: Root directory for ablation results
            quick_test: If True, use reduced epochs for quick testing
            num_gpus: Number of GPUs to use for each training run
            ablation_type: Type of ablation to run ('all', 'visual', 'proprio')
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        self.train_task_ids = train_task_ids
        self.num_gpus = num_gpus
        self.output_root = Path(output_root)
        self.quick_test = quick_test
        self.ablation_type = ablation_type

        # Select ablations based on type
        if ablation_type == 'visual':
            self.ABLATION_CONFIGS = self.VISUAL_FREQ_ABLATIONS
        elif ablation_type == 'proprio':
            self.ABLATION_CONFIGS = self.PROPRIO_FREQ_ABLATIONS
        elif ablation_type == 'pure_proprio':
            self.ABLATION_CONFIGS = self.PURE_PROPRIO_FREQ_ABLATIONS
        elif ablation_type == 'pure_visual':
            self.ABLATION_CONFIGS = self.PURE_VISUAL_FREQ_ABLATIONS
        elif ablation_type == 'all':
            self.ABLATION_CONFIGS = {
                **self.VISUAL_FREQ_ABLATIONS,
                **self.PROPRIO_FREQ_ABLATIONS,
                **self.PURE_PROPRIO_FREQ_ABLATIONS,
                **self.PURE_VISUAL_FREQ_ABLATIONS
            }
        elif ablation_type == 'multimodal':
            # Only multi-modal (baseline) ablations
            self.ABLATION_CONFIGS = {**self.VISUAL_FREQ_ABLATIONS, **self.PROPRIO_FREQ_ABLATIONS}
        elif ablation_type == 'pure':
            # Only pure (single modality) ablations
            self.ABLATION_CONFIGS = {**self.PURE_PROPRIO_FREQ_ABLATIONS, **self.PURE_VISUAL_FREQ_ABLATIONS}
        else:
            raise ValueError(f"Invalid ablation_type: {ablation_type}. "
                           f"Must be 'all', 'visual', 'proprio', 'pure_visual', 'pure_proprio', 'multimodal', or 'pure'")

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
        self.results = []

    def create_ablation_config(
        self,
        ablation_name: str
    ) -> Dict[str, Any]:
        """Create configuration for frequency sensitivity ablation.

        Args:
            ablation_name: Name of ablation study

        Returns:
            Modified configuration dictionary
        """
        if ablation_name not in self.ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation: {ablation_name}")

        # Load base config as dict
        config_dict = yaml.safe_load(open(self.base_config_path))

        # Get ablation modifications
        ablation = self.ABLATION_CONFIGS[ablation_name]
        modifications = ablation['modifications'].copy()

        # Set unique checkpoint directory for this ablation
        exp_name = modifications['experiment_name']
        modifications['training.checkpoint_dir'] = f'./checkpoints/{exp_name}'

        # Disable checkpoint saving (we don't need checkpoints for frequency ablation)
        modifications['training.save_best_only'] = False
        # Note: We'll still save final model for evaluation, but no intermediate checkpoints

        # Disable wandb for ablation studies
        modifications['wandb.enabled'] = False

        # Set task IDs for multi-task training
        modifications['data.train_task_ids'] = self.train_task_ids
        modifications['data.eval_task_ids'] = self.train_task_ids

        # Quick test modifications
        if self.quick_test:
            modifications['training.num_epochs'] = 5
            modifications['training.early_stopping'] = True
            modifications['training.patience'] = 5
            modifications['data.max_samples'] = 1000

        # Apply modifications using nested key access
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

    def _find_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Path to latest checkpoint, or None if not found
        """
        if not checkpoint_dir.exists():
            return None

        # Look for periodic checkpoints (checkpoint_epoch_N.pt)
        periodic_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if periodic_checkpoints:
            # Sort by epoch number
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

    def run_training(
        self,
        config_path: Path,
        checkpoint_dir: Path,
        num_gpus: int = 1
    ) -> Dict[str, Any]:
        """Run single-GPU training.

        Args:
            config_path: Path to configuration file
            checkpoint_dir: Directory for checkpoints
            num_gpus: Number of GPUs to use

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Training: {config_path.stem}")
        print(f"Config: {config_path}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"GPUs: {num_gpus}")
        print(f"{'='*80}\n")

        # Check training status
        status_info = self._check_training_status(checkpoint_dir)
        print(f"Training status: {status_info['message']}")

        # If training is complete, skip
        if status_info['status'] == 'complete':
            print("Training already completed, skipping...")
            final_checkpoint = checkpoint_dir / 'final_model.pt'
            return {
                'status': 'completed',
                'checkpoint': str(final_checkpoint)
            }

        # Construct python command
        cmd = [
            'python',
            'scripts/train.py',
            '--config', str(config_path)
        ]

        # Add resume flag if we can resume from a checkpoint
        if status_info['status'] == 'resumable':
            resume_checkpoint = status_info['checkpoint_path']
            print(f"Resuming training from: {resume_checkpoint}")
            cmd.extend(['--resume', resume_checkpoint])

        # Run training
        print(f"\nRunning command: {' '.join(cmd)}\n")
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"WARNING: Training failed for {config_path.stem}")
            return {'status': 'failed'}

        # Find checkpoint after training completes
        final_checkpoint = checkpoint_dir / 'final_model.pt'
        if not final_checkpoint.exists():
            # Fallback to best_model.pt
            final_checkpoint = checkpoint_dir / 'best_model.pt'

        return {
            'status': 'completed',
            'checkpoint': str(final_checkpoint) if final_checkpoint.exists() else None
        }

    def run_evaluation(
        self,
        checkpoint_path: Path,
        config_path: Path,
        task_ids: List[int],
        output_dir: Path,
        eval_name: str = "evaluation"
    ) -> Dict[str, Any]:
        """Run evaluation on checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            task_ids: Task IDs to evaluate on
            output_dir: Output directory for results
            eval_name: Name for this evaluation

        Returns:
            Evaluation results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Evaluating ({eval_name}): {checkpoint_path.stem}")
        print(f"Config: {config_path}")
        print(f"Tasks: {task_ids}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")

        # Use evaluate_ablation.py script
        cmd = [
            'python', 'scripts/evaluate_ablation.py',
            '--checkpoint', str(checkpoint_path),
            '--config', str(config_path),
            '--task-ids'] + [str(tid) for tid in task_ids] + [
            '--output-dir', str(output_dir)
        ]

        # Run evaluation
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"WARNING: Evaluation failed")
            return {'status': 'failed'}

        # Load results from JSON
        results_file = output_dir / 'evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            return all_results
        else:
            print(f"WARNING: No results file found at {results_file}")
            return {'status': 'failed'}

    def _run_single_ablation_instance(
        self,
        ablation_name: str,
        run_training: bool,
        run_evaluation: bool
    ) -> Dict[str, Any]:
        """Run single instance of frequency sensitivity ablation."""
        # Create config
        config = self.create_ablation_config(ablation_name)

        # Setup paths
        exp_name = config['experiment_name']
        config_path = self.output_root / 'configs' / f'{exp_name}.yaml'
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        eval_dir = self.output_root / 'evaluations' / exp_name

        # Save config
        self.save_config(config, config_path)
        print(f"Saved config to: {config_path}")

        # Run training
        train_result = None
        if run_training:
            train_result = self.run_training(config_path, checkpoint_dir, self.num_gpus)

            if train_result['status'] != 'completed':
                return {
                    'ablation': ablation_name,
                    'status': 'training_failed',
                    'config_path': str(config_path)
                }

        # Run evaluation on both in-distribution and OOD tasks
        indist_result = None
        ood_result = None

        if run_evaluation:
            checkpoint = train_result.get('checkpoint') if train_result else None

            # Find checkpoint if not provided
            if checkpoint is None and checkpoint_dir.exists():
                final_checkpoint = checkpoint_dir / 'final_model.pt'
                if not final_checkpoint.exists():
                    final_checkpoint = checkpoint_dir / 'best_model.pt'
                if final_checkpoint.exists():
                    checkpoint = final_checkpoint

            if checkpoint:
                # Evaluate on in-distribution tasks
                indist_eval_dir = eval_dir / 'indist'
                indist_result = self.run_evaluation(
                    checkpoint_path=Path(checkpoint),
                    config_path=config_path,
                    task_ids=self.train_task_ids,
                    output_dir=indist_eval_dir,
                    eval_name="in-distribution"
                )

                # Save in-dist results separately
                indist_results_file = eval_dir / 'indist_evaluation_results.json'
                indist_results_file.parent.mkdir(parents=True, exist_ok=True)
                with open(indist_results_file, 'w') as f:
                    json.dump(indist_result, f, indent=2)

                # Evaluate on OOD tasks
                if self.ood_task_ids:
                    ood_eval_dir = eval_dir / 'ood'
                    ood_result = self.run_evaluation(
                        checkpoint_path=Path(checkpoint),
                        config_path=config_path,
                        task_ids=self.ood_task_ids,
                        output_dir=ood_eval_dir,
                        eval_name="out-of-distribution"
                    )

                    # Save OOD results separately
                    ood_results_file = eval_dir / 'ood_evaluation_results.json'
                    with open(ood_results_file, 'w') as f:
                        json.dump(ood_result, f, indent=2)
            else:
                print("WARNING: No checkpoint found for evaluation")

        # Compile results
        result = {
            'ablation': ablation_name,
            'ablation_name': self.ABLATION_CONFIGS[ablation_name]['name'],
            'description': self.ABLATION_CONFIGS[ablation_name]['description'],
            'config_path': str(config_path),
            'checkpoint_dir': str(checkpoint_dir),
            'training': train_result,
            'evaluation_indist': indist_result,
            'evaluation_ood': ood_result,
            'status': 'completed'
        }

        # Save result
        self.results.append(result)

        return result

    def _update_incremental_results(self, new_result: Dict[str, Any]):
        """Update results incrementally after each trial completion."""
        # Add to results list
        self.results.append(new_result)

        # Save detailed results to JSON immediately
        results_file = self.output_root / 'frequency_sensitivity_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Update summary CSVs
        self._update_summary_csvs()

        # Print progress
        print(f"\n{'='*80}")
        print(f"INCREMENTAL UPDATE: {len(self.results)} experiments completed")
        print(f"{'='*80}")
        print(f"Latest: {new_result['ablation_name']}")

        if 'evaluation_indist' in new_result and new_result['evaluation_indist']:
            print(f"\nIn-Distribution Performance:")
            self._print_aggregate_metrics(new_result['evaluation_indist'])

        if 'evaluation_ood' in new_result and new_result['evaluation_ood']:
            print(f"\nOut-of-Distribution Performance:")
            self._print_aggregate_metrics(new_result['evaluation_ood'])

        print(f"{'='*80}\n")

    def _print_aggregate_metrics(self, evaluation: Dict[str, Any]):
        """Print aggregate metrics across tasks."""
        # Collect metrics across all tasks
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
            for metric_name in ['f1@2.0', 'seg_f1@50', 'framewise_accuracy', 'edit_distance']:
                if metric_name in all_metrics:
                    avg_value = sum(all_metrics[metric_name]) / len(all_metrics[metric_name])
                    print(f"  avg_{metric_name}: {avg_value:.4f}")

    def _update_summary_csvs(self):
        """Update the summary CSV files with current results."""
        # Create separate summaries for in-dist and OOD
        for eval_type in ['indist', 'ood']:
            summary_rows = []
            eval_key = f'evaluation_{eval_type}'

            for result in self.results:
                if eval_key in result and result[eval_key]:
                    # Extract metrics for each task
                    for task_key, task_eval in result[eval_key].items():
                        if isinstance(task_eval, dict):
                            if 'status' in task_eval and task_eval['status'] == 'failed':
                                continue

                            row = {
                                'ablation': result['ablation_name'],
                                'task': task_key,
                                'eval_type': eval_type,
                                'status': result.get('status', 'unknown')
                            }

                            # Add metrics if available
                            # Boundary F1 scores
                            for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
                                f1_key = f'f1@{tol}'
                                if f1_key in task_eval:
                                    row[f'F1@{tol}'] = task_eval[f1_key]

                            # Segmental F1@IoU
                            for iou_pct in [10, 25, 50]:
                                seg_f1_key = f'seg_f1@{iou_pct}'
                                if seg_f1_key in task_eval:
                                    row[f'Seg-F1@{iou_pct}%'] = task_eval[seg_f1_key]

                            # Edit distance
                            if 'edit_distance' in task_eval:
                                row['Edit-Dist'] = task_eval['edit_distance']

                            # Framewise accuracy
                            if 'framewise_accuracy' in task_eval:
                                row['Acc'] = task_eval['framewise_accuracy']

                            summary_rows.append(row)

            if summary_rows:
                df = pd.DataFrame(summary_rows)
                csv_file = self.output_root / f'frequency_sensitivity_summary_{eval_type}.csv'
                df.to_csv(csv_file, index=False)

    def _create_aggregate_summary(self):
        """Create aggregate summary with mean metrics across tasks for each ablation."""
        for eval_type in ['indist', 'ood']:
            # Load per-task summary
            csv_file = self.output_root / f'frequency_sensitivity_summary_{eval_type}.csv'
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            # Group by ablation and compute mean
            metric_columns = [col for col in df.columns if col not in ['ablation', 'task', 'eval_type', 'status']]

            agg_data = []
            for ablation in df['ablation'].unique():
                ablation_df = df[df['ablation'] == ablation]

                row = {'ablation': ablation}
                for col in metric_columns:
                    if col in ablation_df.columns:
                        values = ablation_df[col].dropna()
                        if len(values) > 0:
                            row[col] = values.mean()

                agg_data.append(row)

            # Create aggregate dataframe
            if agg_data:
                agg_df = pd.DataFrame(agg_data)

                # Sort columns for better readability
                ordered_cols = ['ablation']
                f1_cols = sorted([c for c in agg_df.columns if c.startswith('F1@')])
                seg_f1_cols = sorted([c for c in agg_df.columns if c.startswith('Seg-F1@')])
                other_cols = sorted([c for c in agg_df.columns if c not in ordered_cols + f1_cols + seg_f1_cols])

                ordered_cols.extend(f1_cols)
                ordered_cols.extend(seg_f1_cols)
                ordered_cols.extend(other_cols)

                # Reorder columns
                agg_df = agg_df[[c for c in ordered_cols if c in agg_df.columns]]

                # Save aggregate summary
                agg_file = self.output_root / f'frequency_sensitivity_aggregate_{eval_type}.csv'
                agg_df.to_csv(agg_file, index=False)

    def _load_completed_trials(self) -> set:
        """Load information about completed trials from previous runs."""
        completed_trials = set()

        # Check if results file exists
        results_file = self.output_root / 'frequency_sensitivity_results.json'
        if not results_file.exists():
            return completed_trials

        try:
            with open(results_file, 'r') as f:
                previous_results = json.load(f)

            # Extract completed trials
            for result in previous_results:
                if result.get('status') == 'completed':
                    ablation_name = result.get('ablation')
                    if ablation_name:
                        completed_trials.add(ablation_name)

            # Also load into self.results for incremental updates
            self.results = previous_results

            if completed_trials:
                print(f"\n{'='*80}")
                print(f"FOUND {len(completed_trials)} COMPLETED TRIALS FROM PREVIOUS RUN")
                print(f"{'='*80}")
                for ablation_name in sorted(completed_trials):
                    ablation_display = self.ABLATION_CONFIGS.get(ablation_name, {}).get('name', ablation_name)
                    print(f"  ✓ {ablation_display}")
                print(f"{'='*80}\n")

        except Exception as e:
            print(f"Warning: Could not load previous results: {e}")

        return completed_trials

    def generate_summary_report(self):
        """Generate final summary report of all results."""
        print(f"\n{'='*80}")
        print(f"FREQUENCY SENSITIVITY STUDY FINAL SUMMARY")
        print(f"{'='*80}\n")

        # Save results to JSON
        results_file = self.output_root / 'frequency_sensitivity_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to: {results_file}")

        # Update summary CSVs one final time
        self._update_summary_csvs()

        # Create aggregate summary
        self._create_aggregate_summary()

        # Display per-task summaries
        for eval_type in ['indist', 'ood']:
            csv_file = self.output_root / f'frequency_sensitivity_summary_{eval_type}.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                print(f"\n{'='*80}")
                print(f"{eval_type.upper()} Results Summary ({len(df)} task evaluations):")
                print(f"{'='*80}")
                print(df.to_string(index=False))
                print(f"\nPer-task CSV: {csv_file}")

        # Display aggregate summaries
        print(f"\n{'='*80}")
        print(f"AGGREGATE PERFORMANCE (Mean Across Tasks)")
        print(f"{'='*80}\n")

        for eval_type in ['indist', 'ood']:
            agg_file = self.output_root / f'frequency_sensitivity_aggregate_{eval_type}.csv'
            if agg_file.exists():
                df = pd.read_csv(agg_file)
                print(f"\n{eval_type.upper()} Performance:")
                print("-" * 80)
                print(df.to_string(index=False))
                print(f"\nAggregate CSV: {agg_file}")

        print(f"\n{'='*80}")
        print(f"Frequency sensitivity study complete!")
        print(f"Results directory: {self.output_root}")
        print(f"{'='*80}\n")

    def run_all_ablations(
        self,
        ablation_names: List[str] = None,
        num_parallel: int = None
    ):
        """Run all frequency sensitivity ablation studies.

        Args:
            ablation_names: List of ablation names (None = all)
            num_parallel: Number of parallel jobs (None = sequential)
        """
        if ablation_names is None:
            ablation_names = list(self.ABLATION_CONFIGS.keys())

        print(f"\n{'='*80}")
        print(f"RUNNING FREQUENCY SENSITIVITY ABLATION STUDY")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config_path}")
        print(f"Training tasks: {self.train_task_ids} ({len(self.train_task_ids)} tasks)")
        print(f"OOD tasks: {self.ood_task_ids} ({len(self.ood_task_ids)} tasks)")
        print(f"Ablation type: {self.ablation_type}")
        print(f"Ablations: {ablation_names}")
        print(f"Output root: {self.output_root}")
        print(f"Quick test: {self.quick_test}")
        print(f"GPUs per ablation: {self.num_gpus}")
        print(f"Max parallel ablations: {num_parallel if num_parallel else 'unlimited'}")
        print(f"{'='*80}\n")

        # Load completed trials from previous runs
        completed_trials = self._load_completed_trials()

        # Filter out completed trials
        trials_to_run = [name for name in ablation_names if name not in completed_trials]

        if len(trials_to_run) < len(ablation_names):
            skipped = set(ablation_names) - set(trials_to_run)
            print(f"\n{'='*80}")
            print(f"SKIPPING {len(skipped)} ALREADY COMPLETED TRIALS")
            print(f"{'='*80}")
            for ablation_name in sorted(skipped):
                ablation_display = self.ABLATION_CONFIGS.get(ablation_name, {}).get('name', ablation_name)
                print(f"  ⊘ {ablation_display}")
            print(f"{'='*80}\n")

        print(f"Total trials to run: {len(trials_to_run)}")

        # If no trials to run, just generate the summary and exit
        if len(trials_to_run) == 0:
            print("\n⚠ All trials already completed! Generating summary report...")
            self.generate_summary_report()
            return

        # Run trials with Ray Tune if parallel execution requested
        if ray.is_initialized() and num_parallel is not None and num_parallel > 1:
            # Store current working directory for Ray workers
            working_dir = os.getcwd()

            trial_configs = []
            for ablation_name in trials_to_run:
                trial_configs.append({
                    'ablation_name': ablation_name,
                    'base_config_path': self.base_config_path,
                    'train_task_ids': self.train_task_ids,
                    'ood_task_ids': self.ood_task_ids,
                    'output_root': str(self.output_root),
                    'quick_test': self.quick_test,
                    'num_gpus': self.num_gpus,
                    'working_dir': working_dir
                })

            # Configure progress reporter
            reporter = CLIReporter(
                max_progress_rows=len(trial_configs) + 5,
                metric_columns=["status", "ablation"]
            )

            # Create callback for incremental updates
            incremental_callback = IncrementalResultsCallback(self)

            # Run trials with Ray Tune
            storage_path = str(Path(working_dir) / self.output_root)
            print(f"Storage path: {storage_path}\n")

            analysis = tune.run(
                run_frequency_trainable,
                config=tune.grid_search(trial_configs),
                resources_per_trial={"cpu": self.num_gpus, "gpu": self.num_gpus},
                max_concurrent_trials=num_parallel,
                verbose=1,
                progress_reporter=reporter,
                storage_path=storage_path,
                name="frequency_sensitivity_study",
                log_to_file=True,
                raise_on_failed_trial=False,
                callbacks=[incremental_callback]
            )

            # Most results have been collected incrementally by the callback
            # Just do a final check for any trials that might have been missed
            print(f"\n{'='*80}")
            print(f"FINALIZING RESULTS")
            print(f"{'='*80}\n")

            processed_trial_ids = incremental_callback.processed_trials

            for trial in analysis.trials:
                trial_id = trial.trial_id
                if trial_id in processed_trial_ids:
                    continue

                if trial.status == "TERMINATED":
                    ablation_name = trial.config['ablation_name']
                    exp_name = self.ABLATION_CONFIGS[ablation_name]['modifications']['experiment_name']

                    # Load evaluation results
                    eval_dir = self.output_root / 'evaluations' / exp_name
                    indist_results_file = eval_dir / 'indist_evaluation_results.json'
                    ood_results_file = eval_dir / 'ood_evaluation_results.json'

                    if indist_results_file.exists() and ood_results_file.exists():
                        with open(indist_results_file, 'r') as f:
                            indist_results = json.load(f)
                        with open(ood_results_file, 'r') as f:
                            ood_results = json.load(f)

                        result = {
                            'ablation': ablation_name,
                            'ablation_name': self.ABLATION_CONFIGS[ablation_name]['name'],
                            'description': self.ABLATION_CONFIGS[ablation_name]['description'],
                            'config_path': str(self.output_root / 'configs' / f'{exp_name}.yaml'),
                            'training': {'status': 'completed'},
                            'evaluation_indist': indist_results,
                            'evaluation_ood': ood_results,
                            'status': 'completed'
                        }

                        self._update_incremental_results(result)
                        print(f"✓ Completed (late): {result['ablation_name']}")
                    else:
                        print(f"✗ Missing results: {exp_name}")
                elif trial.status == "ERROR":
                    ablation_name = trial.config.get('ablation_name', 'unknown')
                    print(f"✗ Failed: {ablation_name} - status: ERROR")
        else:
            # Run sequentially without Ray
            for ablation_name in trials_to_run:
                print(f"\n{'='*80}")
                print(f"Running ablation: {ablation_name}")
                print(f"{'='*80}\n")

                result = self._run_single_ablation_instance(
                    ablation_name=ablation_name,
                    run_training=True,
                    run_evaluation=True
                )

                self._update_incremental_results(result)

        # Generate final summary report
        self.generate_summary_report()


def run_frequency_trainable(config: Dict[str, Any]):
    """Ray Tune trainable function for running a single frequency sensitivity ablation.

    Args:
        config: Dictionary containing:
            - ablation_name: Name of ablation to run
            - base_config_path: Path to base configuration
            - train_task_ids: Task IDs to train on
            - ood_task_ids: OOD task IDs to evaluate on
            - output_root: Root directory for results
            - quick_test: Whether to run quick test
            - num_gpus: Number of GPUs to use (default: 1)
            - working_dir: Working directory to change to
    """
    ablation_name = config['ablation_name']
    base_config_path = config['base_config_path']
    train_task_ids = config['train_task_ids']
    ood_task_ids = config['ood_task_ids']
    output_root = config['output_root']
    quick_test = config['quick_test']
    num_gpus = config['num_gpus']
    working_dir = config['working_dir']

    # Change to the correct working directory
    original_dir = os.getcwd()
    os.chdir(working_dir)

    print(f"\n{'='*80}")
    print(f"Running frequency sensitivity ablation: {ablation_name}")
    print(f"Training tasks: {train_task_ids}")
    print(f"OOD tasks: {ood_task_ids}")
    print(f"Working directory: {os.getcwd()}")
    print(f"GPUs: {num_gpus}")
    print(f"{'='*80}\n")

    try:
        # Determine ablation type from the output_root
        # Extract the parent to figure out which configs to use
        study_path = Path(output_root)

        # Create ablation study instance
        # We need to determine the ablation_type based on the ablation_name
        if ablation_name.startswith('visual_freq_'):
            ablation_type = 'visual'
        elif ablation_name.startswith('proprio_freq_'):
            ablation_type = 'proprio'
        elif ablation_name.startswith('pure_visual_freq_'):
            ablation_type = 'pure_visual'
        elif ablation_name.startswith('pure_proprio_freq_'):
            ablation_type = 'pure_proprio'
        else:
            ablation_type = 'all'

        study = FrequencySensitivityStudy(
            base_config_path=base_config_path,
            train_task_ids=train_task_ids,
            ood_task_ids=ood_task_ids,
            output_root=output_root,
            quick_test=quick_test,
            num_gpus=num_gpus,
            ablation_type=ablation_type
        )

        # Run the ablation
        result = study._run_single_ablation_instance(
            ablation_name=ablation_name,
            run_training=True,
            run_evaluation=True
        )

        print(f"\n✓ Completed frequency sensitivity ablation: {ablation_name}")

    except Exception as e:
        print(f"\n✗ Failed: {ablation_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original directory
        os.chdir(original_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run frequency sensitivity ablation study for boundary detection',
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
        default='./frequency_sensitivity_results',
        help='Root directory for ablation results'
    )

    parser.add_argument(
        '--ablation-type',
        type=str,
        choices=['all', 'visual', 'proprio', 'pure_visual', 'pure_proprio', 'multimodal', 'pure'],
        default='all',
        help='Type of ablation to run: '
             'all (all ablations), '
             'visual (multi-modal visual freq), '
             'proprio (multi-modal proprio freq), '
             'pure_visual (pure visual freq), '
             'pure_proprio (pure proprio freq), '
             'multimodal (all multi-modal), '
             'pure (all pure single-modality)'
    )

    parser.add_argument(
        '--train-task-ids',
        type=int,
        nargs='+',
        help='Task IDs to train on (default: use from base config)'
    )

    parser.add_argument(
        '--ood-task-ids',
        type=int,
        nargs='+',
        help='OOD task IDs to evaluate on (default: auto-detect tasks >= 400)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (fewer epochs, less data)'
    )

    parser.add_argument(
        '--num-parallel',
        type=int,
        default=8,
        help='Number of parallel jobs to run (default: 8, for 8 GPUs)'
    )

    parser.add_argument(
        '--gpus-per-trial',
        type=int,
        default=1,
        help='Number of GPUs per trial (default: 1)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize Ray only if needed (for parallel execution)
    use_ray = args.num_parallel is not None and args.num_parallel > 1

    if use_ray and not ray.is_initialized():
        print(f"\nInitializing Ray for parallel execution (max {args.num_parallel} concurrent trials)...")
        ray.init()
        print(f"Started local Ray cluster")
        print(f"Ray cluster resources: {ray.cluster_resources()}\n")
    elif not use_ray:
        print(f"\nRunning in sequential mode (no Ray parallelization)\n")

    try:
        # Load base config to get default task IDs
        base_config = load_config(args.base_config)

        # Get training task IDs
        if args.train_task_ids:
            train_task_ids = args.train_task_ids
        else:
            # Use from config or default
            train_task_ids = [374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354,
                             360, 372, 359, 378, 390, 377, 384]

        print(f"Using training task IDs: {train_task_ids}")

        # Get OOD task IDs
        if args.ood_task_ids:
            ood_task_ids = args.ood_task_ids
        else:
            ood_task_ids = [380, 424, 392]

        print(f"Using OOD task IDs: {ood_task_ids}")

        # Initialize ablation study
        study = FrequencySensitivityStudy(
            base_config_path=args.base_config,
            train_task_ids=train_task_ids,
            ood_task_ids=ood_task_ids,
            output_root=args.output_root,
            quick_test=args.quick_test,
            num_gpus=args.gpus_per_trial,
            ablation_type=args.ablation_type
        )

        # Run ablations
        study.run_all_ablations(
            ablation_names=None,  # Run all ablations for the selected type
            num_parallel=args.num_parallel
        )

    finally:
        # Shutdown Ray if it was initialized
        if ray.is_initialized():
            print("\nShutting down Ray...")
            ray.shutdown()


if __name__ == '__main__':
    main()
