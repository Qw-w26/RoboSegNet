#!/usr/bin/env python3
"""Run comprehensive ablation study for boundary detection models using Ray Tune.

This script runs ablation studies across different model architectures,
modalities, and loss functions for all task IDs. Each baseline trains a
separate model per task (no multi-task training) and evaluates on that task.

Key improvements:
- Uses Ray Tune for proper GPU resource management
- Automatically sets CUDA_VISIBLE_DEVICES for each trial
- Prevents GPU contention by properly allocating GPU resources
- Supports flexible GPU allocation (e.g., 0.5 GPUs to run 2 trials per GPU)
- ALL baselines train on single tasks independently (one model per task)
- INCREMENTAL result saving: Results are saved and displayed after each trial completes

Ablation studies (all trained per task):
1. Pure proprioceptive (no visual features)
2. Pure visual (no proprioceptive features)
3. MLP + MSE loss
4. Pure DETR (no cross attention)
5. Cross attention + MSE loss
6. Single task baseline (default architecture)

Output metrics:
Boundary Detection Metrics:
- F1@1.0, F1@2.0, F1@3.0, etc. (Boundary F1 at different frame tolerances)
- AP@1.0, AP@2.0, AP@3.0, etc. (Average Precision at different tolerances)
- mAP (mean Average Precision)
- Precision, Recall, Accuracy

Action Segmentation Metrics (Standard):
- Seg-F1@10%, Seg-F1@25%, Seg-F1@50% (Segmental F1 at IoU thresholds)
- Edit-Dist (Segmental edit distance / Levenshtein distance)
- Acc (Framewise accuracy)

Usage:
    # Run all ablations for all tasks (1 GPU per trial)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Run specific ablations
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --ablations pure_proprio pure_visual mlp_mse

    # Run specific tasks only
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --task-ids 327 352

    # Quick test run (fewer epochs)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --quick-test

    # Limit parallelism (run 2 trials at a time)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-parallel 2

    # Use 0.5 GPUs per trial (run 2 trials per GPU)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --gpus-per-trial 0.5

    # Connect to existing Ray cluster
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --ray-address auto

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

    def __init__(self, ablation_study: 'AblationStudy'):
        """Initialize callback.

        Args:
            ablation_study: Reference to AblationStudy instance
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
        task_ids = trial.config.get('task_ids', [])

        if not ablation_name:
            return

        # Reconstruct experiment name
        base_exp_name = self.ablation_study.ABLATION_CONFIGS[ablation_name]['modifications']['experiment_name']
        exp_name = f'{base_exp_name}_{task_ids[0]}'

        # Load evaluation results from filesystem
        eval_dir = self.ablation_study.output_root / 'evaluations' / exp_name
        results_file = eval_dir / 'evaluation_results.json'

        if results_file.exists():
            with open(results_file, 'r') as f:
                eval_results = json.load(f)

            result = {
                'ablation': ablation_name,
                'ablation_name': self.ablation_study.ABLATION_CONFIGS[ablation_name]['name'],
                'description': self.ablation_study.ABLATION_CONFIGS[ablation_name]['description'],
                'tasks': task_ids,
                'config_path': str(self.ablation_study.output_root / 'configs' / f'{exp_name}.yaml'),
                'checkpoint_dir': f'./checkpoints/{exp_name}',
                'training': {'status': 'completed'},
                'evaluation': eval_results,
                'status': 'completed'
            }

            # Update incrementally
            self.ablation_study._update_incremental_results(result)
            self.processed_trials.add(trial_id)


class AblationStudy:
    """Manages ablation study experiments."""

    ABLATION_CONFIGS = {
        'pure_proprio': {
            'name': 'Pure Proprioceptive',
            'description': 'Only proprioceptive features, no visual input',
            'modifications': {
                'model.use_proprio': True,
                'model.use_dino': False,
                'experiment_name': 'ablation_pure_proprio'
            }
        },
        'pure_visual': {
            'name': 'Pure Visual',
            'description': 'Only visual features, no proprioceptive input',
            'modifications': {
                'model.use_proprio': False,
                'model.use_dino': True,
                'experiment_name': 'ablation_pure_visual'
            }
        },
        'mlp_mse': {
            'name': 'MLP + MSE',
            'description': 'Simple MLP architecture with MSE loss',
            'modifications': {
                'model.architecture': 'mlp',
                'training.loss_type': 'mse_raw',
                'training.optimizer': 'adam',
                'data.batch_size': 128,  # Can use larger batch with simpler model
                'experiment_name': 'ablation_mlp_mse'
            }
        },
        'detr_no_cross_attn': {
            'name': 'DETR No Cross Attention',
            'description': 'DETR with both modalities but simple addition instead of cross-attention',
            'modifications': {
                'model.architecture': 'boundary_detr',
                'model.use_proprio': True,  # Use both modalities
                'model.use_dino': True,
                'model.fusion_mode': 'add',  # Simple addition instead of cross-attention
                'experiment_name': 'ablation_detr_no_cross_attn'
            }
        },
        'cross_attn_mse': {
            'name': 'Cross Attention + MSE',
            'description': 'Cross-modal transformer with MSE loss (no Hungarian matching)',
            'modifications': {
                'model.architecture': 'cross_modal_transformer',
                'training.loss_type': 'mse_raw',
                'training.optimizer': 'adamw',
                'experiment_name': 'ablation_cross_attn_mse'
            }
        },
        'single_task': {
            'name': 'Single Task Baseline',
            'description': 'Baseline with default architecture trained per task',
            'modifications': {
                # Uses default settings from base config
                'experiment_name': 'ablation_single_task'
            }
        }
    }

    def __init__(
        self,
        base_config_path: str,
        output_root: str = './ablation_results',
        quick_test: bool = False
    ):
        """Initialize ablation study.

        Args:
            base_config_path: Path to base configuration file
            output_root: Root directory for ablation results
            quick_test: If True, use reduced epochs for quick testing
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        self.output_root = Path(output_root)
        self.quick_test = quick_test

        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = []

    def create_ablation_config(
        self,
        ablation_name: str,
        task_ids: List[int],
        eval_task_ids: List[int] = None
    ) -> Dict[str, Any]:
        """Create configuration for ablation study.

        Args:
            ablation_name: Name of ablation study
            task_ids: List of task IDs to train on
            eval_task_ids: List of task IDs to evaluate on (defaults to task_ids)

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

        # All ablations now train on single tasks, so add task_id to experiment name
        if len(task_ids) != 1:
            raise ValueError(f"Each ablation trial must train on exactly one task ID, got {task_ids}")

        task_id = task_ids[0]
        base_exp_name = modifications['experiment_name']
        modifications['experiment_name'] = f'{base_exp_name}_{task_id}'

        # Set unique checkpoint directory for this ablation
        exp_name = modifications['experiment_name']
        modifications['training.checkpoint_dir'] = f'./checkpoints/{exp_name}'

        # Disable wandb
        modifications['wandb.enabled'] = False

        # Update task IDs
        modifications['data.train_task_ids'] = task_ids
        modifications['data.eval_task_ids'] = eval_task_ids or task_ids

        # Quick test modifications
        if self.quick_test:
            modifications['training.num_epochs'] = 1  # Minimum for DETR to learn
            modifications['training.early_stopping'] = True
            modifications['training.patience'] = 5  # Stop early if not improving
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

    def run_training(
        self,
        config_path: Path,
        checkpoint_dir: Path
    ) -> Dict[str, Any]:
        """Run training with given configuration.

        Args:
            config_path: Path to configuration file
            checkpoint_dir: Directory for checkpoints

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Training: {config_path.stem}")
        print(f"Config: {config_path}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"{'='*80}\n")

        # Construct training command
        cmd = [
            'python', 'scripts/train.py',
            '--config', str(config_path)
        ]

        # Run training
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"WARNING: Training failed for {config_path.stem}")
            return {'status': 'failed'}

        # Find best checkpoint
        best_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('best_*.pt'))
            if checkpoints:
                best_checkpoint = checkpoints[0]

        return {
            'status': 'completed',
            'checkpoint': str(best_checkpoint) if best_checkpoint else None
        }

    def run_evaluation(
        self,
        checkpoint_path: Path,
        config_path: Path,
        task_ids: List[int],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run evaluation on checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            task_ids: Task IDs to evaluate on
            output_dir: Output directory for results

        Returns:
            Evaluation results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {checkpoint_path.stem}")
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
        task_ids: List[int],
        run_training: bool,
        run_evaluation: bool
    ) -> Dict[str, Any]:
        """Run single instance of ablation study."""
        # Create config
        config = self.create_ablation_config(ablation_name, task_ids)

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
            train_result = self.run_training(config_path, checkpoint_dir)

            if train_result['status'] != 'completed':
                return {
                    'ablation': ablation_name,
                    'tasks': task_ids,
                    'status': 'training_failed',
                    'config_path': str(config_path)
                }

        # Run evaluation
        eval_result = None
        if run_evaluation:
            checkpoint = train_result.get('checkpoint') if train_result else None

            # Find checkpoint if not provided
            if checkpoint is None and checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('best_*.pt'))
                if checkpoints:
                    checkpoint = checkpoints[0]

            if checkpoint:
                eval_result = self.run_evaluation(
                    checkpoint_path=Path(checkpoint),
                    config_path=config_path,
                    task_ids=task_ids,
                    output_dir=eval_dir
                )
            else:
                print("WARNING: No checkpoint found for evaluation")

        # Compile results
        result = {
            'ablation': ablation_name,
            'ablation_name': self.ABLATION_CONFIGS[ablation_name]['name'],
            'description': self.ABLATION_CONFIGS[ablation_name]['description'],
            'tasks': task_ids,
            'config_path': str(config_path),
            'checkpoint_dir': str(checkpoint_dir),
            'training': train_result,
            'evaluation': eval_result,
            'status': 'completed'
        }

        # Save result
        self.results.append(result)

        return result

    def _update_incremental_results(self, new_result: Dict[str, Any]):
        """Update results incrementally after each trial completion.

        Args:
            new_result: Result dictionary from completed trial
        """
        # Add to results list
        self.results.append(new_result)

        # Save detailed results to JSON immediately
        results_file = self.output_root / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Update summary CSV
        self._update_summary_csv()

        # Print progress
        print(f"\n{'='*80}")
        print(f"INCREMENTAL UPDATE: {len(self.results)} experiments completed")
        print(f"{'='*80}")
        print(f"Latest: {new_result['ablation_name']} on tasks {new_result['tasks']}")
        if 'evaluation' in new_result and new_result['evaluation']:
            for task_key, metrics in new_result['evaluation'].items():
                if isinstance(metrics, dict) and 'status' not in metrics:
                    print(f"\n{task_key} metrics:")
                    # Print key metrics
                    for metric_name in ['f1@2.0', 'seg_f1@50', 'framewise_accuracy', 'edit_distance']:
                        if metric_name in metrics:
                            print(f"  {metric_name}: {metrics[metric_name]:.4f}")
        print(f"{'='*80}\n")

    def _update_summary_csv(self):
        """Update the summary CSV file with current results."""
        summary_rows = []
        for result in self.results:
            if 'evaluation' in result and result['evaluation']:
                # Extract metrics for each task
                for task_key, task_eval in result['evaluation'].items():
                    if isinstance(task_eval, dict):
                        if 'status' in task_eval and task_eval['status'] == 'failed':
                            continue

                        row = {
                            'ablation': result['ablation_name'],
                            'task': task_key,
                            'status': result.get('status', 'unknown')
                        }

                        # Add metrics if available
                        # Extract boundary-level F1 scores at different tolerances
                        for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
                            f1_key = f'f1@{tol}'
                            if f1_key in task_eval:
                                row[f'F1@{tol}'] = task_eval[f1_key]

                        # Extract AP scores at different tolerances
                        for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
                            ap_key = f'ap@{tol}'
                            if ap_key in task_eval:
                                row[f'AP@{tol}'] = task_eval[ap_key]

                        # Add mAP if available
                        if 'map' in task_eval:
                            row['mAP'] = task_eval['map']

                        # Add segmental F1@IoU scores (standard action segmentation metrics)
                        for iou_pct in [10, 25, 50]:
                            seg_f1_key = f'seg_f1@{iou_pct}'
                            if seg_f1_key in task_eval:
                                row[f'Seg-F1@{iou_pct}%'] = task_eval[seg_f1_key]

                        # Add edit distance metrics
                        if 'edit_distance' in task_eval:
                            row['Edit-Dist'] = task_eval['edit_distance']
                        if 'normalized_edit_distance' in task_eval:
                            row['Norm-Edit-Dist'] = task_eval['normalized_edit_distance']

                        # Add framewise accuracy (Acc)
                        if 'framewise_accuracy' in task_eval:
                            row['Acc'] = task_eval['framewise_accuracy']

                        # Also add frame-level metrics
                        for metric_name in ['precision', 'recall', 'f1_score', 'accuracy']:
                            if metric_name in task_eval:
                                row[metric_name] = task_eval[metric_name]

                        summary_rows.append(row)

        if summary_rows:
            df = pd.DataFrame(summary_rows)

            # Save to CSV
            csv_file = self.output_root / 'ablation_summary.csv'
            df.to_csv(csv_file, index=False)

    def generate_summary_report(self):
        """Generate final summary report of all results."""
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY FINAL SUMMARY")
        print(f"{'='*80}\n")

        # Save results to JSON
        results_file = self.output_root / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to: {results_file}")

        # Update summary CSV one final time
        self._update_summary_csv()

        # Load and display final summary
        csv_file = self.output_root / 'ablation_summary.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"\nFinal Results Summary ({len(df)} experiments):")
            print(df.to_string(index=False))
            print(f"\nSummary CSV: {csv_file}")

        print(f"\n{'='*80}")
        print(f"Ablation study complete!")
        print(f"Results directory: {self.output_root}")
        print(f"{'='*80}\n")

    def run_all_ablations(
        self,
        task_ids: List[int],
        ablation_names: List[str] = None,
        num_parallel: int = None,
        gpus_per_trial: float = 1.0
    ):
        """Run all ablation studies in parallel using Ray Tune.

        Ray Tune properly manages GPU allocation and sets CUDA_VISIBLE_DEVICES
        for each trial, preventing GPU contention.

        Results are updated INCREMENTALLY as each trial completes:
        - ablation_results.json is updated after each trial
        - ablation_summary.csv is updated after each trial
        - Key metrics are printed to console after each trial
        This allows monitoring progress even if the run is interrupted.

        Args:
            task_ids: Task IDs to run on
            ablation_names: List of ablation names (None = all)
            num_parallel: Number of parallel jobs (None = unlimited)
            gpus_per_trial: Number of GPUs per trial (default: 1.0)
        """
        if ablation_names is None:
            ablation_names = list(self.ABLATION_CONFIGS.keys())

        print(f"\n{'='*80}")
        print(f"RUNNING ABLATION STUDY WITH RAY TUNE")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config_path}")
        print(f"Task IDs: {task_ids}")
        print(f"Ablations: {ablation_names}")
        print(f"Output root: {self.output_root}")
        print(f"Quick test: {self.quick_test}")
        print(f"GPUs per trial: {gpus_per_trial}")

        # Get available GPUs
        try:
            cluster_resources = ray.cluster_resources()
            num_gpus = int(cluster_resources.get('GPU', 0))
            print(f"Available GPUs: {num_gpus}")
        except:
            num_gpus = 0

        print(f"Max parallel jobs: {num_parallel if num_parallel else 'unlimited'}")
        print(f"{'='*80}\n")

        # Prepare trial configurations
        # Store current working directory for Ray workers
        working_dir = os.getcwd()

        trial_configs = []
        for ablation_name in ablation_names:
            # Create one trial per task for ALL ablations (not multi-task training)
            for task_id in task_ids:
                trial_configs.append({
                    'ablation_name': ablation_name,
                    'task_ids': [task_id],
                    'base_config_path': self.base_config_path,
                    'output_root': str(self.output_root),
                    'quick_test': self.quick_test,
                    'working_dir': working_dir
                })

        print(f"Total trials to run: {len(trial_configs)}")

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
            run_ablation_trainable,
            config=tune.grid_search(trial_configs),
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            max_concurrent_trials=num_parallel,
            verbose=1,
            progress_reporter=reporter,
            storage_path=storage_path,
            name="ablation_study",
            log_to_file=True,
            raise_on_failed_trial=False,  # Don't stop if one trial fails
            callbacks=[incremental_callback]  # Add incremental results callback
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
                # Trial completed but wasn't caught by callback
                ablation_name = trial.config['ablation_name']
                task_ids = trial.config['task_ids']

                # Reconstruct experiment name
                base_exp_name = self.ABLATION_CONFIGS[ablation_name]['modifications']['experiment_name']
                exp_name = f'{base_exp_name}_{task_ids[0]}'

                # Load evaluation results
                eval_dir = self.output_root / 'evaluations' / exp_name
                results_file = eval_dir / 'evaluation_results.json'

                if results_file.exists():
                    with open(results_file, 'r') as f:
                        eval_results = json.load(f)

                    result = {
                        'ablation': ablation_name,
                        'ablation_name': self.ABLATION_CONFIGS[ablation_name]['name'],
                        'description': self.ABLATION_CONFIGS[ablation_name]['description'],
                        'tasks': task_ids,
                        'config_path': str(self.output_root / 'configs' / f'{exp_name}.yaml'),
                        'checkpoint_dir': f'./checkpoints/{exp_name}',
                        'training': {'status': 'completed'},
                        'evaluation': eval_results,
                        'status': 'completed'
                    }

                    # Update incrementally
                    self._update_incremental_results(result)
                    print(f"✓ Completed (late): {result['ablation_name']} on tasks {task_ids}")
                else:
                    print(f"✗ Missing results: {exp_name}")
            elif trial.status == "ERROR":
                ablation_name = trial.config.get('ablation_name', 'unknown')
                print(f"✗ Failed: {ablation_name} - status: ERROR")

        # Generate final summary report
        self.generate_summary_report()


def run_ablation_trainable(config: Dict[str, Any]):
    """Ray Tune trainable function for running a single ablation study.

    Ray Tune automatically sets CUDA_VISIBLE_DEVICES based on allocated GPUs,
    so subprocesses will use the correct GPU without contention.

    Args:
        config: Dictionary containing:
            - ablation_name: Name of ablation to run
            - task_ids: Task IDs to use
            - base_config_path: Path to base configuration
            - output_root: Root directory for results
            - quick_test: Whether to run quick test
            - working_dir: Working directory to change to
    """
    ablation_name = config['ablation_name']
    task_ids = config['task_ids']
    base_config_path = config['base_config_path']
    output_root = config['output_root']
    quick_test = config['quick_test']
    working_dir = config['working_dir']

    # Change to the correct working directory (Ray workers may start in different dir)
    original_dir = os.getcwd()
    os.chdir(working_dir)

    print(f"\n{'='*80}")
    print(f"Running ablation: {ablation_name}")
    print(f"Task IDs: {task_ids}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Allocated GPUs (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"{'='*80}\n")

    try:
        # Create ablation study instance
        study = AblationStudy(
            base_config_path=base_config_path,
            output_root=output_root,
            quick_test=quick_test
        )

        # Run the ablation (without generating report, that's done in main process)
        result = study._run_single_ablation_instance(
            ablation_name=ablation_name,
            task_ids=task_ids,
            run_training=True,
            run_evaluation=True
        )

        # Ray Tune will automatically mark the trial as completed when function returns
        # Results are collected from filesystem by main process
        print(f"\n✓ Completed ablation: {ablation_name}")

    except Exception as e:
        print(f"\n✗ Failed: {ablation_name} with tasks {task_ids}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to mark trial as failed
        raise
    finally:
        # Restore original directory
        os.chdir(original_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run ablation study for boundary detection',
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
        default='./ablation_results',
        help='Root directory for ablation results'
    )

    parser.add_argument(
        '--ablations',
        type=str,
        nargs='+',
        choices=list(AblationStudy.ABLATION_CONFIGS.keys()) + ['all'],
        default=['all'],
        help='Ablations to run'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        help='Task IDs to run (default: use all from base config)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (fewer epochs, less data)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (only run evaluation on existing checkpoints)'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation (only run training)'
    )

    parser.add_argument(
        '--num-parallel',
        type=int,
        default=None,
        help='Number of parallel jobs to run (default: unlimited)'
    )

    parser.add_argument(
        '--gpus-per-trial',
        type=float,
        default=0.1,
        help='Number of GPUs per trial (default: 1.0, use 0.5 to run 2 trials per GPU)'
    )

    parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address (default: start local cluster)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        print(f"\nInitializing Ray...")
        if args.ray_address:
            ray.init(address=args.ray_address)
            print(f"Connected to Ray cluster at {args.ray_address}")
        else:
            ray.init()
            print(f"Started local Ray cluster")
        print(f"Ray cluster resources: {ray.cluster_resources()}\n")

    try:
        # Initialize ablation study
        study = AblationStudy(
            base_config_path=args.base_config,
            output_root=args.output_root,
            quick_test=args.quick_test
        )

        # Get task IDs
        task_ids = args.task_ids
        if task_ids is None:
            # Use task IDs from base config
            task_ids = study.base_config.data.train_task_ids

        print(f"Using task IDs: {task_ids}")

        # Get ablation names
        ablation_names = args.ablations
        if 'all' in ablation_names:
            ablation_names = list(AblationStudy.ABLATION_CONFIGS.keys())

        # Run ablations in parallel with Ray Tune
        study.run_all_ablations(
            task_ids=task_ids,
            ablation_names=ablation_names,
            num_parallel=args.num_parallel,
            gpus_per_trial=args.gpus_per_trial
        )

    finally:
        # Shutdown Ray
        if ray.is_initialized():
            print("\nShutting down Ray...")
            ray.shutdown()


if __name__ == '__main__':
    main()
