#!/usr/bin/env python3
"""Run multi-task ASFormer baseline experiments using Ray Tune.

This script trains and evaluates ASFormer models on MULTIPLE tasks simultaneously
(multi-task training), making results comparable to multi-task ablation studies.

Key differences from single-task ASFormer baseline:
- Trains ONE model on ALL training tasks (multi-task training)
- Evaluates on both in-distribution tasks (training tasks) and OOD tasks
- Reports separate metrics for in-dist vs OOD performance

ASFormer configuration (smaller, faster model):
- 2 decoder refinement stages
- 4 encoder layers, 2 decoder layers
- 64 feature maps, 4 attention heads
- BCE loss with pos_weight=10.0
- Learning rate: 1e-3 with cosine schedule
- 30 epochs with gradient clipping

Training set: Configurable list of tasks for multi-task training
OOD test set: Configurable list of OOD tasks (default: tasks >= 400)

All variants:
- Use DINO features from existing dataloader
- Train on multiple tasks simultaneously (one model for all tasks)
- Use same evaluation metrics as other models
- Drop-in replacement for multi-task ablation studies

Usage:
    # Run all ASFormer variants on multiple tasks
    python scripts/run_multi_task_asformer_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Specify training tasks explicitly
    python scripts/run_multi_task_asformer_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --train-task-ids 327 352 356

    # Quick test
    python scripts/run_multi_task_asformer_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --quick-test

    # Sequential execution (no Ray parallelization)
    python scripts/run_multi_task_asformer_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-parallel 1

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

    def __init__(self, baseline_study: 'MultiTaskASFormerBaselineStudy'):
        self.baseline_study = baseline_study
        self.processed_trials = set()

    def on_trial_complete(self, iteration: int, trials, trial, **info):
        """Called when a trial completes."""
        trial_id = trial.trial_id
        if trial_id in self.processed_trials:
            return

        variant_name = trial.config.get('variant_name')

        if not variant_name:
            return

        # Reconstruct experiment name
        exp_name = self.baseline_study.ASFORMER_VARIANTS[variant_name]['modifications']['experiment_name']

        # Load evaluation results from filesystem
        eval_dir = self.baseline_study.output_root / 'evaluations' / exp_name

        # Load both in-distribution and OOD results
        indist_results_file = eval_dir / 'indist_evaluation_results.json'
        ood_results_file = eval_dir / 'ood_evaluation_results.json'

        if indist_results_file.exists() and ood_results_file.exists():
            with open(indist_results_file, 'r') as f:
                indist_results = json.load(f)
            with open(ood_results_file, 'r') as f:
                ood_results = json.load(f)

            result = {
                'variant': variant_name,
                'variant_name': self.baseline_study.ASFORMER_VARIANTS[variant_name]['name'],
                'description': self.baseline_study.ASFORMER_VARIANTS[variant_name]['description'],
                'config_path': str(self.baseline_study.output_root / 'configs' / f'{exp_name}.yaml'),
                'checkpoint_dir': f'./checkpoints/{exp_name}',
                'training': {'status': 'completed'},
                'evaluation_indist': indist_results,
                'evaluation_ood': ood_results,
                'status': 'completed'
            }

            # Update incrementally
            self.baseline_study._update_incremental_results(result)
            self.processed_trials.add(trial_id)


class MultiTaskASFormerBaselineStudy:
    """Manages multi-task ASFormer baseline experiments."""

    ASFORMER_VARIANTS = {
        'asformer': {
            'name': 'ASFormer',
            'description': 'ASFormer with optimized hyperparameters (multi-task)',
            'modifications': {
                'model.architecture': 'asformer',
                'asformer.num_decoder_stages': 2,  # Reduced from 4
                'asformer.num_encoder_layers': 4,  # Reduced from 10
                'asformer.num_decoder_layers': 2,  # Reduced from 5
                'asformer.num_f_maps': 64,  # Reduced from 128
                'asformer.num_heads': 4,  # Reduced from 8
                'asformer.attention_type': 'normal',
                'model.use_proprio': False,
                'model.use_dino': True,
                'model.mlp_dropout': 0.3,
                'model.dino_encoder_dropout': 0.2,
                'data.batch_size': 256,  # Reduced from 4096 for better GPU utilization
                'training.gradient_accumulation_steps': 8,  # Effective batch size = 256 * 8 = 2048
                'training.loss_type': 'bce',
                'training.loss_pos_weight': 10.0,
                'training.optimizer': 'adam',
                'training.learning_rate': 1e-3,
                'training.weight_decay': 1e-4,
                'training.scheduler': 'cosine',
                'training.warmup_steps': 100,
                'training.gradient_clip_val': 1.0,
                'training.num_epochs': 30,  # Converges quickly
                'experiment_name': 'multi_task_asformer'
            }
        },
    }

    def __init__(
        self,
        base_config_path: str,
        train_task_ids: List[int],
        ood_task_ids: List[int] = None,
        output_root: str = './multi_task_asformer_results',
        quick_test: bool = False,
        num_gpus: int = 1
    ):
        """Initialize multi-task ASFormer baseline study.

        Args:
            base_config_path: Path to base configuration file
            train_task_ids: List of task IDs to train on
            ood_task_ids: List of OOD task IDs to evaluate on (default: tasks >= 400)
            output_root: Root directory for results
            quick_test: If True, use reduced epochs for testing
            num_gpus: Number of GPUs to use for each training run (default: 1)
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
        self.results = []

    def create_variant_config(
        self,
        variant_name: str
    ) -> Dict[str, Any]:
        """Create configuration for multi-task ASFormer variant.

        Args:
            variant_name: Name of variant

        Returns:
            Modified configuration dictionary
        """
        if variant_name not in self.ASFORMER_VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name}")

        # Load base config
        config_dict = yaml.safe_load(open(self.base_config_path))

        # Get variant modifications
        variant = self.ASFORMER_VARIANTS[variant_name]
        modifications = variant['modifications'].copy()

        # Multi-task training: use all training tasks
        exp_name = modifications['experiment_name']

        # Set checkpoint directory
        modifications['training.checkpoint_dir'] = f'./checkpoints/{exp_name}'

        # Disable wandb
        modifications['wandb.enabled'] = False

        # Update task IDs for multi-task training
        modifications['data.train_task_ids'] = self.train_task_ids
        modifications['data.eval_task_ids'] = self.train_task_ids  # Evaluate on training tasks during training

        # Quick test modifications
        if self.quick_test:
            modifications['training.num_epochs'] = 5  # Reduced for quick testing
            modifications['training.early_stopping'] = True
            modifications['training.patience'] = 10  # More patience
            modifications['data.max_samples'] = None  # Use all data

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

        # Check for best_model.pt
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
        """Run single-GPU training with automatic resuming support."""
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
            best_checkpoint = checkpoint_dir / 'best_model.pt'
            final_checkpoint = checkpoint_dir / 'final_model.pt'

            checkpoint_path = best_checkpoint if best_checkpoint.exists() else final_checkpoint

            return {
                'status': 'completed',
                'checkpoint': str(checkpoint_path)
            }

        # Construct python command
        cmd = [
            'python',
            'scripts/train.py',
            '--config', str(config_path)
        ]

        # Add resume flag if we can resume
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

    def _run_single_variant_instance(
        self,
        variant_name: str,
        run_training: bool,
        run_evaluation: bool
    ) -> Dict[str, Any]:
        """Run single multi-task ASFormer variant instance."""
        # Create config
        config = self.create_variant_config(variant_name)

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
                    'variant': variant_name,
                    'variant_name': self.ASFORMER_VARIANTS[variant_name]['name'],
                    'description': self.ASFORMER_VARIANTS[variant_name]['description'],
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
                checkpoints = list(checkpoint_dir.glob('best_*.pt'))
                if checkpoints:
                    checkpoint = checkpoints[0]

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
            'variant': variant_name,
            'variant_name': self.ASFORMER_VARIANTS[variant_name]['name'],
            'description': self.ASFORMER_VARIANTS[variant_name]['description'],
            'config_path': str(config_path),
            'checkpoint_dir': str(checkpoint_dir),
            'training': train_result,
            'evaluation_indist': indist_result,
            'evaluation_ood': ood_result,
            'status': 'completed'
        }

        self.results.append(result)
        return result

    def _update_incremental_results(self, new_result: Dict[str, Any]):
        """Update results incrementally."""
        self.results.append(new_result)

        # Save JSON
        results_file = self.output_root / 'multi_task_asformer_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Update CSVs
        self._update_summary_csvs()

        # Print progress
        print(f"\n{'='*80}")
        print(f"COMPLETED: {len(self.results)} experiments")
        print(f"{'='*80}")
        print(f"Latest: {new_result['variant_name']}")

        if 'evaluation_indist' in new_result and new_result['evaluation_indist']:
            print(f"\nIn-Distribution Performance:")
            self._print_aggregate_metrics(new_result['evaluation_indist'])

        if 'evaluation_ood' in new_result and new_result['evaluation_ood']:
            print(f"\nOut-of-Distribution Performance:")
            self._print_aggregate_metrics(new_result['evaluation_ood'])

        print(f"{'='*80}\n")

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

        if num_tasks > 0:
            print(f"  Tasks evaluated: {num_tasks}")
            for metric_name in ['f1@2.0', 'seg_f1@50', 'framewise_accuracy', 'edit_distance']:
                if metric_name in all_metrics:
                    avg_value = sum(all_metrics[metric_name]) / len(all_metrics[metric_name])
                    print(f"  avg_{metric_name}: {avg_value:.4f}")

    def _update_summary_csvs(self):
        """Update summary CSV files."""
        for eval_type in ['indist', 'ood']:
            summary_rows = []
            eval_key = f'evaluation_{eval_type}'

            for result in self.results:
                if eval_key in result and result[eval_key]:
                    for task_key, task_eval in result[eval_key].items():
                        if isinstance(task_eval, dict):
                            if 'status' in task_eval and task_eval['status'] == 'failed':
                                continue

                            row = {
                                'method': result['variant_name'],
                                'task': task_key,
                                'eval_type': eval_type,
                                'num_decoder_stages': self.ASFORMER_VARIANTS[result['variant']]['modifications'].get('asformer.num_decoder_stages', 'N/A'),
                                'num_encoder_layers': self.ASFORMER_VARIANTS[result['variant']]['modifications'].get('asformer.num_encoder_layers', 'N/A'),
                                'status': result.get('status', 'unknown')
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
                csv_file = self.output_root / f'multi_task_asformer_summary_{eval_type}.csv'
                df.to_csv(csv_file, index=False)

    def _load_completed_trials(self) -> set:
        """Load completed trials from previous runs."""
        completed_trials = set()
        results_file = self.output_root / 'multi_task_asformer_results.json'

        if not results_file.exists():
            return completed_trials

        try:
            with open(results_file, 'r') as f:
                previous_results = json.load(f)

            for result in previous_results:
                if result.get('status') == 'completed':
                    variant_name = result.get('variant')
                    if variant_name:
                        completed_trials.add(variant_name)

            self.results = previous_results

            if completed_trials:
                print(f"\n{'='*80}")
                print(f"FOUND {len(completed_trials)} COMPLETED TRIALS")
                print(f"{'='*80}\n")

        except Exception as e:
            print(f"Warning: Could not load previous results: {e}")

        return completed_trials

    def run_all_variants(
        self,
        variant_names: List[str] = None,
        num_parallel: int = None
    ):
        """Run all ASFormer variants."""
        if variant_names is None:
            variant_names = list(self.ASFORMER_VARIANTS.keys())

        print(f"\n{'='*80}")
        print(f"RUNNING MULTI-TASK ASFORMER BASELINE STUDY")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config_path}")
        print(f"Training tasks: {self.train_task_ids} ({len(self.train_task_ids)} tasks)")
        print(f"OOD tasks: {self.ood_task_ids} ({len(self.ood_task_ids)} tasks)")
        print(f"Variants: {variant_names}")
        print(f"Output: {self.output_root}")
        print(f"Quick test: {self.quick_test}")
        print(f"GPUs per variant: {self.num_gpus}")
        print(f"{'='*80}\n")

        # Load completed trials
        completed_trials = self._load_completed_trials()

        # Filter out completed trials
        trials_to_run = [name for name in variant_names if name not in completed_trials]

        if len(trials_to_run) < len(variant_names):
            skipped = set(variant_names) - set(trials_to_run)
            print(f"\n{'='*80}")
            print(f"SKIPPING {len(skipped)} ALREADY COMPLETED TRIALS")
            print(f"{'='*80}")
            for variant_name in sorted(skipped):
                variant_display = self.ASFORMER_VARIANTS.get(variant_name, {}).get('name', variant_name)
                print(f"  ⊘ {variant_display}")
            print(f"{'='*80}\n")

        print(f"Total trials to run: {len(trials_to_run)}")

        if len(trials_to_run) == 0:
            print("All trials completed!")
            self.generate_summary_report()
            return

        # Run trials
        if ray.is_initialized():
            working_dir = os.getcwd()
            trial_configs = []

            for variant_name in trials_to_run:
                trial_configs.append({
                    'variant_name': variant_name,
                    'base_config_path': self.base_config_path,
                    'train_task_ids': self.train_task_ids,
                    'ood_task_ids': self.ood_task_ids,
                    'output_root': str(self.output_root),
                    'quick_test': self.quick_test,
                    'num_gpus': self.num_gpus,
                    'working_dir': working_dir
                })

            reporter = CLIReporter(
                max_progress_rows=len(trial_configs) + 5,
                metric_columns=["status", "variant"]
            )

            callback = IncrementalResultsCallback(self)

            storage_path = str(Path(working_dir) / self.output_root)
            analysis = tune.run(
                run_variant_trainable,
                config=tune.grid_search(trial_configs),
                resources_per_trial={"cpu": self.num_gpus, "gpu": self.num_gpus},
                max_concurrent_trials=num_parallel,
                verbose=1,
                progress_reporter=reporter,
                storage_path=storage_path,
                name="multi_task_asformer_baseline",
                log_to_file=True,
                raise_on_failed_trial=False,
                callbacks=[callback]
            )

            # Finalize
            print(f"\n{'='*80}")
            print(f"FINALIZING RESULTS")
            print(f"{'='*80}\n")

            processed_trial_ids = callback.processed_trials

            for trial in analysis.trials:
                trial_id = trial.trial_id
                if trial_id in processed_trial_ids:
                    continue

                if trial.status == "TERMINATED":
                    variant_name = trial.config['variant_name']
                    exp_name = self.ASFORMER_VARIANTS[variant_name]['modifications']['experiment_name']

                    eval_dir = self.output_root / 'evaluations' / exp_name
                    indist_results_file = eval_dir / 'indist_evaluation_results.json'
                    ood_results_file = eval_dir / 'ood_evaluation_results.json'

                    if indist_results_file.exists() and ood_results_file.exists():
                        with open(indist_results_file, 'r') as f:
                            indist_results = json.load(f)
                        with open(ood_results_file, 'r') as f:
                            ood_results = json.load(f)

                        result = {
                            'variant': variant_name,
                            'variant_name': self.ASFORMER_VARIANTS[variant_name]['name'],
                            'description': self.ASFORMER_VARIANTS[variant_name]['description'],
                            'config_path': str(self.output_root / 'configs' / f'{exp_name}.yaml'),
                            'checkpoint_dir': f'./checkpoints/{exp_name}',
                            'training': {'status': 'completed'},
                            'evaluation_indist': indist_results,
                            'evaluation_ood': ood_results,
                            'status': 'completed'
                        }

                        self._update_incremental_results(result)
                        print(f"✓ Completed (late): {result['variant_name']}")
                    else:
                        print(f"✗ Missing results: {exp_name}")
                elif trial.status == "ERROR":
                    variant_name = trial.config.get('variant_name', 'unknown')
                    print(f"✗ Failed: {variant_name}")
        else:
            # Sequential execution
            for variant_name in trials_to_run:
                print(f"\n{'='*80}")
                print(f"Running variant: {variant_name}")
                print(f"{'='*80}\n")

                result = self._run_single_variant_instance(
                    variant_name=variant_name,
                    run_training=True,
                    run_evaluation=True
                )

                self._update_incremental_results(result)

        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate final summary report."""
        print(f"\n{'='*80}")
        print(f"MULTI-TASK ASFORMER BASELINE FINAL SUMMARY")
        print(f"{'='*80}\n")

        # Save JSON
        results_file = self.output_root / 'multi_task_asformer_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results: {results_file}")

        # Update CSVs
        self._update_summary_csvs()

        # Display summaries
        for eval_type in ['indist', 'ood']:
            csv_file = self.output_root / f'multi_task_asformer_summary_{eval_type}.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                print(f"\n{eval_type.upper()} Summary ({len(df)} experiments):")
                print(df.to_string(index=False))
                print(f"\nCSV: {csv_file}")

        print(f"\n{'='*80}")
        print(f"Multi-task ASFormer baseline study complete!")
        print(f"{'='*80}\n")


def run_variant_trainable(config: Dict[str, Any]):
    """Ray Tune trainable for single variant."""
    variant_name = config['variant_name']
    base_config_path = config['base_config_path']
    train_task_ids = config['train_task_ids']
    ood_task_ids = config['ood_task_ids']
    output_root = config['output_root']
    quick_test = config['quick_test']
    num_gpus = config['num_gpus']
    working_dir = config['working_dir']

    original_dir = os.getcwd()
    os.chdir(working_dir)

    try:
        study = MultiTaskASFormerBaselineStudy(
            base_config_path=base_config_path,
            train_task_ids=train_task_ids,
            ood_task_ids=ood_task_ids,
            output_root=output_root,
            quick_test=quick_test,
            num_gpus=num_gpus
        )

        result = study._run_single_variant_instance(
            variant_name=variant_name,
            run_training=True,
            run_evaluation=True
        )

        print(f"\n✓ Completed: {variant_name}")

    except Exception as e:
        print(f"\n✗ Failed: {variant_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        os.chdir(original_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run multi-task ASFormer baseline experiments',
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
        default='./multi_task_asformer_results',
        help='Root directory for results'
    )

    parser.add_argument(
        '--variants',
        type=str,
        nargs='+',
        choices=list(MultiTaskASFormerBaselineStudy.ASFORMER_VARIANTS.keys()) + ['all'],
        default=['asformer'],
        help='ASFormer variants to run'
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
        help='Quick test mode (fewer epochs)'
    )

    parser.add_argument(
        '--num-parallel',
        type=int,
        default=None,
        help='Number of parallel jobs'
    )

    parser.add_argument(
        '--gpus-per-trial',
        type=int,
        default=1,
        help='GPUs per trial'
    )

    parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize Ray
    use_ray = args.num_parallel is not None and args.num_parallel > 1

    if use_ray and not ray.is_initialized():
        print(f"\nInitializing Ray...")
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init()
        print(f"Ray resources: {ray.cluster_resources()}\n")
    elif not use_ray:
        print(f"\nRunning in sequential mode\n")

    try:
        # Load base config
        base_config = load_config(args.base_config)

        # Get training task IDs
        if args.train_task_ids:
            train_task_ids = args.train_task_ids
        else:
            train_task_ids = [374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354,
                             360, 372, 359, 378, 390, 377, 384]

        print(f"Using training task IDs: {train_task_ids}")

        # Get OOD task IDs
        if args.ood_task_ids:
            ood_task_ids = args.ood_task_ids
        else:
            ood_task_ids = [380, 424, 392]

        print(f"Using OOD task IDs: {ood_task_ids}")

        # Initialize study
        study = MultiTaskASFormerBaselineStudy(
            base_config_path=args.base_config,
            train_task_ids=train_task_ids,
            ood_task_ids=ood_task_ids,
            output_root=args.output_root,
            quick_test=args.quick_test,
            num_gpus=args.gpus_per_trial
        )

        # Get variants
        variant_names = args.variants
        if 'all' in variant_names:
            variant_names = list(MultiTaskASFormerBaselineStudy.ASFORMER_VARIANTS.keys())

        # Run
        study.run_all_variants(
            variant_names=variant_names,
            num_parallel=args.num_parallel
        )

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    main()
