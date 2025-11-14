#!/usr/bin/env python3
"""Run MS-TCN baseline experiments using the same infrastructure as ablation study.

This script trains and evaluates MS-TCN models in the same setup as other baselines,
making results directly comparable. Uses Ray Tune for parallel execution.

MS-TCN variants tested:
1. ms_tcn_2stage_5layer: Standard MS-TCN (2 stages, 5 layers) - recommended for sliding windows
2. ms_tcn_3stage_7layer: Deeper MS-TCN (3 stages, 7 layers) - more refinement
3. ms_tcn_4stage_10layer: Full MS-TCN (4 stages, 10 layers) - original paper setup

All variants:
- Use DINO features from existing dataloader
- Train on single tasks (one model per task)
- Use same evaluation metrics as other models
- Drop-in replacement for MLP/Transformer/DETR

Usage:
    # Run all MS-TCN variants on all tasks
    python scripts/run_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Run specific variant on specific tasks
    python scripts/run_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --variants ms_tcn_2stage_5layer --task-ids 357 358

    # Quick test
    python scripts/run_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --quick-test --task-ids 357

    # Run 2 trials per GPU
    python scripts/run_ms_tcn_baseline.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --gpus-per-trial 0.5

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

    def __init__(self, baseline_study: 'MSTCNBaselineStudy'):
        self.baseline_study = baseline_study
        self.processed_trials = set()

    def on_trial_complete(self, iteration: int, trials, trial, **info):
        """Called when a trial completes."""
        trial_id = trial.trial_id
        if trial_id in self.processed_trials:
            return

        variant_name = trial.config.get('variant_name')
        task_ids = trial.config.get('task_ids', [])

        if not variant_name:
            return

        # Reconstruct experiment name
        base_exp_name = self.baseline_study.MS_TCN_VARIANTS[variant_name]['modifications']['experiment_name']
        exp_name = f'{base_exp_name}_{task_ids[0]}'

        # Load evaluation results
        eval_dir = self.baseline_study.output_root / 'evaluations' / exp_name
        results_file = eval_dir / 'evaluation_results.json'

        if results_file.exists():
            with open(results_file, 'r') as f:
                eval_results = json.load(f)

            result = {
                'variant': variant_name,
                'variant_name': self.baseline_study.MS_TCN_VARIANTS[variant_name]['name'],
                'description': self.baseline_study.MS_TCN_VARIANTS[variant_name]['description'],
                'tasks': task_ids,
                'config_path': str(self.baseline_study.output_root / 'configs' / f'{exp_name}.yaml'),
                'checkpoint_dir': f'./checkpoints/{exp_name}',
                'training': {'status': 'completed'},
                'evaluation': eval_results,
                'status': 'completed'
            }

            self.baseline_study._update_incremental_results(result)
            self.processed_trials.add(trial_id)


class MSTCNBaselineStudy:
    """Manages MS-TCN baseline experiments."""

    MS_TCN_VARIANTS = {
        'ms_tcn_2stage_5layer': {
            'name': 'MS-TCN (2 stages, 5 layers)',
            'description': 'Standard MS-TCN for sliding windows - recommended baseline',
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
                'experiment_name': 'mstcn_2stage_5layer'
            }
        },
        'ms_tcn_3stage_7layer': {
            'name': 'MS-TCN (3 stages, 7 layers)',
            'description': 'Deeper MS-TCN with more refinement stages',
            'modifications': {
                'model.architecture': 'ms_tcn',
                'ms_tcn.num_stages': 3,
                'ms_tcn.num_layers': 7,
                'ms_tcn.num_f_maps': 64,
                'model.use_proprio': False,  # MS-TCN uses only visual features
                'model.use_dino': True,
                'training.loss_type': 'bce',  # Use BCE with sigmoid for proper convergence
                'training.loss_pos_weight': 10.0,
                'training.optimizer': 'adam',
                'experiment_name': 'mstcn_3stage_7layer'
            }
        },
        'ms_tcn_4stage_10layer': {
            'name': 'MS-TCN (4 stages, 10 layers)',
            'description': 'Full MS-TCN as in original paper - very deep',
            'modifications': {
                'model.architecture': 'ms_tcn',
                'ms_tcn.num_stages': 4,
                'ms_tcn.num_layers': 10,
                'ms_tcn.num_f_maps': 64,
                'model.use_proprio': False,  # MS-TCN uses only visual features
                'model.use_dino': True,
                'training.loss_type': 'bce',  # Use BCE with sigmoid for proper convergence
                'training.loss_pos_weight': 10.0,
                'training.optimizer': 'adam',
                'experiment_name': 'mstcn_4stage_10layer'
            }
        },
        'ms_tcn_visual_only': {
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
                'experiment_name': 'mstcn_visual_only'
            }
        },
        'ms_tcn_large': {
            'name': 'MS-TCN Large (128 feature maps)',
            'description': 'MS-TCN with larger capacity',
            'modifications': {
                'model.architecture': 'ms_tcn',
                'ms_tcn.num_stages': 2,
                'ms_tcn.num_layers': 5,
                'ms_tcn.num_f_maps': 128,
                'model.use_proprio': False,  # MS-TCN uses only visual features
                'model.use_dino': True,
                'training.loss_type': 'bce',  # Use BCE with sigmoid for proper convergence
                'training.loss_pos_weight': 10.0,
                'training.optimizer': 'adam',
                'experiment_name': 'mstcn_large'
            }
        }
    }

    def __init__(
        self,
        base_config_path: str,
        output_root: str = './ms_tcn_results',
        quick_test: bool = False
    ):
        """Initialize MS-TCN baseline study.

        Args:
            base_config_path: Path to base configuration file
            output_root: Root directory for results
            quick_test: If True, use reduced epochs for testing
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        self.output_root = Path(output_root)
        self.quick_test = quick_test

        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = []

    def create_variant_config(
        self,
        variant_name: str,
        task_ids: List[int],
        eval_task_ids: List[int] = None
    ) -> Dict[str, Any]:
        """Create configuration for MS-TCN variant.

        Args:
            variant_name: Name of variant
            task_ids: List of task IDs to train on
            eval_task_ids: List of task IDs to evaluate on

        Returns:
            Modified configuration dictionary
        """
        if variant_name not in self.MS_TCN_VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name}")

        # Load base config
        config_dict = yaml.safe_load(open(self.base_config_path))

        # Get variant modifications
        variant = self.MS_TCN_VARIANTS[variant_name]
        modifications = variant['modifications'].copy()

        # Single task training
        if len(task_ids) != 1:
            raise ValueError(f"Each trial must train on exactly one task, got {task_ids}")

        task_id = task_ids[0]
        base_exp_name = modifications['experiment_name']
        modifications['experiment_name'] = f'{base_exp_name}_{task_id}'

        # Set checkpoint directory
        exp_name = modifications['experiment_name']
        modifications['training.checkpoint_dir'] = f'./checkpoints/{exp_name}'

        # Disable wandb
        modifications['wandb.enabled'] = False

        # Update task IDs
        modifications['data.train_task_ids'] = task_ids
        modifications['data.eval_task_ids'] = eval_task_ids or task_ids

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

    def run_training(self, config_path: Path, checkpoint_dir: Path) -> Dict[str, Any]:
        """Run training with given configuration."""
        print(f"\n{'='*80}")
        print(f"Training: {config_path.stem}")
        print(f"{'='*80}\n")

        # Check for existing checkpoints
        resume_checkpoint = None
        if checkpoint_dir.exists():
            best_checkpoints = list(checkpoint_dir.glob('best_*.pt'))
            if best_checkpoints:
                resume_checkpoint = best_checkpoints[0]
                print(f"Found checkpoint: {resume_checkpoint}\n")

        # Training command
        cmd = ['python', 'scripts/train.py', '--config', str(config_path)]
        if resume_checkpoint:
            cmd.extend(['--resume', str(resume_checkpoint)])

        # Run training
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
        """Run evaluation on checkpoint."""
        print(f"\n{'='*80}")
        print(f"Evaluating: {checkpoint_path.stem}")
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
        task_ids: List[int],
        run_training: bool,
        run_evaluation: bool
    ) -> Dict[str, Any]:
        """Run single MS-TCN variant instance."""
        # Create config
        config = self.create_variant_config(variant_name, task_ids)

        # Setup paths
        exp_name = config['experiment_name']
        config_path = self.output_root / 'configs' / f'{exp_name}.yaml'
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        eval_dir = self.output_root / 'evaluations' / exp_name

        # Save config
        self.save_config(config, config_path)

        # Run training
        train_result = None
        if run_training:
            train_result = self.run_training(config_path, checkpoint_dir)
            if train_result['status'] != 'completed':
                return {
                    'variant': variant_name,
                    'tasks': task_ids,
                    'status': 'training_failed',
                    'config_path': str(config_path)
                }

        # Run evaluation
        eval_result = None
        if run_evaluation:
            checkpoint = train_result.get('checkpoint') if train_result else None

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

        # Compile results
        result = {
            'variant': variant_name,
            'variant_name': self.MS_TCN_VARIANTS[variant_name]['name'],
            'description': self.MS_TCN_VARIANTS[variant_name]['description'],
            'tasks': task_ids,
            'config_path': str(config_path),
            'checkpoint_dir': str(checkpoint_dir),
            'training': train_result,
            'evaluation': eval_result,
            'status': 'completed'
        }

        self.results.append(result)
        return result

    def _update_incremental_results(self, new_result: Dict[str, Any]):
        """Update results incrementally."""
        self.results.append(new_result)

        # Save JSON
        results_file = self.output_root / 'ms_tcn_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Update CSV
        self._update_summary_csv()

        # Print progress
        print(f"\n{'='*80}")
        print(f"COMPLETED: {len(self.results)} experiments")
        print(f"{'='*80}")
        print(f"Latest: {new_result['variant_name']} on tasks {new_result['tasks']}")
        if 'evaluation' in new_result and new_result['evaluation']:
            for task_key, metrics in new_result['evaluation'].items():
                if isinstance(metrics, dict) and 'status' not in metrics:
                    print(f"\n{task_key}:")
                    for metric in ['f1@2.0', 'seg_f1@50', 'framewise_accuracy']:
                        if metric in metrics:
                            print(f"  {metric}: {metrics[metric]:.4f}")
        print(f"{'='*80}\n")

    def _update_summary_csv(self):
        """Update summary CSV."""
        summary_rows = []
        for result in self.results:
            if 'evaluation' in result and result['evaluation']:
                for task_key, task_eval in result['evaluation'].items():
                    if isinstance(task_eval, dict) and 'status' not in task_eval:
                        row = {
                            'method': result['variant_name'],
                            'task': task_key,
                            'num_stages': self.MS_TCN_VARIANTS[result['variant']]['modifications'].get('model.ms_tcn.num_stages', 'N/A'),
                            'num_layers': self.MS_TCN_VARIANTS[result['variant']]['modifications'].get('model.ms_tcn.num_layers', 'N/A'),
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
            csv_file = self.output_root / 'ms_tcn_summary.csv'
            df.to_csv(csv_file, index=False)

    def _load_completed_trials(self) -> set:
        """Load completed trials from previous runs."""
        completed_trials = set()
        results_file = self.output_root / 'ms_tcn_results.json'

        if not results_file.exists():
            return completed_trials

        try:
            with open(results_file, 'r') as f:
                previous_results = json.load(f)

            for result in previous_results:
                if result.get('status') == 'completed':
                    variant_name = result.get('variant')
                    tasks = result.get('tasks', [])
                    if variant_name and tasks:
                        for task_id in tasks:
                            completed_trials.add((variant_name, task_id))

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
        task_ids: List[int],
        variant_names: List[str] = None,
        num_parallel: int = None,
        gpus_per_trial: float = 1.0
    ):
        """Run all MS-TCN variants in parallel."""
        if variant_names is None:
            variant_names = list(self.MS_TCN_VARIANTS.keys())

        print(f"\n{'='*80}")
        print(f"RUNNING MS-TCN BASELINE STUDY")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config_path}")
        print(f"Task IDs: {task_ids}")
        print(f"Variants: {variant_names}")
        print(f"Output: {self.output_root}")
        print(f"Quick test: {self.quick_test}")
        print(f"GPUs per trial: {gpus_per_trial}")
        print(f"{'='*80}\n")

        # Load completed trials
        completed_trials = self._load_completed_trials()

        # Prepare trial configs
        working_dir = os.getcwd()
        trial_configs = []
        skipped_trials = []

        for variant_name in variant_names:
            for task_id in task_ids:
                if (variant_name, task_id) in completed_trials:
                    skipped_trials.append((variant_name, task_id))
                    continue

                trial_configs.append({
                    'variant_name': variant_name,
                    'task_ids': [task_id],
                    'base_config_path': self.base_config_path,
                    'output_root': str(self.output_root),
                    'quick_test': self.quick_test,
                    'working_dir': working_dir
                })

        if skipped_trials:
            print(f"Skipping {len(skipped_trials)} completed trials\n")

        print(f"Total trials to run: {len(trial_configs)}\n")

        if len(trial_configs) == 0:
            print("All trials completed!")
            self.generate_summary_report()
            return

        # Configure reporter
        reporter = CLIReporter(
            max_progress_rows=len(trial_configs) + 5,
            metric_columns=["status", "variant"]
        )

        # Create callback
        callback = IncrementalResultsCallback(self)

        # Run with Ray Tune
        storage_path = str(Path(working_dir) / self.output_root)
        analysis = tune.run(
            run_variant_trainable,
            config=tune.grid_search(trial_configs),
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            max_concurrent_trials=num_parallel,
            verbose=1,
            progress_reporter=reporter,
            storage_path=storage_path,
            name="ms_tcn_baseline",
            log_to_file=True,
            raise_on_failed_trial=False,
            callbacks=[callback]
        )

        # Generate final report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate final summary report."""
        print(f"\n{'='*80}")
        print(f"MS-TCN BASELINE FINAL SUMMARY")
        print(f"{'='*80}\n")

        # Save JSON
        results_file = self.output_root / 'ms_tcn_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results: {results_file}")

        # Update CSV
        self._update_summary_csv()

        # Display
        csv_file = self.output_root / 'ms_tcn_summary.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"\nSummary ({len(df)} experiments):")
            print(df.to_string(index=False))
            print(f"\nCSV: {csv_file}")

        print(f"\n{'='*80}")
        print(f"MS-TCN baseline study complete!")
        print(f"{'='*80}\n")


def run_variant_trainable(config: Dict[str, Any]):
    """Ray Tune trainable for single variant."""
    variant_name = config['variant_name']
    task_ids = config['task_ids']
    base_config_path = config['base_config_path']
    output_root = config['output_root']
    quick_test = config['quick_test']
    working_dir = config['working_dir']

    original_dir = os.getcwd()
    os.chdir(working_dir)

    try:
        study = MSTCNBaselineStudy(
            base_config_path=base_config_path,
            output_root=output_root,
            quick_test=quick_test
        )

        result = study._run_single_variant_instance(
            variant_name=variant_name,
            task_ids=task_ids,
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
        description='Run MS-TCN baseline experiments',
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
        default='./ms_tcn_results',
        help='Root directory for results'
    )

    parser.add_argument(
        '--variants',
        type=str,
        nargs='+',
        choices=list(MSTCNBaselineStudy.MS_TCN_VARIANTS.keys()) + ['all'],
        default=['all'],
        help='MS-TCN variants to run'
    )

    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        help='Task IDs to run (default: use from base config)'
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
        type=float,
        default=1.0,
        help='GPUs per trial (use 0.5 for 2 trials per GPU)'
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
    if not ray.is_initialized():
        print(f"\nInitializing Ray...")
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init()
        print(f"Ray resources: {ray.cluster_resources()}\n")

    try:
        # Initialize study
        study = MSTCNBaselineStudy(
            base_config_path=args.base_config,
            output_root=args.output_root,
            quick_test=args.quick_test
        )

        # Get task IDs
        task_ids = args.task_ids
        if task_ids is None:
            task_ids = study.base_config.data.train_task_ids

        # Get variants
        variant_names = args.variants
        if 'all' in variant_names:
            variant_names = list(MSTCNBaselineStudy.MS_TCN_VARIANTS.keys())

        # Run
        study.run_all_variants(
            task_ids=task_ids,
            variant_names=variant_names,
            num_parallel=args.num_parallel,
            gpus_per_trial=args.gpus_per_trial
        )

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    main()
