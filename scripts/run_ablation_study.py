#!/usr/bin/env python3
"""Run comprehensive ablation study for boundary detection models.

This script runs ablation studies across different model architectures,
modalities, and loss functions for all task IDs. It trains and evaluates
each configuration on the same task data and reports F1 scores at different
granularities.

Ablation studies:
1. Pure proprioceptive (no visual features)
2. Pure visual (no proprioceptive features)
3. MLP + MSE loss
4. Pure DETR (no cross attention)
5. Cross attention + MSE loss
6. Single task training (per task)

Usage:
    # Run all ablations for all tasks
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Run specific ablations
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --ablations pure_proprio pure_visual mlp_mse

    # Run specific tasks only
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --task-ids 327 352 356

    # Quick test run (fewer epochs)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --quick-test --num-epochs 10

    # Run all ablations (parallel by default)
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml

    # Run specific ablations with limited parallelism
    python scripts/run_ablation_study.py --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-parallel 2 --ablations pure_proprio pure_visual

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config


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
            'name': 'Single Task',
            'description': 'Train on single task only (run per task)',
            'modifications': {
                # This will be modified per task dynamically
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

        # Handle single task case
        if ablation_name == 'single_task':
            if len(task_ids) != 1:
                raise ValueError("Single task ablation requires exactly one task ID")
            task_id = task_ids[0]
            modifications['experiment_name'] = f'ablation_single_task_{task_id}'

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

    def generate_summary_report(self):
        """Generate summary report of all results."""
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY SUMMARY")
        print(f"{'='*80}\n")

        # Save results to JSON
        results_file = self.output_root / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to: {results_file}")

        # Create summary table
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
                        # Extract F1 scores at different tolerances
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
            print(f"Saved summary CSV to: {csv_file}")

            # Print summary table
            print("\nResults Summary:")
            print(df.to_string(index=False))

        print(f"\n{'='*80}")
        print(f"Ablation study complete!")
        print(f"Results directory: {self.output_root}")
        print(f"{'='*80}\n")

    def run_all_ablations(
        self,
        task_ids: List[int],
        ablation_names: List[str] = None,
        num_parallel: int = None
    ):
        """Run all ablation studies in parallel using Ray.

        Args:
            task_ids: Task IDs to run on
            ablation_names: List of ablation names (None = all)
            num_parallel: Number of parallel jobs (None = all at once)
        """
        if ablation_names is None:
            ablation_names = list(self.ABLATION_CONFIGS.keys())

        print(f"\n{'='*80}")
        print(f"RUNNING ABLATION STUDY")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config_path}")
        print(f"Task IDs: {task_ids}")
        print(f"Ablations: {ablation_names}")
        print(f"Output root: {self.output_root}")
        print(f"Quick test: {self.quick_test}")

        # Get available GPUs
        try:
            cluster_resources = ray.cluster_resources()
            num_gpus = int(cluster_resources.get('GPU', 0))
            print(f"Available GPUs: {num_gpus}")
        except:
            num_gpus = 0

        print(f"Max parallel jobs: {num_parallel if num_parallel else num_gpus if num_gpus > 0 else 'unlimited'}")
        print(f"{'='*80}\n")

        # Prepare ablation jobs
        jobs = []
        for ablation_name in ablation_names:
            # For single_task ablation, create one job per task
            if ablation_name == 'single_task':
                for task_id in task_ids:
                    jobs.append((ablation_name, [task_id]))
            else:
                jobs.append((ablation_name, task_ids))

        print(f"Total jobs to run: {len(jobs)}")

        # Run jobs in parallel using Ray
        if num_parallel is None or num_parallel >= len(jobs):
            # Run all jobs in parallel
            print(f"Running all {len(jobs)} jobs in parallel...")
            futures = [
                run_ablation_remote.remote(
                    base_config_path=self.base_config_path,
                    output_root=str(self.output_root),
                    quick_test=self.quick_test,
                    ablation_name=ablation_name,
                    task_ids=job_task_ids
                )
                for ablation_name, job_task_ids in jobs
            ]
            results = ray.get(futures)
        else:
            # Run in batches
            print(f"Running jobs in batches of {num_parallel}...")
            results = []
            for i in range(0, len(jobs), num_parallel):
                batch = jobs[i:i + num_parallel]
                print(f"\nBatch {i//num_parallel + 1}/{(len(jobs) + num_parallel - 1)//num_parallel}: Running {len(batch)} jobs...")

                futures = [
                    run_ablation_remote.remote(
                        base_config_path=self.base_config_path,
                        output_root=str(self.output_root),
                        quick_test=self.quick_test,
                        ablation_name=ablation_name,
                        task_ids=job_task_ids
                    )
                    for ablation_name, job_task_ids in batch
                ]
                batch_results = ray.get(futures)
                results.extend(batch_results)

        # Collect results
        for result in results:
            if result is not None:
                self.results.append(result)
                ablation_name = result.get('ablation', 'unknown')
                print(f"\n✓ Completed: {ablation_name}")

        # Generate summary report
        self.generate_summary_report()


@ray.remote(num_gpus=0.5)
def run_ablation_remote(
    base_config_path: str,
    output_root: str,
    quick_test: bool,
    ablation_name: str,
    task_ids: List[int]
) -> Dict[str, Any]:
    """Ray remote function to run a single ablation study.

    Requires 1 GPU per task for training and evaluation.

    Args:
        base_config_path: Path to base configuration
        output_root: Root directory for results
        quick_test: Whether to run quick test
        ablation_name: Name of ablation to run
        task_ids: Task IDs to use

    Returns:
        Results dictionary
    """
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

        return result

    except Exception as e:
        print(f"\n✗ Failed: {ablation_name} with tasks {task_ids}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'ablation': ablation_name,
            'tasks': task_ids,
            'status': 'failed',
            'error': str(e)
        }


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

        # Run ablations in parallel
        study.run_all_ablations(
            task_ids=task_ids,
            ablation_names=ablation_names,
            num_parallel=args.num_parallel
        )

    finally:
        # Shutdown Ray
        if ray.is_initialized():
            print("\nShutting down Ray...")
            ray.shutdown()


if __name__ == '__main__':
    main()
