#!/usr/bin/env python3
"""Hyperparameter tuning script using Ray Tune for boundary prediction models.

This script enables parallel hyperparameter search across multiple configurations.
It's based on train.py but adapted for Ray Tune's distributed optimization.

Example usage:
    # Basic tuning with default search space
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml

    # With custom number of trials and GPUs
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
        --num-trials 20 --gpus-per-trial 0.5

    # With specific search algorithm
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
        --search-alg hyperopt --scheduler asha

Author: Claude Code
Date: January 2025
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

from config import load_config, validate_config, parse_cli_overrides
from models import create_model
from models.losses import get_loss_function
from training import (
    BoundaryDataset,
    TrainingDatasetConfig,
    collate_dict,
    BoundaryTrainer,
)
from data import load_multi_task_split
from utils.reproducibility import set_all_seeds


def train_boundary_model(config_dict: Dict[str, Any],
                        base_config_path: str):
    """Trainable function for Ray Tune.

    Args:
        config_dict: Hyperparameters from Ray Tune
        base_config_path: Path to base YAML config file
    """
    # Load base configuration
    base_config = load_config(base_config_path)

    # Convert relative paths to absolute paths (Ray Tune workers may have different CWD)
    if not Path(base_config.data.meta_dir).is_absolute():
        # Get the directory of the config file and resolve paths relative to it
        config_dir = Path(base_config_path).parent.parent.parent
        base_config.data.meta_dir = str((config_dir / base_config.data.meta_dir).resolve())

    if hasattr(base_config.data, 'cache_dir') and base_config.data.cache_dir:
        if not Path(base_config.data.cache_dir).is_absolute():
            config_dir = Path(base_config_path).parent.parent.parent
            base_config.data.cache_dir = str((config_dir / base_config.data.cache_dir).resolve())

    if not Path(base_config.training.checkpoint_dir).is_absolute():
        config_dir = Path(base_config_path).parent.parent.parent
        base_config.training.checkpoint_dir = str((config_dir / base_config.training.checkpoint_dir).resolve())

    # Override with hyperparameters from Ray Tune
    base_config.training.learning_rate = config_dict['learning_rate']
    base_config.training.classification_threshold = config_dict['classification_threshold']
    base_config.training.weight_decay = config_dict.get('weight_decay', base_config.training.weight_decay)
    base_config.data.batch_size = config_dict.get('batch_size', base_config.data.batch_size)

    # Update loss parameters if tuning them
    if 'loss_pos_weight' in config_dict:
        base_config.training.loss_pos_weight = config_dict['loss_pos_weight']
    if 'loss_alpha' in config_dict:
        base_config.training.loss_alpha = config_dict['loss_alpha']
    if 'loss_gamma' in config_dict:
        base_config.training.loss_gamma = config_dict['loss_gamma']

    # Update model parameters if tuning them
    if 'transformer_dropout' in config_dict:
        base_config.model.transformer_dropout = config_dict['transformer_dropout']
    if 'mlp_dropout' in config_dict:
        base_config.model.mlp_dropout = config_dict['mlp_dropout']
    if 'feature_noise_std' in config_dict:
        base_config.model.feature_noise_std = config_dict['feature_noise_std']

    # Disable WandB for individual trials (we'll log aggregate results)
    base_config.wandb.enabled = False

    # Set seed for reproducibility
    set_all_seeds(base_config.training.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data
    train_trajectories = load_multi_task_split(
        'train',
        task_ids=base_config.data.train_task_ids,
        meta_dir=base_config.data.meta_dir
    )

    val_trajectories = load_multi_task_split(
        'val',
        task_ids=base_config.data.eval_task_ids,
        meta_dir=base_config.data.meta_dir
    )

    # Create datasets
    dataset_config = TrainingDatasetConfig(
        window_size=base_config.data.window_size,
        stride=base_config.data.stride,
        vocab_size=base_config.data.vocab_size,
        fps=base_config.data.fps,
        device=str(device),
        label_smoother_type=base_config.data.label_smoother_type,
        label_smoother_sigma=base_config.data.label_smoother_sigma,
        label_smoother_peak_value=base_config.data.label_smoother_peak_value,
        action_dim=base_config.data.action_dim,
        max_samples=base_config.data.max_samples,
        num_dino_frames=base_config.data.num_dino_frames
    )

    train_dataset = BoundaryDataset(train_trajectories, dataset_config)
    val_dataset = BoundaryDataset(val_trajectories, dataset_config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config.data.batch_size,
        shuffle=True,
        num_workers=base_config.data.num_workers,
        collate_fn=collate_dict,
        pin_memory=base_config.data.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config.data.batch_size,
        shuffle=False,
        num_workers=base_config.data.num_workers,
        collate_fn=collate_dict,
        pin_memory=base_config.data.pin_memory
    )

    # Create model
    model = create_model(base_config.model, device=device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=base_config.training.learning_rate,
        weight_decay=base_config.training.weight_decay
    )

    # Create scheduler
    scheduler = None
    if base_config.training.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=base_config.training.num_epochs,
            eta_min=base_config.training.min_lr
        )

    # Create loss function
    if base_config.training.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(base_config.training.loss_pos_weight)
        )
    else:
        loss_kwargs = {}
        if base_config.training.loss_type in ['weighted_bce', 'bce']:
            loss_kwargs['pos_weight'] = base_config.training.loss_pos_weight
        if base_config.training.loss_type in ['soft_focal', 'focal_bce', 'focal_dist_ce']:
            if hasattr(base_config.training, 'loss_alpha'):
                loss_kwargs['alpha'] = base_config.training.loss_alpha
            if hasattr(base_config.training, 'loss_gamma'):
                loss_kwargs['gamma'] = base_config.training.loss_gamma

        criterion = get_loss_function(
            base_config.training.loss_type,
            **loss_kwargs
        )

    # Start from epoch 0 (no checkpoint loading)
    start_epoch = 0

    # Determine loss output type
    distribution_losses = ['dist_ce', 'kl_div', 'js_div', 'focal_dist_ce']
    raw_output_losses = ['mse_raw']

    use_distribution_loss = base_config.training.loss_type in distribution_losses
    use_raw_output = base_config.training.loss_type in raw_output_losses

    # Create trainer (without callbacks for Ray Tune)
    trainer = BoundaryTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=[],  # No callbacks for tuning
        scheduler=scheduler,
        prediction_stride=base_config.data.stride,
        prediction_window_size=base_config.model.prediction_window_size,
        classification_threshold=base_config.training.classification_threshold,
        use_distribution_loss=use_distribution_loss,
        use_raw_output=use_raw_output,
        freq_multiplier=train_dataset.freq_multiplier,
        peak_detection_distance=base_config.training.peak_detection_distance,
        iou_window_size=base_config.training.iou_window_size,
        metric_tolerances=base_config.training.metric_tolerances,
        train_task_ids=base_config.data.train_task_ids
    )

    # Training loop with Ray Tune reporting
    for epoch in range(start_epoch, base_config.training.num_epochs):
        # Set current epoch for trainer
        trainer.current_epoch = epoch

        # Train for one epoch
        train_metrics = trainer._train_epoch(train_loader, log_every=10, collect_trajectory_data=False)

        # Validate
        val_metrics = trainer._validate_epoch(val_loader, collect_trajectory_data=False)

        # Update learning rate scheduler if present
        if scheduler is not None:
            scheduler.step()

        # Prepare metrics for Ray Tune
        metrics = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1_score'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'epoch': epoch
        }

        # Add tolerance-specific metrics
        if 'metrics_by_tolerance' in val_metrics:
            for tol, tol_metrics in val_metrics['metrics_by_tolerance'].items():
                metrics[f'val_f1_tol_{tol}'] = tol_metrics.get('f1_score', 0.0)
                metrics[f'val_precision_tol_{tol}'] = tol_metrics.get('precision', 0.0)
                metrics[f'val_recall_tol_{tol}'] = tol_metrics.get('recall', 0.0)

        # Report metrics to Ray Tune (without checkpoints)
        tune.report(metrics)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning with Ray Tune',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base YAML config file'
    )

    parser.add_argument(
        '--num-trials',
        type=int,
        default=10,
        help='Number of trials to run'
    )

    parser.add_argument(
        '--gpus-per-trial',
        type=float,
        default=0.5,
        help='Number of GPUs per trial (can be fractional for sharing)'
    )

    parser.add_argument(
        '--cpus-per-trial',
        type=int,
        default=4,
        help='Number of CPUs per trial'
    )

    parser.add_argument(
        '--max-concurrent-trials',
        type=int,
        default=16,
        help='Maximum number of concurrent trials'
    )

    parser.add_argument(
        '--search-alg',
        type=str,
        choices=['random', 'hyperopt', 'optuna'],
        default='hyperopt',
        help='Search algorithm to use'
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['asha', 'pbt', 'none'],
        default='asha',
        help='Trial scheduler to use'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='val_f1',
        help='Metric to optimize'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['min', 'max'],
        default='max',
        help='Optimization mode (min or max)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./ray_results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name for the experiment'
    )

    return parser.parse_args()


def get_search_space(base_config_path: str) -> Dict[str, Any]:
    """Define the hyperparameter search space.

    Args:
        base_config_path: Path to base config to extract reasonable ranges

    Returns:
        Dictionary defining the search space
    """
    # Load base config to get reasonable ranges
    base_config = load_config(base_config_path)

    search_space = {
        # Learning rate: log-uniform search
        'learning_rate': tune.loguniform(1e-5, 1e-3),

        # Classification threshold: uniform search around base value
        'classification_threshold': tune.uniform(0.3, 0.6),

        # Weight decay: log-uniform search
        'weight_decay': tune.loguniform(1e-6, 1e-3),

        # Batch size: choice of powers of 2
        'batch_size': tune.choice([512, 1024, 2048, 4096]),

        # Dropout rates
        'transformer_dropout': tune.uniform(0.1, 0.5),
        'mlp_dropout': tune.uniform(0.1, 0.5),

        # Feature noise for regularization
        'feature_noise_std': tune.uniform(0.0, 0.1),
    }

    # Add loss-specific parameters based on loss type
    if hasattr(base_config.training, 'loss_pos_weight'):
        search_space['loss_pos_weight'] = tune.uniform(5.0, 20.0)

    if hasattr(base_config.training, 'loss_alpha'):
        search_space['loss_alpha'] = tune.uniform(0.1, 0.5)

    if hasattr(base_config.training, 'loss_gamma'):
        search_space['loss_gamma'] = tune.uniform(1.0, 3.0)

    return search_space


def main():
    """Main hyperparameter tuning function."""
    args = parse_args()

    print("="*80)
    print("Ray Tune Hyperparameter Search")
    print("="*80)
    print(f"Base config: {args.config}")
    print(f"Number of trials: {args.num_trials}")
    print(f"GPUs per trial: {args.gpus_per_trial}")
    print(f"Search algorithm: {args.search_alg}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Metric to optimize: {args.metric} ({args.mode})")
    print(f"Results directory: {args.results_dir}")
    print("="*80)

    # Get search space
    search_space = get_search_space(args.config)

    print("\nSearch space:")
    for key, value in search_space.items():
        print(f"  {key}: {value}")

    # Setup search algorithm
    search_alg = None
    if args.search_alg == 'hyperopt':
        search_alg = HyperOptSearch(metric=args.metric, mode=args.mode)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent_trials)
    elif args.search_alg == 'optuna':
        search_alg = OptunaSearch(metric=args.metric, mode=args.mode)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent_trials)

    # Setup scheduler
    scheduler = None
    if args.scheduler == 'asha':
        scheduler = ASHAScheduler(
            metric=args.metric,
            mode=args.mode,
            max_t=100,  # Maximum epochs
            grace_period=10,  # Minimum epochs before stopping
            reduction_factor=3
        )
    elif args.scheduler == 'pbt':
        scheduler = PopulationBasedTraining(
            time_attr='epoch',
            metric=args.metric,
            mode=args.mode,
            perturbation_interval=5,
            hyperparam_mutations={
                'learning_rate': tune.loguniform(1e-5, 1e-3),
                'weight_decay': tune.loguniform(1e-6, 1e-3),
            }
        )

    # Setup reporter
    reporter = CLIReporter(
        metric_columns=[args.metric, "val_loss", "train_loss", "epoch"],
        max_report_frequency=30
    )

    # Determine experiment name
    experiment_name = args.name
    if experiment_name is None:
        base_config = load_config(args.config)
        experiment_name = f"tune_{base_config.experiment_name}"

    # Convert paths to absolute paths (required by Ray Tune workers)
    results_dir = str(Path(args.results_dir).resolve())
    config_path = str(Path(args.config).resolve())

    # Run tuning
    print("\n" + "="*80)
    print("Starting hyperparameter search...")
    print("="*80 + "\n")

    analysis = tune.run(
        tune.with_parameters(
            train_boundary_model,
            base_config_path=config_path
        ),
        config=search_space,
        num_samples=args.num_trials,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        storage_path=results_dir,
        name=experiment_name,
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial
        },
        verbose=1,
        raise_on_failed_trial=False
    )

    # Print results
    print("\n" + "="*80)
    print("Hyperparameter search complete!")
    print("="*80)

    best_trial = analysis.get_best_trial(args.metric, args.mode, "last")

    if best_trial is None:
        print("\n⚠️  No trials completed successfully!")
        print(f"\nResults directory: {results_dir}/{experiment_name}")
        print("\nCheck the error logs in the results directory for more details.")

        # List all trials and their status
        print("\nTrial summary:")
        for trial in analysis.trials:
            print(f"  {trial.trial_id}: {trial.status}")
        return

    print(f"\nBest trial: {best_trial.trial_id}")
    print(f"Best {args.metric}: {best_trial.last_result[args.metric]:.4f}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.config.items():
        print(f"  {key}: {value}")

    print(f"\nAll results saved to: {results_dir}/{experiment_name}")

    # Save best config
    best_config_path = Path(results_dir) / experiment_name / "best_config.txt"
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_config_path, 'w') as f:
        f.write(f"Best {args.metric}: {best_trial.last_result[args.metric]:.4f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in best_trial.config.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nBest config saved to: {best_config_path}")


if __name__ == '__main__':
    main()
