#!/usr/bin/env python3
"""Training script for boundary prediction models using YAML configuration.

This script provides a clean training pipeline with YAML-based configuration.
All parameters are specified in YAML config files, with optional CLI overrides.

Example usage:
    # Basic training
    python scripts/train.py --config configs/experiments/task_357.yaml

    # With WandB logging
    python scripts/train.py --config configs/experiments/task_357.yaml \\
        wandb.enabled=true

    # Override parameters
    python scripts/train.py --config configs/experiments/task_357.yaml \\
        training.learning_rate=0.001 \\
        data.batch_size=256

    # Debug mode
    python scripts/train.py --config configs/experiments/debug.yaml

Author: Claude Code
Date: October 25, 2025
"""

import argparse
import sys
import os
from pathlib import Path

# Disable tokenizers parallelism warning (before importing anything that uses tokenizers)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import load_config, validate_config, parse_cli_overrides
from models import create_model
from models.losses import get_loss_function
from training import (
    BoundaryDataset,
    TrainingDatasetConfig,
    collate_dict,
    BoundaryTrainer,
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    EmbeddingVisualizationCallback,
)
from data import load_parquet_split
from utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train boundary prediction model with YAML config',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (e.g., configs/experiments/task_357.yaml)'
    )

    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in key=value format (e.g., training.lr=0.001)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Parse CLI overrides
    overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    # Load configuration
    print("="*80)
    print("Loading configuration...")
    print("="*80)
    config = load_config(args.config, overrides=overrides)
    validate_config(config)

    print(f"Experiment: {config.experiment_name}")
    print(f"Config file: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")

    # Set random seed
    set_all_seeds(config.training.seed)
    print(f"Random seed: {config.training.seed}")

    # Setup device
    if config.training.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.training.device)
    print(f"Device: {device}")

    # Load data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)

    train_trajectories = load_parquet_split('train', splits_dir=config.data.splits_dir)
    val_trajectories = load_parquet_split('val', splits_dir=config.data.splits_dir)

    # Create datasets
    dataset_config = TrainingDatasetConfig(
        window_size=config.data.window_size,
        stride=config.data.stride,
        vocab_size=config.data.vocab_size,
        fps=config.data.fps,
        device=str(device),
        label_smoother_type=config.data.label_smoother_type,
        label_smoother_sigma=config.data.label_smoother_sigma,
        label_smoother_peak_value=config.data.label_smoother_peak_value,
        action_dim=config.data.action_dim,
        max_samples=config.data.max_samples,
        num_dino_frames=config.data.num_dino_frames  # Pass num_dino_frames from config
    )

    print("\nCreating training dataset...")
    train_dataset = BoundaryDataset(train_trajectories, dataset_config)

    print("\nCreating validation dataset...")
    val_dataset = BoundaryDataset(val_trajectories, dataset_config)

    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_dict,
        pin_memory=config.data.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_dict,
        pin_memory=config.data.pin_memory
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)

    model = create_model(config.model, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Create scheduler
    scheduler = None
    if config.training.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.min_lr
        )

    # Create loss function
    if config.training.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.training.loss_pos_weight)
        )
    else:
        # Build kwargs based on loss type
        loss_kwargs = {}

        # Add parameters based on what the loss function accepts
        if config.training.loss_type in ['weighted_bce', 'bce']:
            loss_kwargs['pos_weight'] = config.training.loss_pos_weight

        if config.training.loss_type in ['soft_focal', 'focal_bce', 'focal_dist_ce']:
            if hasattr(config.training, 'loss_alpha'):
                loss_kwargs['alpha'] = config.training.loss_alpha
            if hasattr(config.training, 'loss_gamma'):
                loss_kwargs['gamma'] = config.training.loss_gamma

        # Distribution-based losses don't need pos_weight, alpha, or gamma by default
        # They have their own default parameters

        criterion = get_loss_function(
            config.training.loss_type,
            **loss_kwargs
        )

    print(f"Optimizer: Adam (lr={config.training.learning_rate}, wd={config.training.weight_decay})")
    print(f"Scheduler: {config.training.scheduler}")
    print(f"Loss: {config.training.loss_type}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Create callbacks
    print("\n" + "="*80)
    print("Configuring callbacks...")
    print("="*80)

    callbacks = []

    # WandB callback
    if config.wandb.enabled:
        wandb_callback = WandBCallback(
            project=config.wandb.project,
            name=config.wandb.name or config.experiment_name,
            config=config.to_dict(),
            log_plots_every=config.training.log_plots_every,
            num_plots=config.training.num_plots,
            offline=(config.wandb.mode == 'offline')
        )
        callbacks.append(wandb_callback)
        print("✓ WandB logging enabled")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_dir=config.training.checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_every=config.training.save_every
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ Checkpointing enabled (save_dir={config.training.checkpoint_dir})")

    # Early stopping callback
    if config.training.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            monitor='val_loss',
            patience=config.training.patience,
            mode='min'
        )
        callbacks.append(early_stopping_callback)
        print(f"✓ Early stopping enabled (patience={config.training.patience})")

    # Scheduler callback
    if scheduler:
        scheduler_callback = LearningRateSchedulerCallback(scheduler)
        callbacks.append(scheduler_callback)

    # Embedding visualization callback (only if WandB is enabled)
    if config.wandb.enabled:
        embedding_vis_callback = EmbeddingVisualizationCallback(
            method='pca',  # Fast PCA visualization
            num_trajectories=3,  # Visualize 3 trajectories per split
            max_windows_per_traj=None,  # Include all windows
            layer='proprio_encoded',  # Visualize proprioceptive encoder output (after transformer)
            color_by='action',  # Color by action labels instead of timestep
            log_on_save=True,  # Log when best model is saved
            log_on_first=True,  # Log on first epoch
            log_on_last=True  # Log at end of training
        )
        callbacks.append(embedding_vis_callback)
        print("✓ Trajectory embedding visualization enabled (layer=proprio_encoded, color=action)")

    # Create trainer
    print("\n" + "="*80)
    print("Initializing trainer...")
    print("="*80)

    # Determine loss output type
    distribution_losses = ['dist_ce', 'kl_div', 'js_div', 'focal_dist_ce']
    raw_output_losses = ['mse_raw']  # Losses that use raw logits (no sigmoid/softmax)

    use_distribution_loss = config.training.loss_type in distribution_losses
    use_raw_output = config.training.loss_type in raw_output_losses

    trainer = BoundaryTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        scheduler=scheduler,
        prediction_stride=config.data.stride,
        prediction_window_size=config.model.prediction_window_size,
        classification_threshold=config.training.classification_threshold,
        use_distribution_loss=use_distribution_loss,
        use_raw_output=use_raw_output,
        freq_multiplier=train_dataset.freq_multiplier,  # Get from dataset
        peak_detection_distance=config.training.peak_detection_distance,
        iou_window_size=config.training.iou_window_size,
        metric_tolerances=config.training.metric_tolerances
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        log_every=config.training.log_every,
        collect_plots_every=config.training.log_plots_every
    )

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Print final statistics
    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    if val_loader:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
    print(f"Experiment: {config.experiment_name}")


if __name__ == '__main__':
    main()
