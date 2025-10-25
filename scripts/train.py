#!/usr/bin/env python3
"""Training script for boundary prediction models.

This script provides a complete training pipeline with:
- Configuration via dataclasses (or Hydra in the future)
- Multi-modal feature extraction (DINO + Fast-DROID)
- Flexible callback system (WandB, checkpointing, early stopping)
- Automatic experiment tracking and reproducibility

Example usage:
    # Basic training
    python scripts/train.py --task-ids 327 --num-epochs 100

    # With WandB logging
    python scripts/train.py --task-ids 327 --wandb-project my-project

    # Resume from checkpoint
    python scripts/train.py --resume ./checkpoints/best_model.pt

    # Custom configuration
    python scripts/train.py --config configs/task_327.json

Author: Claude Code
Date: October 24, 2025
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import json
from dataclasses import asdict

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model, ModelConfig
from models.losses import get_loss_function
from training import (
    BoundaryDataset,
    TrainingDatasetConfig,
    collate_dict,
    BoundaryTrainer,
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    LearningRateSchedulerCallback,
    TrainingConfig,
    WandBConfig
)
from data.splits import load_split_manifest
from utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train boundary prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument(
        '--task-ids',
        type=int,
        nargs='+',
        help='Task IDs to train on (alternative to using split manifests)'
    )

    parser.add_argument(
        '--train-split',
        type=str,
        help='Path to train split manifest JSON'
    )

    parser.add_argument(
        '--val-split',
        type=str,
        help='Path to validation split manifest JSON'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Cache directory for features'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=5.0,
        help='Target FPS for data (should match preprocessing target_fps)'
    )

    # Model
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=2048,
        help='Tokenizer vocabulary size'
    )

    parser.add_argument(
        '--prediction-window-size',
        type=int,
        default=18,
        help='Prediction window size'
    )

    # Training
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size'
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-3,
        help='Learning rate'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay'
    )

    parser.add_argument(
        '--loss-type',
        type=str,
        default='bce',
        choices=['bce', 'focal', 'soft_focal', 'weighted_bce'],
        help='Loss function type'
    )

    parser.add_argument(
        '--bce-pos-weight',
        type=float,
        default=10.0,
        help='Positive class weight for BCE loss'
    )

    # Callbacks
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='WandB project name (enables WandB logging)'
    )

    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='WandB run name'
    )

    parser.add_argument(
        '--wandb-offline',
        action='store_true',
        help='Run WandB in offline mode'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )

    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )

    # System
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file (overrides CLI args)'
    )

    # Resume
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def create_datasets(
    args: argparse.Namespace,
    device: torch.device
) -> tuple:
    """Create training and validation datasets.

    Args:
        args: Command line arguments
        device: Device for feature extraction

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("\n" + "="*80)
    print("Creating datasets...")
    print("="*80)

    # Dataset configuration
    dataset_config = TrainingDatasetConfig(
        window_size=180,
        stride=10,
        vocab_size=args.vocab_size,
        fps=args.fps,
        num_workers=args.num_workers,
        device=str(device),
        ray_worker_device=str(device),
        cache_path=args.cache_dir
    )

    # Load training data
    if args.train_split:
        print(f"Loading training split from {args.train_split}")
        train_manifest = load_split_manifest(args.train_split)
        train_trajectories = train_manifest.trajectories
    elif args.task_ids:
        print(f"Loading training data for tasks: {args.task_ids}")
        # Load from task IDs
        from agibot.dataset.simple_dataloader import create_dataloader
        data_loader = create_dataloader(task_ids=args.task_ids)
        train_trajectories = list(data_loader)
    else:
        raise ValueError("Must specify either --train-split or --task-ids")

    print(f"Creating training dataset from {len(train_trajectories)} trajectories...")
    train_dataset = BoundaryDataset(train_trajectories, dataset_config)

    # Load validation data
    val_dataset = None
    if args.val_split:
        print(f"\nLoading validation split from {args.val_split}")
        val_manifest = load_split_manifest(args.val_split)
        val_trajectories = val_manifest.trajectories

        print(f"Creating validation dataset from {len(val_trajectories)} trajectories...")
        val_dataset = BoundaryDataset(val_trajectories, dataset_config)

    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: BoundaryDataset,
    val_dataset: Optional[BoundaryDataset],
    args: argparse.Namespace
) -> tuple:
    """Create data loaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        args: Command line arguments

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("\nCreating data loaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_dict,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_dict,
            pin_memory=True
        )

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


def create_model_and_optimizer(
    args: argparse.Namespace,
    device: torch.device
) -> tuple:
    """Create model, optimizer, scheduler, and loss function.

    Args:
        args: Command line arguments
        device: Device to use

    Returns:
        Tuple of (model, optimizer, scheduler, criterion)
    """
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)

    # Model configuration
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        prediction_window_size=args.prediction_window_size,
        use_dino=True,
        use_proprio=True
    )

    model = create_model(model_config, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # Loss function
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(args.bce_pos_weight)
        )
    else:
        criterion = get_loss_function(
            args.loss_type,
            pos_weight=args.bce_pos_weight
        )

    print(f"Optimizer: Adam (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Loss: {args.loss_type}")

    return model, optimizer, scheduler, criterion


def create_callbacks(args: argparse.Namespace) -> list:
    """Create training callbacks.

    Args:
        args: Command line arguments

    Returns:
        List of callback instances
    """
    callbacks = []

    # WandB callback
    if args.wandb_project:
        wandb_callback = WandBCallback(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            log_plots_every=3,
            num_plots=3,
            offline=args.wandb_offline
        )
        callbacks.append(wandb_callback)
        print("✓ WandB logging enabled")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_dir=args.checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_every=args.save_every
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ Checkpointing enabled (save_dir={args.checkpoint_dir})")

    # Early stopping callback
    if args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            monitor='val_loss',
            patience=args.patience,
            mode='min'
        )
        callbacks.append(early_stopping_callback)
        print(f"✓ Early stopping enabled (patience={args.patience})")

    return callbacks


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[any] = None
) -> int:
    """Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore (optional)

    Returns:
        Epoch number to resume from
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    print(f"Resumed from epoch {epoch}")

    return epoch


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if specified
    if args.config:
        config = load_config_from_file(args.config)
        # Update args with config
        for key, value in config.items():
            setattr(args, key, value)

    # Print banner
    print("="*80)
    print("Boundary Prediction Model Training")
    print("="*80)

    # Set random seed
    set_all_seeds(args.seed)
    print(f"Random seed: {args.seed}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(args, device)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args)

    # Create model, optimizer, scheduler, and loss
    model, optimizer, scheduler, criterion = create_model_and_optimizer(args, device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Create callbacks
    print("\n" + "="*80)
    print("Configuring callbacks...")
    print("="*80)
    callbacks = create_callbacks(args)

    # Add scheduler callback
    scheduler_callback = LearningRateSchedulerCallback(scheduler)
    callbacks.append(scheduler_callback)

    # Create trainer
    print("\n" + "="*80)
    print("Initializing trainer...")
    print("="*80)

    trainer = BoundaryTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        scheduler=scheduler,
        prediction_stride=10,
        prediction_window_size=args.prediction_window_size
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        log_every=10,
        collect_plots_every=3
    )

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Print final statistics
    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    if val_loader:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
