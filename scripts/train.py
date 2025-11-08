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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import load_config, validate_config, parse_cli_overrides
from models import create_model, create_detr_criterion
from models.losses import get_loss_function
from training import (
    BoundaryDataset,
    LazyBoundaryDataset,
    TrainingDatasetConfig,
    collate_dict,
    BoundaryTrainer,
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    EmbeddingVisualizationCallback,
)
from training.detr_utils import collate_detr_batch, compute_detr_metrics
from data import load_parquet_split
from utils.reproducibility import set_all_seeds


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr, base_lr):
    """Create a learning rate scheduler with linear warmup and cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate after cosine decay
        base_lr: Base learning rate (max learning rate after warmup)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))

        # Scale between min_lr and base_lr
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialize distributed training.

    Returns:
        Tuple of (local_rank, global_rank, world_size, is_distributed)
    """
    # Check if running in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Distributed training via torchrun
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Set master port if not already set (fallback)
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'  # Change this default if needed

        # Initialize process group
        dist.init_process_group(backend='nccl')

        # Set device
        torch.cuda.set_device(local_rank)

        return local_rank, rank, world_size, True
    else:
        # Single GPU or CPU training
        return 0, 0, 1, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


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
    # Setup distributed training
    local_rank, rank, world_size, is_distributed = setup_distributed()

    args = parse_args()

    # Parse CLI overrides
    overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    # Load configuration
    if is_main_process(rank):
        print("="*80)
        print("Loading configuration...")
        print("="*80)
    config = load_config(args.config, overrides=overrides)
    validate_config(config)

    if is_main_process(rank):
        print(f"Experiment: {config.experiment_name}")
        print(f"Config file: {args.config}")
        if overrides:
            print(f"Overrides: {overrides}")

    # Set random seed (different for each rank for data augmentation diversity)
    seed = config.training.seed + rank
    set_all_seeds(seed)
    if is_main_process(rank):
        print(f"Random seed: {config.training.seed} (base) + {rank} (rank) = {seed}")

    # Setup device for DDP
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        if config.training.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(config.training.device)

    if is_main_process(rank):
        if is_distributed:
            print(f"Distributed training: {world_size} GPUs")
            print(f"Device: {device} (rank {rank}/{world_size})")
        else:
            print(f"Device: {device}")

    # Load data
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Loading data...")
        print("="*80)

    from data import load_multi_task_split

    if is_main_process(rank):
        print(f"Meta directory: {config.data.meta_dir}")
        print(f"Training tasks: {config.data.train_task_ids}")
        print(f"Evaluation tasks: {config.data.eval_task_ids}")

    train_trajectories = load_multi_task_split(
        'train',
        task_ids=config.data.train_task_ids,
        meta_dir=config.data.meta_dir
    )

    val_trajectories = load_multi_task_split(
        'val',
        task_ids=config.data.eval_task_ids,
        meta_dir=config.data.meta_dir
    )

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

    if is_main_process(rank):
        print("\nCreating training dataset (lazy loading)...")
    # Use LazyBoundaryDataset for faster initialization
    train_dataset = LazyBoundaryDataset(train_trajectories, dataset_config)

    if is_main_process(rank):
        print("\nCreating validation dataset (lazy loading)...")
    val_dataset = LazyBoundaryDataset(val_trajectories, dataset_config)

    if is_main_process(rank):
        print(f"\nDataset statistics:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

    # Determine collate function based on architecture
    is_detr = getattr(config.model, 'architecture', 'mlp') == 'boundary_detr'
    collate_fn = collate_detr_batch if is_detr else collate_dict

    if is_main_process(rank) and is_detr:
        print("Using DETR collate function (set-based targets)")

    # Create data loaders with DistributedSampler
    if is_main_process(rank):
        print("\nCreating data loaders...")

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if is_distributed else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=2  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False,
        prefetch_factor=2
    )

    if is_main_process(rank):
        print(f"  Train batches per GPU: {len(train_loader)}")
        print(f"  Val batches per GPU: {len(val_loader)}")
        if is_distributed:
            print(f"  Total train batches: {len(train_loader) * world_size}")
            print(f"  Total val batches: {len(val_loader) * world_size}")

    # Create model
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Creating model...")
        print("="*80)

    model = create_model(config.model, device=device)

    if is_main_process(rank):
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap model with DDP for distributed training
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Enable to handle unused parameters and inplace ops
        )
        if is_main_process(rank):
            print(f"Model wrapped with DistributedDataParallel (find_unused_parameters=True)")

    # Create optimizer (use AdamW for DETR)
    optimizer_class = optim.AdamW if is_detr else optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Create scheduler
    scheduler = None
    if config.training.scheduler == 'cosine':
        # Check if warmup is specified
        warmup_steps = getattr(config.training, 'warmup_steps', 0)

        if warmup_steps > 0:
            # Use warmup scheduler with cosine annealing
            # Calculate total steps (approximate, using epochs and dataset size)
            # This will be updated in the training loop for step-based scheduling
            total_steps = config.training.num_epochs * len(train_loader)

            scheduler = get_warmup_cosine_scheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=config.training.min_lr,
                base_lr=config.training.learning_rate
            )
        else:
            # Use standard cosine annealing (epoch-based)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.training.min_lr
            )

    # Create loss function
    if is_detr:
        # Use DETR SetCriterion
        criterion = create_detr_criterion(
            type('Config', (), {
                'num_boundary_queries': config.model.num_boundary_queries,
                'loss_position_weight': config.training.loss_position_weight,
                'loss_confidence_weight': config.training.loss_confidence_weight,
                'loss_eos_coef': config.training.loss_eos_coef,
                'get': lambda self, key, default=None: getattr(self, key, default)
            })(),
            use_focal_loss=getattr(config.training, 'use_focal_loss', False)
        )
        loss_type_name = 'SetCriterion (Hungarian matching)'
    elif config.training.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.training.loss_pos_weight)
        )
        loss_type_name = config.training.loss_type
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
        loss_type_name = config.training.loss_type

    if is_main_process(rank):
        optimizer_name = "AdamW" if is_detr else "Adam"
        print(f"Optimizer: {optimizer_name} (lr={config.training.learning_rate}, wd={config.training.weight_decay})")
        print(f"Scheduler: {config.training.scheduler}")
        if is_detr:
            print(f"Loss: {loss_type_name}")
            print(f"  Position weight: {config.training.loss_position_weight}")
            print(f"  Confidence weight: {config.training.loss_confidence_weight}")
            print(f"  EOS coefficient: {config.training.loss_eos_coef}")
        else:
            print(f"Loss: {loss_type_name}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if is_main_process(rank):
            print(f"\nLoading checkpoint from {args.resume}")
        # Load checkpoint on all ranks
        checkpoint = torch.load(args.resume, map_location=device)

        # Load model state (handle DDP wrapper)
        if is_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Checkpoint saves the last completed epoch, so resume from the next one
        start_epoch = checkpoint.get('epoch', 0) + 1

        if is_main_process(rank):
            print(f"Resumed from checkpoint (last completed epoch: {checkpoint.get('epoch', 0)})")
            print(f"Starting training from epoch {start_epoch}")

    # Create callbacks (only on main process to avoid duplicate logging)
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Configuring callbacks...")
        print("="*80)

    # Create callbacks only on main process to avoid duplicate logging
    callbacks = []

    if is_main_process(rank):
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

    # Create trainer or use custom loop for DETR
    if is_main_process(rank):
        print("\n" + "="*80)
        print("Initializing trainer...")
        print("="*80)

    if is_detr:
        # Use custom training loop for DETR
        from training.detr_trainer import train_detr

        if is_main_process(rank):
            print("Using custom DETR training loop")

        try:
            train_detr(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.training.num_epochs,
                device=device,
                scheduler=scheduler,
                callbacks=callbacks,
                gradient_clip=config.training.gradient_clip_val,
                gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 1),
                window_size=config.model.prediction_window_size,
                window_fps=config.data.fps,
                metric_tolerances=config.training.metric_tolerances,
                confidence_threshold=getattr(config.training, 'confidence_threshold', 0.5),
                checkpoint_dir=config.training.checkpoint_dir,
                log_every=config.training.log_every,
                is_main_process=is_main_process(rank),
                rank=rank,
                start_epoch=start_epoch
            )

            if is_main_process(rank):
                print("\n" + "="*80)
                print("Training complete!")
                print("="*80)
                print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
                print(f"Experiment: {config.experiment_name}")

        finally:
            # Cleanup distributed training
            cleanup_distributed()

    else:
        # Use standard BoundaryTrainer for non-DETR models
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
            metric_tolerances=config.training.metric_tolerances,
            train_task_ids=config.data.train_task_ids  # For cross-task evaluation tracking
        )

        # Train
        if is_main_process(rank):
            print("\n" + "="*80)
            print("Starting training...")
            print("="*80)

        try:
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.training.num_epochs,
                log_every=config.training.log_every,
                collect_plots_every=config.training.log_plots_every,
                start_epoch=start_epoch
            )

            if is_main_process(rank):
                print("\n" + "="*80)
                print("Training complete!")
                print("="*80)

                # Print final statistics
                print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
                if val_loader:
                    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

                print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}")
                print(f"Experiment: {config.experiment_name}")

        finally:
            # Cleanup distributed training
            cleanup_distributed()


if __name__ == '__main__':
    main()
