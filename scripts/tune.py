#!/usr/bin/env python3
"""Hyperparameter tuning script for DETR-based boundary detection models.

This script enables parallel hyperparameter search specifically optimized for DETR
architecture with Hungarian matching and set-based supervision. It uses Ray Tune
for distributed optimization.

Key DETR-specific parameters tuned:
    - Loss weights (position, confidence) and EOS coefficient
    - Hungarian matcher costs
    - Warmup steps and gradient clipping
    - DETR dropout rates
    - Architecture capacity (queries, layers, d_model)

Example usage:
    # Standard search (Tier 1+2+3: losses + optimization + dropout)
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-trials 20 --tier standard

    # Essential only (Tier 1+2: losses + optimization, fastest)
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-trials 10 --tier essential

    # Full search (Tier 1+2+3+4: includes architecture)
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-trials 50 --tier full --gpus-per-trial 0.5

    # Experimental (Tier 1-5: includes focal loss exploration)
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-trials 100 --tier experimental

    # With specific search algorithm and scheduler
    python scripts/tune.py --config configs/experiments/multi_task_327_352_356_detr.yaml \
        --num-trials 30 --search-alg hyperopt --scheduler asha --tier standard

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
from models import create_model, create_detr_criterion
from models.losses import get_loss_function
from training import (
    BoundaryDataset,
    LazyBoundaryDataset,
    TrainingDatasetConfig,
    collate_dict,
    BoundaryTrainer,
)
from training.detr_utils import collate_detr_batch, compute_detr_metrics
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

    # Override with hyperparameters from Ray Tune (DETR-specific)
    base_config.training.learning_rate = config_dict['learning_rate']
    base_config.training.weight_decay = config_dict['weight_decay']
    base_config.data.batch_size = config_dict['batch_size']

    # Tier 1: Loss weights and Hungarian matcher (MOST CRITICAL)
    base_config.training.loss_position_weight = config_dict['loss_position_weight']
    base_config.training.loss_confidence_weight = config_dict['loss_confidence_weight']
    base_config.training.loss_eos_coef = config_dict['loss_eos_coef']
    base_config.training.matcher_cost_position = config_dict.get('matcher_cost_position', getattr(base_config.training, 'matcher_cost_position', 5.0))
    base_config.training.matcher_cost_confidence = config_dict.get('matcher_cost_confidence', getattr(base_config.training, 'matcher_cost_confidence', 2.0))

    # Tier 2: Optimization (training stability)
    if 'warmup_steps' in config_dict:
        base_config.training.warmup_steps = config_dict['warmup_steps']
    if 'gradient_clip_val' in config_dict:
        base_config.training.gradient_clip_val = config_dict['gradient_clip_val']

    # Tier 3: Regularization (dropout)
    if 'detr_dropout' in config_dict:
        base_config.model.detr_dropout = config_dict['detr_dropout']
    if 'attention_dropout' in config_dict:
        base_config.model.attention_dropout = config_dict['attention_dropout']
    if 'visual_dropout' in config_dict:
        base_config.model.visual_dropout = config_dict['visual_dropout']

    # Tier 4: Architecture (model capacity - optional for advanced tuning)
    if 'num_boundary_queries' in config_dict:
        base_config.model.num_boundary_queries = config_dict['num_boundary_queries']
    if 'num_encoder_layers' in config_dict:
        base_config.model.num_encoder_layers = config_dict['num_encoder_layers']
    if 'num_decoder_layers' in config_dict:
        base_config.model.num_decoder_layers = config_dict['num_decoder_layers']
    if 'd_model' in config_dict:
        base_config.model.d_model = config_dict['d_model']
    if 'detr_feedforward_dim' in config_dict:
        base_config.model.detr_feedforward_dim = config_dict['detr_feedforward_dim']

    # Tier 5: Focal loss (optional - if use_focal_loss is True)
    if 'use_focal_loss' in config_dict:
        base_config.training.use_focal_loss = config_dict['use_focal_loss']
    if 'loss_focal_alpha' in config_dict:
        base_config.training.loss_focal_alpha = config_dict['loss_focal_alpha']
    if 'loss_focal_gamma' in config_dict:
        base_config.training.loss_focal_gamma = config_dict['loss_focal_gamma']

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

    # Use LazyBoundaryDataset for faster initialization (lazy loading)
    train_dataset = LazyBoundaryDataset(train_trajectories, dataset_config)
    val_dataset = LazyBoundaryDataset(val_trajectories, dataset_config)

    # Determine collate function based on architecture
    is_detr = getattr(base_config.model, 'architecture', 'mlp') == 'boundary_detr'
    collate_fn = collate_detr_batch if is_detr else collate_dict

    # Create data loaders
    # IMPORTANT: Use num_workers=0 for Ray Tune to avoid nested multiprocessing deadlocks
    # Ray already handles parallelism across trials
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config.data.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing in Ray workers
        collate_fn=collate_fn,
        pin_memory=base_config.data.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config.data.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing in Ray workers
        collate_fn=collate_fn,
        pin_memory=base_config.data.pin_memory
    )

    # Create model
    model = create_model(base_config.model, device=device)

    # Create optimizer (AdamW recommended for transformers/DETR)
    optimizer_class = optim.AdamW if is_detr else optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=base_config.training.learning_rate,
        weight_decay=base_config.training.weight_decay
    )

    # Create scheduler with warmup support
    scheduler = None
    if is_detr and hasattr(base_config.training, 'warmup_steps') and base_config.training.warmup_steps > 0:
        # DETR uses warmup + cosine annealing
        # Import warmup scheduler from train.py
        from scripts.train import get_warmup_cosine_scheduler

        total_steps = base_config.training.num_epochs * len(train_dataset) // base_config.data.batch_size
        scheduler = get_warmup_cosine_scheduler(
            optimizer=optimizer,
            warmup_steps=base_config.training.warmup_steps,
            total_steps=total_steps,
            min_lr=getattr(base_config.training, 'min_lr', 1e-6),
            base_lr=base_config.training.learning_rate
        )
    elif base_config.training.scheduler == 'cosine':
        # Regular cosine without warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=base_config.training.num_epochs,
            eta_min=getattr(base_config.training, 'min_lr', 1e-6)
        )

    # Create loss function
    if is_detr:
        # DETR uses SetCriterion with Hungarian matching
        # Create combined config object for criterion
        criterion = create_detr_criterion(
            type('Config', (), {
                'num_boundary_queries': base_config.model.num_boundary_queries,
                'loss_position_weight': base_config.training.loss_position_weight,
                'loss_confidence_weight': base_config.training.loss_confidence_weight,
                'loss_eos_coef': base_config.training.loss_eos_coef,
                'matcher_cost_position': getattr(base_config.training, 'matcher_cost_position', 5.0),
                'matcher_cost_confidence': getattr(base_config.training, 'matcher_cost_confidence', 2.0),
                'loss_focal_alpha': getattr(base_config.training, 'loss_focal_alpha', 0.25),
                'loss_focal_gamma': getattr(base_config.training, 'loss_focal_gamma', 2.0),
                'get': lambda self, key, default=None: getattr(self, key, default)
            })(),
            use_focal_loss=getattr(base_config.training, 'use_focal_loss', False)
        )
    elif base_config.training.loss_type == 'bce':
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

    if is_detr:
        # DETR training loop
        from training.detr_trainer import train_detr, move_targets_to_device

        # Convert metric tolerances to frame-based
        metric_tolerances = getattr(base_config.training, 'metric_tolerances', [1.0, 2.0, 3.0])
        tolerance_frames = [tol * base_config.data.fps for tol in metric_tolerances]

        # Get gradient clipping value
        gradient_clip = getattr(base_config.training, 'gradient_clip_val', 0.1)

        # Training loop with Ray Tune reporting
        print(f"Starting training with {len(train_loader)} batches per epoch")

        for epoch in range(start_epoch, base_config.training.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{base_config.training.num_epochs}")
            print(f"{'='*80}")

            # Training
            model.train()
            train_loss_sum = 0.0
            train_metrics_sum = {}
            for tol in tolerance_frames:
                train_metrics_sum[f'precision@{tol}'] = 0.0
                train_metrics_sum[f'recall@{tol}'] = 0.0
                train_metrics_sum[f'f1@{tol}'] = 0.0
            num_train_batches = 0

            # For periodic reporting during epoch
            log_interval = max(1, len(train_loader) // 10)  # Log 10 times per epoch

            for batch_idx, batch in enumerate(train_loader):
                # Log progress every N batches to show we're not stuck
                if batch_idx % log_interval == 0:
                    print(f"  Processing batch {batch_idx}/{len(train_loader)} "
                          f"(Epoch {epoch+1}/{base_config.training.num_epochs})")

                tokenized_actions = batch['tokenized_actions'].to(device)
                dino_features = batch['dino_features'].to(device)
                targets = move_targets_to_device(batch['targets'], device)

                predictions = model(tokenized_actions, None, dino_features)
                losses = criterion(predictions, targets)
                loss = losses['loss_total']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

                # Step warmup scheduler per batch if using warmup
                if scheduler is not None and hasattr(base_config.training, 'warmup_steps') and base_config.training.warmup_steps > 0:
                    scheduler.step()

                train_loss_sum += loss.item()

                # Compute training metrics periodically (less frequently to save time)
                # Only compute every 50 batches during training to avoid slowdown
                if batch_idx % 50 == 0:
                    batch_metrics = compute_detr_metrics(
                        predictions, targets, base_config.model.prediction_window_size,
                        getattr(base_config.training, 'confidence_threshold', 0.5),
                        tolerance_frames,
                        compute_map=False  # Disable mAP during training for speed
                    )
                    for key, value in batch_metrics.items():
                        if key in train_metrics_sum:
                            # Convert to Python scalar if it's a tensor
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            train_metrics_sum[key] += value

                num_train_batches += 1

                # Log running average every log_interval batches
                if (batch_idx + 1) % log_interval == 0 and num_train_batches > 0:
                    avg_loss_so_far = train_loss_sum / num_train_batches
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                          f"Avg Loss={avg_loss_so_far:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

            avg_train_loss = train_loss_sum / num_train_batches
            # Average metrics: count how many times we computed them (every 50 batches)
            num_metric_computations = max((num_train_batches + 49) // 50, 1)
            avg_train_metrics = {k: v / num_metric_computations for k, v in train_metrics_sum.items()}

            first_tol = tolerance_frames[0]
            train_f1 = avg_train_metrics.get(f'f1@{first_tol}', 0.0)
            print(f"\n  Training complete: Avg Loss={avg_train_loss:.4f}, F1={train_f1:.3f}")

            # Validation
            print(f"\n  Starting validation ({len(val_loader)} batches)...")
            model.eval()
            val_loss_sum = 0.0
            val_metrics_sum = {}
            for tol in tolerance_frames:
                val_metrics_sum[f'precision@{tol}'] = 0.0
                val_metrics_sum[f'recall@{tol}'] = 0.0
                val_metrics_sum[f'f1@{tol}'] = 0.0
            num_val_batches = 0

            val_log_interval = max(1, len(val_loader) // 5)  # Log 5 times during validation

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx % val_log_interval == 0:
                        print(f"    Validating batch {batch_idx}/{len(val_loader)}")

                    tokenized_actions = batch['tokenized_actions'].to(device)
                    dino_features = batch['dino_features'].to(device)
                    targets = move_targets_to_device(batch['targets'], device)

                    predictions = model(tokenized_actions, None, dino_features)
                    losses = criterion(predictions, targets)

                    val_loss_sum += losses['loss_total'].item()

                    batch_metrics = compute_detr_metrics(
                        predictions, targets, base_config.model.prediction_window_size,
                        getattr(base_config.training, 'confidence_threshold', 0.5),
                        tolerance_frames,
                        compute_map=False  # Disable mAP for speed during tuning
                    )
                    for key, value in batch_metrics.items():
                        if key in val_metrics_sum:
                            # Convert to Python scalar if it's a tensor
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            val_metrics_sum[key] += value

                    num_val_batches += 1

            avg_val_loss = val_loss_sum / num_val_batches
            avg_val_metrics = {k: v / num_val_batches for k, v in val_metrics_sum.items()}

            val_f1 = avg_val_metrics.get(f'f1@{first_tol}', 0.0)
            val_precision = avg_val_metrics.get(f'precision@{first_tol}', 0.0)
            val_recall = avg_val_metrics.get(f'recall@{first_tol}', 0.0)
            print(f"  Validation complete: Loss={avg_val_loss:.4f}, "
                  f"P={val_precision:.3f}, R={val_recall:.3f}, F1={val_f1:.3f}")

            # Update epoch-based scheduler (not warmup scheduler - that steps per batch)
            if scheduler is not None and not (hasattr(base_config.training, 'warmup_steps') and base_config.training.warmup_steps > 0):
                scheduler.step()

            # Prepare metrics for Ray Tune (ensure all values are Python scalars)
            def to_scalar(value):
                """Convert value to Python scalar."""
                if isinstance(value, torch.Tensor):
                    return value.item()
                return float(value)

            metrics = {
                'train_loss': to_scalar(avg_train_loss),
                'train_f1': to_scalar(train_f1),
                'train_precision': to_scalar(avg_train_metrics.get(f'precision@{first_tol}', 0.0)),
                'train_recall': to_scalar(avg_train_metrics.get(f'recall@{first_tol}', 0.0)),
                'val_loss': to_scalar(avg_val_loss),
                'val_f1': to_scalar(val_f1),
                'val_precision': to_scalar(val_precision),
                'val_recall': to_scalar(val_recall),
                'epoch': int(epoch),
                'learning_rate': float(optimizer.param_groups[0]['lr'])
            }

            # Add tolerance-specific metrics
            for tol in tolerance_frames:
                metrics[f'train_f1_tol_{tol}'] = to_scalar(avg_train_metrics.get(f'f1@{tol}', 0.0))
                metrics[f'train_precision_tol_{tol}'] = to_scalar(avg_train_metrics.get(f'precision@{tol}', 0.0))
                metrics[f'train_recall_tol_{tol}'] = to_scalar(avg_train_metrics.get(f'recall@{tol}', 0.0))
                metrics[f'val_f1_tol_{tol}'] = to_scalar(avg_val_metrics.get(f'f1@{tol}', 0.0))
                metrics[f'val_precision_tol_{tol}'] = to_scalar(avg_val_metrics.get(f'precision@{tol}', 0.0))
                metrics[f'val_recall_tol_{tol}'] = to_scalar(avg_val_metrics.get(f'recall@{tol}', 0.0))

            # Report to Ray Tune
            print(f"\n  Reporting to Ray Tune: epoch={epoch+1}, val_f1={val_f1:.3f}, val_loss={avg_val_loss:.4f}")
            tune.report(metrics)

    else:
        # Regular BoundaryTrainer loop
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
        default=32,
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

    parser.add_argument(
        '--tier',
        type=str,
        choices=['essential', 'standard', 'full', 'experimental'],
        default='standard',
        help='Search space tier (essential=Tier1+2, standard=+Tier3, full=+Tier4, experimental=+Tier5)'
    )

    return parser.parse_args()


def get_search_space(base_config_path: str, tier: str = 'full') -> Dict[str, Any]:
    """Define the DETR-specific hyperparameter search space.

    Args:
        base_config_path: Path to base config to extract reasonable ranges
        tier: Search tier:
            - 'essential': Only most critical parameters (Tier 1+2)
            - 'standard': Essential + regularization (Tier 1+2+3)
            - 'full': All parameters including architecture (Tier 1+2+3+4)
            - 'experimental': Full + focal loss exploration

    Returns:
        Dictionary defining the search space
    """
    # Load base config to get reasonable ranges
    base_config = load_config(base_config_path)

    # ===== TIER 1: Loss Weights and Hungarian Matching (ALWAYS INCLUDED) =====
    # These are the MOST CRITICAL parameters for DETR performance
    search_space = {
        # Loss component weights: balance position accuracy vs confidence calibration
        'loss_position_weight': tune.uniform(2.0, 10.0),  # Higher = prioritize position accuracy
        'loss_confidence_weight': tune.uniform(1.0, 5.0),  # Higher = prioritize confidence calibration

        # EOS coefficient: controls class imbalance (lower = upweight rare boundaries)
        # 0.1 means positive class weighted 10x more than negative
        'loss_eos_coef': tune.loguniform(0.05, 0.3),  # Critical for handling sparse boundaries

        # Hungarian matcher costs: control how predictions are matched to ground truth
        # These should generally match the loss weights but can be tuned independently
        'matcher_cost_position': tune.uniform(2.0, 10.0),
        'matcher_cost_confidence': tune.uniform(1.0, 5.0),
    }

    # ===== TIER 2: Optimization (ALWAYS INCLUDED) =====
    # Core training hyperparameters
    search_space.update({
        # Learning rate: DETR needs lower LR than typical models
        'learning_rate': tune.loguniform(1e-5, 5e-4),  # Conservative range for stability

        # Weight decay: important for AdamW optimizer
        'weight_decay': tune.loguniform(1e-5, 1e-3),

        # Batch size: smaller for DETR due to transformer memory requirements
        'batch_size': tune.choice([32, 64, 128, 256]),  # Much smaller than baseline

        # Warmup steps: critical for DETR training stability
        'warmup_steps': tune.choice([500, 1000, 2000, 3000]),

        # Gradient clipping: prevents exploding gradients in transformers
        'gradient_clip_val': tune.uniform(0.05, 0.5),
    })

    # ===== TIER 3: Regularization (Dropout) =====
    if tier in ['standard', 'full', 'experimental']:
        search_space.update({
            # DETR transformer dropout
            'detr_dropout': tune.uniform(0.05, 0.3),

            # Cross-modal attention dropout
            'attention_dropout': tune.uniform(0.05, 0.2),

            # Visual encoder dropout
            'visual_dropout': tune.uniform(0.1, 0.4),
        })

    # ===== TIER 4: Architecture (Model Capacity) =====
    if tier in ['full', 'experimental']:
        search_space.update({
            # Number of boundary queries: max boundaries model can detect
            'num_boundary_queries': tune.choice([6, 9, 12]),  # 9 is baseline for 18-frame window

            # Encoder/decoder depth: deeper = more capacity but slower
            'num_encoder_layers': tune.choice([4, 6, 8]),
            'num_decoder_layers': tune.choice([4, 6, 8]),

            # Model dimension: width of transformer
            'd_model': tune.choice([256, 512, 768]),  # 512 is baseline

            # Feedforward dimension: typically 4x d_model
            # Note: This will be automatically adjusted based on d_model
            'detr_feedforward_dim': tune.choice([1024, 2048, 3072]),
        })

    # ===== TIER 5: Experimental (Focal Loss) =====
    if tier == 'experimental':
        search_space.update({
            # Whether to use focal loss instead of BCE for confidence
            'use_focal_loss': tune.choice([False, True]),

            # Focal loss parameters (only used if use_focal_loss=True)
            'loss_focal_alpha': tune.uniform(0.2, 0.4),  # Class balance
            'loss_focal_gamma': tune.uniform(1.5, 3.0),  # Focus on hard examples
        })

    return search_space


def main():
    """Main hyperparameter tuning function."""
    args = parse_args()

    print("="*80)
    print("Ray Tune Hyperparameter Search (DETR-Optimized)")
    print("="*80)
    print(f"Base config: {args.config}")
    print(f"Number of trials: {args.num_trials}")
    print(f"GPUs per trial: {args.gpus_per_trial}")
    print(f"Search tier: {args.tier}")
    print(f"Search algorithm: {args.search_alg}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Metric to optimize: {args.metric} ({args.mode})")
    print(f"Results directory: {args.results_dir}")
    print("="*80)

    # Get DETR-specific search space
    search_space = get_search_space(args.config, tier=args.tier)

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
        # DETR-specific PBT mutations
        scheduler = PopulationBasedTraining(
            time_attr='epoch',
            metric=args.metric,
            mode=args.mode,
            perturbation_interval=5,
            hyperparam_mutations={
                # Core optimization
                'learning_rate': tune.loguniform(1e-5, 5e-4),
                'weight_decay': tune.loguniform(1e-5, 1e-3),
                # Loss weights (critical for DETR)
                'loss_position_weight': tune.uniform(2.0, 10.0),
                'loss_confidence_weight': tune.uniform(1.0, 5.0),
                'loss_eos_coef': tune.loguniform(0.05, 0.3),
            }
        )

    # Setup reporter
    reporter = CLIReporter(
        metric_columns=[args.metric, "val_loss", "train_loss", "learning_rate", "epoch"],
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
