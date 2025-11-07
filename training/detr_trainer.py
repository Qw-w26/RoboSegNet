"""Training loop for DETR-style boundary detection.

This module provides a custom training loop for DETR models that handles:
- Set-based targets and Hungarian matching loss
- DETR-specific metrics (precision, recall, F1)
- Gradient clipping (critical for DETR stability)
- Callback support for logging and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

from training.detr_utils import compute_detr_metrics


def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Move target tensors to specified device.

    Args:
        targets: List of target dictionaries
        device: Target device

    Returns:
        List of target dictionaries with tensors on device
    """
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in target.items()}
            for target in targets]


class DETRTrainerWrapper:
    """Minimal trainer wrapper for callback compatibility."""
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_model(self):
        """Get unwrapped model (handles DDP)."""
        return self.model.module if hasattr(self.model, 'module') else self.model


def train_detr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    callbacks: Optional[List[Any]] = None,
    gradient_clip: float = 0.1,
    gradient_accumulation_steps: int = 1,
    window_size: int = 18,
    window_fps: float = 2.0,
    metric_tolerances: List[float] = None,
    confidence_threshold: float = 0.5,
    checkpoint_dir: str = './checkpoints',
    log_every: int = 10,
    is_main_process: bool = True,
    rank: int = 0
):
    """Train DETR model with set-based supervision.

    Args:
        model: DETR model
        optimizer: Optimizer (AdamW recommended)
        criterion: SetCriterion loss
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        scheduler: Optional learning rate scheduler
        callbacks: List of callbacks for logging/checkpointing
        gradient_clip: Gradient clipping value (critical for DETR)
        gradient_accumulation_steps: Number of steps to accumulate gradients (effective_batch_size = batch_size * accumulation_steps)
        window_size: Window size in frames
        window_fps: FPS of the window (for time-to-frame conversion)
        metric_tolerances: List of tolerances in seconds for computing metrics
        confidence_threshold: Threshold for extracting predictions
        checkpoint_dir: Directory to save checkpoints
        log_every: Log frequency (batches)
        is_main_process: Whether this is the main process
        rank: Process rank (for distributed training)
    """
    if metric_tolerances is None:
        metric_tolerances = [1.0, 2.0, 3.0]  # Default tolerances in seconds
    if callbacks is None:
        callbacks = []

    # Convert tolerances from seconds to frames
    tolerance_frames = [tol * window_fps for tol in metric_tolerances]

    # Detect if scheduler is warmup-based (needs per-batch stepping)
    # Warmup schedulers are typically LambdaLR
    from torch.optim.lr_scheduler import LambdaLR
    is_warmup_scheduler = isinstance(scheduler, LambdaLR) if scheduler is not None else False

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create trainer wrapper for callbacks
    trainer_wrapper = DETRTrainerWrapper(model, optimizer, scheduler)

    # Call on_train_begin for all callbacks (to initialize wandb, etc.)
    if is_main_process:
        for callback in callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(trainer_wrapper)

    # Training history
    history = {
        'train_loss': [],
        'train_loss_position': [],
        'train_loss_confidence': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("=" * 80)

        # Training
        model.train()
        train_loss_sum = 0.0
        train_pos_loss_sum = 0.0
        train_conf_loss_sum = 0.0
        num_train_batches = 0

        train_iterator = tqdm(train_loader, desc="Training") if is_main_process else train_loader

        # Zero gradients at start of epoch
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_iterator):
            # Move data to device
            tokenized_actions = batch['tokenized_actions'].to(device)
            dino_features = batch['dino_features'].to(device)
            targets = move_targets_to_device(batch['targets'], device)

            # Forward pass
            predictions = model(tokenized_actions, None, dino_features)

            # Compute loss with Hungarian matching
            losses = criterion(predictions, targets)
            loss = losses['loss_total']

            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Accumulate losses (unscaled for logging)
            train_loss_sum += loss.item() * gradient_accumulation_steps
            train_pos_loss_sum += losses['loss_position'].item()
            train_conf_loss_sum += losses['loss_confidence'].item()
            num_train_batches += 1

            # Only update weights every accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping (CRITICAL for DETR stability)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Step warmup scheduler per batch (after optimizer step)
                if is_warmup_scheduler:
                    scheduler.step()

            # Logging
            if is_main_process and (batch_idx + 1) % log_every == 0:
                avg_loss = train_loss_sum / num_train_batches
                avg_pos_loss = train_pos_loss_sum / num_train_batches
                avg_conf_loss = train_conf_loss_sum / num_train_batches
                lr = optimizer.param_groups[0]['lr']

                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={avg_loss:.4f} (pos={avg_pos_loss:.4f}, conf={avg_conf_loss:.4f}), "
                      f"LR={lr:.6f}")

            global_step += 1

        # Average training metrics
        avg_train_loss = train_loss_sum / num_train_batches
        avg_train_pos_loss = train_pos_loss_sum / num_train_batches
        avg_train_conf_loss = train_conf_loss_sum / num_train_batches

        history['train_loss'].append(avg_train_loss)
        history['train_loss_position'].append(avg_train_pos_loss)
        history['train_loss_confidence'].append(avg_train_conf_loss)

        if is_main_process:
            print(f"\nTrain Loss: {avg_train_loss:.4f} "
                  f"(pos={avg_train_pos_loss:.4f}, conf={avg_train_conf_loss:.4f})")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_pos_loss_sum = 0.0
            val_conf_loss_sum = 0.0
            # Initialize metric accumulators for all tolerances
            val_metrics = {}
            for tol_frames, tol_sec in zip(tolerance_frames, metric_tolerances):
                val_metrics[f'precision@{tol_frames}'] = 0.0
                val_metrics[f'recall@{tol_frames}'] = 0.0
                val_metrics[f'f1@{tol_frames}'] = 0.0
            num_val_batches = 0

            val_iterator = tqdm(val_loader, desc="Validation") if is_main_process else val_loader

            with torch.no_grad():
                for batch in val_iterator:
                    tokenized_actions = batch['tokenized_actions'].to(device)
                    dino_features = batch['dino_features'].to(device)
                    targets = move_targets_to_device(batch['targets'], device)

                    predictions = model(tokenized_actions, None, dino_features)
                    losses = criterion(predictions, targets)

                    val_loss_sum += losses['loss_total'].item()
                    val_pos_loss_sum += losses['loss_position'].item()
                    val_conf_loss_sum += losses['loss_confidence'].item()

                    # Compute metrics with mAP disabled for speed
                    batch_metrics = compute_detr_metrics(
                        predictions, targets, window_size, confidence_threshold, tolerance_frames,
                        compute_map=False
                    )
                    # Accumulate metrics for all tolerances
                    for key, value in batch_metrics.items():
                        if key in val_metrics:
                            val_metrics[key] += value

                    num_val_batches += 1

            avg_val_loss = val_loss_sum / num_val_batches
            avg_val_pos_loss = val_pos_loss_sum / num_val_batches
            avg_val_conf_loss = val_conf_loss_sum / num_val_batches

            # Average metrics for all tolerances
            avg_val_metrics = {}
            for key in val_metrics:
                avg_val_metrics[key] = val_metrics[key] / num_val_batches

            # Get first tolerance metrics for display
            first_tol_frames = tolerance_frames[0]
            avg_val_precision = avg_val_metrics.get(f'precision@{first_tol_frames}', 0.0)
            avg_val_recall = avg_val_metrics.get(f'recall@{first_tol_frames}', 0.0)
            avg_val_f1 = avg_val_metrics.get(f'f1@{first_tol_frames}', 0.0)

            history['val_loss'].append(avg_val_loss)
            history['val_f1'].append(avg_val_f1)
            history['val_precision'].append(avg_val_precision)
            history['val_recall'].append(avg_val_recall)

            if is_main_process:
                print(f"Val Loss: {avg_val_loss:.4f} "
                      f"(pos={avg_val_pos_loss:.4f}, conf={avg_val_conf_loss:.4f})")
                print(f"Val Metrics: P={avg_val_precision:.3f}, "
                      f"R={avg_val_recall:.3f}, F1={avg_val_f1:.3f}")

            # Save best model
            if avg_val_loss < best_val_loss and is_main_process:
                best_val_loss = avg_val_loss
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_val_loss,
                    'f1': avg_val_f1
                }, checkpoint_path)
                print(f"✓ Saved best model (val_loss={avg_val_loss:.4f}, val_f1={avg_val_f1:.3f})")

        # Step epoch-based scheduler (warmup schedulers step per batch instead)
        if scheduler is not None and not is_warmup_scheduler:
            scheduler.step()

        # Callbacks
        if is_main_process:
            callback_logs = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_loss_position': avg_train_pos_loss,
                'train_loss_confidence': avg_train_conf_loss,
            }

            if val_loader is not None:
                callback_logs.update({
                    'val_loss': avg_val_loss,
                    'val_loss_position': avg_val_pos_loss,
                    'val_loss_confidence': avg_val_conf_loss,
                    'val_precision': avg_val_precision,
                    'val_recall': avg_val_recall,
                    'val_f1': avg_val_f1
                })

                # Add tolerance-based metrics for validation
                for tol_frames, tol_sec in zip(tolerance_frames, metric_tolerances):
                    precision_key = f'precision@{tol_frames}'
                    recall_key = f'recall@{tol_frames}'
                    f1_key = f'f1@{tol_frames}'

                    if precision_key in avg_val_metrics:
                        callback_logs[f'val_precision_@{tol_sec}s'] = avg_val_metrics[precision_key]
                    if recall_key in avg_val_metrics:
                        callback_logs[f'val_recall_@{tol_sec}s'] = avg_val_metrics[recall_key]
                    if f1_key in avg_val_metrics:
                        callback_logs[f'val_f1_@{tol_sec}s'] = avg_val_metrics[f1_key]

            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(trainer_wrapper, epoch, callback_logs)

        # Periodic checkpoint
        if is_main_process and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path.name}")

    # Save final model
    if is_main_process:
        final_path = Path(checkpoint_dir) / 'final_model.pt'
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history
        }, final_path)
        print(f"\n✓ Saved final model: {final_path}")

        # Call on_train_end for all callbacks
        for callback in callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(trainer_wrapper, history)

    return history
