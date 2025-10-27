"""Training callbacks for model training.

This module provides a flexible callback system for training that supports:
- Weights & Biases logging with prediction visualization
- Checkpoint saving (best model and periodic)
- Early stopping
- TensorBoard logging
- Learning rate scheduling

Example:
    >>> from training import Trainer, WandBCallback, CheckpointCallback
    >>> callbacks = [
    ...     WandBCallback(project='my-project'),
    ...     CheckpointCallback(save_dir='./checkpoints')
    ... ]
    >>> trainer = Trainer(model, callbacks=callbacks)
    >>> trainer.fit(train_loader, val_loader)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks are called at specific points during training:
    - on_train_begin: Before training starts
    - on_train_end: After training ends
    - on_epoch_begin: Before each epoch
    - on_epoch_end: After each epoch
    - on_batch_begin: Before each batch (training only)
    - on_batch_end: After each batch (training only)
    - on_validation_begin: Before validation
    - on_validation_end: After validation
    """

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Called before training starts."""
        pass

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Called after training ends."""
        pass

    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        """Called before each epoch."""
        pass

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Called after each epoch."""
        pass

    def on_batch_begin(self, trainer: 'Trainer', batch_idx: int) -> None:
        """Called before each training batch."""
        pass

    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, logs: Dict[str, Any]) -> None:
        """Called after each training batch."""
        pass

    def on_validation_begin(self, trainer: 'Trainer') -> None:
        """Called before validation."""
        pass

    def on_validation_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Called after validation."""
        pass


class WandBCallback(Callback):
    """Weights & Biases logging callback.

    Logs metrics, hyperparameters, and prediction visualizations to W&B.
    Includes andrew's excellent prediction plotting functionality.

    Args:
        project: W&B project name
        entity: W&B entity (team name)
        name: Run name (auto-generated if None)
        config: Configuration dict to log
        log_model: Whether to log model checkpoints
        log_plots_every: Log prediction plots every N epochs
        num_plots: Number of prediction plots to log
        offline: Run in offline mode

    Example:
        >>> callback = WandBCallback(
        ...     project='boundary-prediction',
        ...     log_plots_every=3,
        ...     num_plots=3
        ... )
    """

    def __init__(
        self,
        project: str = 'task-segmentation',
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = True,
        log_plots_every: int = 3,
        num_plots: int = 3,
        offline: bool = False
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.log_model = log_model
        self.log_plots_every = log_plots_every
        self.num_plots = num_plots
        self.offline = offline
        self.run = None

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Initialize W&B run."""
        try:
            import wandb

            if self.offline:
                import os
                os.environ['WANDB_MODE'] = 'offline'

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config
            )

            # Log model architecture
            if self.log_model:
                wandb.watch(trainer.model, log='all', log_freq=100)

        except ImportError:
            warnings.warn("wandb not installed, skipping W&B logging")

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Log metrics and plots at epoch end."""
        if self.run is None:
            return

        import wandb

        # Log scalar metrics
        metrics = {
            'epoch': epoch,
            'train_loss': logs.get('train_loss', 0.0),
            'val_loss': logs.get('val_loss', 0.0),
            'learning_rate': logs.get('learning_rate', 0.0)
        }

        # Add any additional metrics from logs (including IoU, peak distance, etc.)
        for key, value in logs.items():
            if key not in metrics and isinstance(value, (int, float)):
                metrics[key] = value

        wandb.log(metrics)

        # Log prediction plots if it's time
        if (epoch + 1) % self.log_plots_every == 0:
            # Log training predictions (window-level, legacy)
            if 'train_trajectory_data' in logs and logs['train_trajectory_data']:
                self._log_prediction_plots(
                    logs['train_trajectory_data'],
                    epoch=epoch,
                    data_type='train'
                )

            # Log validation predictions (window-level, legacy)
            if 'val_trajectory_data' in logs and logs['val_trajectory_data']:
                self._log_prediction_plots(
                    logs['val_trajectory_data'],
                    epoch=epoch,
                    data_type='val'
                )

        # Log trajectory-level plots (always log when available, regardless of epoch)
        # These are computed from accumulated data across full trajectories
        if 'train_trajectory_plots' in logs and logs['train_trajectory_plots']:
            self._log_trajectory_timeline_plots(
                logs['train_trajectory_plots'],
                epoch=epoch,
                data_type='train'
            )

        if 'val_trajectory_plots' in logs and logs['val_trajectory_plots']:
            self._log_trajectory_timeline_plots(
                logs['val_trajectory_plots'],
                epoch=epoch,
                data_type='val'
            )

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Finish W&B run."""
        if self.run is not None:
            import wandb
            wandb.finish()

    def _log_prediction_plots(
        self,
        trajectory_data: List[Dict[str, Any]],
        epoch: int,
        data_type: str = 'val'
    ) -> None:
        """Log prediction plots to W&B.

        Ported from andrew's excellent visualization code.
        Filters for interesting datapoints (those with boundaries).
        """
        import wandb

        if len(trajectory_data) == 0:
            print(f"No {data_type} trajectory data available for plotting")
            return

        num_plots = self.num_plots ** 2

        # Search through a large random subset for interesting datapoints
        search_size = min(5000, len(trajectory_data))
        search_indices = random.sample(range(len(trajectory_data)), search_size)

        # Filter for interesting datapoints (at least one label > 0.8)
        interesting_indices = []
        for idx in search_indices:
            if np.max(trajectory_data[idx]['ground_truth']) > 0.8:
                interesting_indices.append(idx)

        # Combine interesting and random indices
        random_indices = random.sample(range(len(trajectory_data)),
                                      max(len(interesting_indices), num_plots))
        selected_indices = random.sample(
            interesting_indices + random_indices,
            min(num_plots, len(interesting_indices) + len(random_indices))
        )

        print(f"Searched {search_size} {data_type} datapoints, "
              f"found {len(interesting_indices)} interesting ones, "
              f"logging {len(selected_indices)} {data_type} prediction plots to wandb "
              f"for epoch {epoch}...")

        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(len(selected_indices))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, ax in zip(selected_indices, axes):
            traj_data = trajectory_data[idx]
            self._create_prediction_plot(traj_data, ax)

        # Hide unused subplots
        for ax in axes[len(selected_indices):]:
            ax.axis('off')

        plt.tight_layout()
        wandb.log({f"epoch_{epoch:03d}_{data_type}_prediction_plots": wandb.Image(fig)})
        plt.close(fig)

        print(f"Logged {len(selected_indices)} {data_type} prediction plots "
              f"to wandb for epoch {epoch}")

    def _create_prediction_plot(self, traj_data: Dict[str, Any], ax: plt.Axes) -> None:
        """Create a prediction plot for one trajectory.

        Ported from andrew's visualization code.
        """
        # Get data
        ground_truth = traj_data['ground_truth'].flatten()
        confidence_scores = traj_data['confidence_scores'].flatten()

        # Create time axis
        time_axis = np.arange(len(ground_truth))

        # Plot ground truth and predictions
        ax.plot(time_axis, ground_truth, 'r-', alpha=0.8, linewidth=2,
               label='Train Labels')
        ax.plot(time_axis, confidence_scores, 'b-', alpha=0.7,
               label='Predicted Probabilities')

        # Add title and labels
        ax.set_title(f'Task {traj_data["task_id"]}, Episode {traj_data["episode_id"]}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Auto-scale y-axis to fit the data (ground truth labels may still be normalized differently)
        max_val = max(np.max(ground_truth), np.max(confidence_scores))
        ax.set_ylim(-0.01, max_val * 1.1)

    def _log_trajectory_timeline_plots(
        self,
        trajectory_plots: List[Dict[str, Any]],
        epoch: int,
        data_type: str = 'val'
    ) -> None:
        """Log trajectory-level timeline plots to W&B.

        Creates plots showing full trajectory timelines with:
        - Prediction scores across the entire trajectory
        - Classification threshold line
        - Binary predictions after thresholding
        - Ground truth transition markers

        Args:
            trajectory_plots: List of trajectory plot data from trainer
            epoch: Current epoch number
            data_type: 'train' or 'val'
        """
        import wandb

        if len(trajectory_plots) == 0:
            print(f"No {data_type} trajectory plots available")
            return

        # Select a subset of trajectories to plot (max 6)
        num_plots = min(6, len(trajectory_plots))
        selected_indices = random.sample(range(len(trajectory_plots)), num_plots)

        print(f"Logging {num_plots} {data_type} trajectory timeline plots to wandb "
              f"for epoch {epoch}...")

        # Create subplot grid (2 columns)
        num_rows = (num_plots + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(16, 4 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for plot_idx, traj_idx in enumerate(selected_indices):
            traj_plot_data = trajectory_plots[traj_idx]
            ax = axes[plot_idx]
            self._create_trajectory_timeline_plot(traj_plot_data, ax)

        # Hide unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')

        plt.tight_layout()
        wandb.log({f"epoch_{epoch:03d}_{data_type}_trajectory_timelines": wandb.Image(fig)})
        plt.close(fig)

        print(f"Logged {num_plots} {data_type} trajectory timeline plots to wandb")

    def _create_trajectory_timeline_plot(
        self,
        plot_data: Dict[str, Any],
        ax: plt.Axes
    ) -> None:
        """Create a single trajectory timeline plot with detected peaks.

        Args:
            plot_data: Dictionary with keys:
                - 'task_id': Task ID
                - 'episode_id': Episode ID
                - 'timeline_scores': Prediction scores for each frame
                - 'ground_truth_transitions': List of ground truth transition frame indices
                - 'predicted_peaks': Array of detected peak frame indices
                - 'peak_scores': Scores at detected peaks
                - 'threshold': Classification threshold value
                - 'matched_predictions': List of matched predicted peaks
                - 'unmatched_predictions': List of unmatched predicted peaks
            ax: Matplotlib axes to plot on
        """
        task_id = plot_data['task_id']
        episode_id = plot_data['episode_id']
        timeline_scores = plot_data['timeline_scores']
        ground_truth_transitions = plot_data['ground_truth_transitions']
        threshold = plot_data['threshold']
        predicted_peaks = plot_data.get('predicted_peaks', np.array([]))
        peak_scores = plot_data.get('peak_scores', np.array([]))
        matched_predictions = plot_data.get('matched_predictions', [])
        unmatched_predictions = plot_data.get('unmatched_predictions', [])

        # Create time axis
        time_axis = np.arange(len(timeline_scores))

        # Plot prediction scores
        ax.plot(time_axis, timeline_scores, 'b-', alpha=0.7, linewidth=1.5,
               label='Prediction Scores')

        # Plot threshold line
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=f'Threshold ({threshold:.3f})')

        # Plot ground truth transitions as vertical lines
        for i, trans_idx in enumerate(ground_truth_transitions):
            if 0 <= trans_idx < len(timeline_scores):
                ax.axvline(x=trans_idx, color='red', linestyle='-',
                          linewidth=2, alpha=0.8,
                          label='GT Transition' if i == 0 else '')

        # Plot detected peaks
        if len(predicted_peaks) > 0:
            # Separate matched and unmatched peaks for different visualization
            matched_mask = np.isin(predicted_peaks, matched_predictions)
            unmatched_mask = np.isin(predicted_peaks, unmatched_predictions)

            # Plot matched peaks (true positives) in green
            if np.any(matched_mask):
                matched_peaks = predicted_peaks[matched_mask]
                matched_scores = peak_scores[matched_mask]
                ax.scatter(matched_peaks, matched_scores, color='green', s=100,
                          marker='o', zorder=5, edgecolors='darkgreen', linewidths=2,
                          label=f'Detected Peaks (TP: {len(matched_peaks)})')

            # Plot unmatched peaks (false positives) in purple
            if np.any(unmatched_mask):
                unmatched_peaks_arr = predicted_peaks[unmatched_mask]
                unmatched_scores = peak_scores[unmatched_mask]
                ax.scatter(unmatched_peaks_arr, unmatched_scores, color='purple', s=80,
                          marker='x', zorder=5, linewidths=2,
                          label=f'False Positives: {len(unmatched_peaks_arr)}')

        # Calculate metrics for title
        num_gt = len(ground_truth_transitions)
        num_matched = len(matched_predictions)
        num_unmatched = len(unmatched_predictions)
        num_fn = num_gt - num_matched

        # Formatting
        title = f'Task {task_id}, Episode {episode_id} - Full Trajectory\n'
        title += f'GT: {num_gt}, Detected: {num_matched + num_unmatched} (TP: {num_matched}, FP: {num_unmatched}, FN: {num_fn})'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Boundary Probability')

        # Auto-adjust y-limits based on data
        max_score = np.max(timeline_scores) if len(timeline_scores) > 0 else 0.1
        ax.set_ylim(-0.01, max(max_score * 1.2, 0.1))
        ax.grid(True, alpha=0.3)

        # Legend (avoid duplicate labels)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)


class CheckpointCallback(Callback):
    """Model checkpointing callback.

    Saves model checkpoints during training. Can save:
    - Best model based on a metric
    - Periodic checkpoints every N epochs
    - Final model at end of training

    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor for best model ('val_loss', 'val_f1', etc.)
        mode: 'min' or 'max' - whether lower or higher is better
        save_best_only: Only save when metric improves
        save_every: Save checkpoint every N epochs (0 to disable)
        verbose: Print when saving checkpoints

    Example:
        >>> callback = CheckpointCallback(
        ...     save_dir='./checkpoints',
        ...     monitor='val_loss',
        ...     mode='min',
        ...     save_every=10
        ... )
    """

    def __init__(
        self,
        save_dir: str = './checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_every: int = 10,
        verbose: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.verbose = verbose

        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Save checkpoints based on criteria."""
        # Get current metric value
        current_metric = logs.get(self.monitor)

        # Check if this is the best model
        is_best = False
        if current_metric is not None:
            if self.mode == 'min':
                is_best = current_metric < self.best_metric
            else:
                is_best = current_metric > self.best_metric

            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch

        # Save best model
        if is_best and (self.save_best_only or current_metric is not None):
            self._save_checkpoint(
                trainer,
                epoch,
                logs,
                filename='best_model.pt'
            )
            if self.verbose:
                print(f"\nNew best model saved with {self.monitor}: {current_metric:.4f}")

        # Save periodic checkpoint
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            self._save_checkpoint(
                trainer,
                epoch,
                logs,
                filename=f'checkpoint_epoch_{epoch+1}.pt'
            )
            if self.verbose:
                print(f"\nCheckpoint saved at epoch {epoch+1}")

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Save final model."""
        self._save_checkpoint(
            trainer,
            logs.get('epoch', -1),
            logs,
            filename='final_model.pt'
        )
        if self.verbose:
            print(f"\nFinal model saved. Best model was at epoch {self.best_epoch+1} "
                  f"with {self.monitor}: {self.best_metric:.4f}")

    def _save_checkpoint(
        self,
        trainer: 'Trainer',
        epoch: int,
        logs: Dict[str, Any],
        filename: str
    ) -> None:
        """Save a checkpoint file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'logs': logs
        }

        # Add scheduler state if available
        if trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)


class EarlyStoppingCallback(Callback):
    """Early stopping callback.

    Stops training when a monitored metric has stopped improving.

    Args:
        monitor: Metric to monitor ('val_loss', 'val_f1', etc.)
        patience: Number of epochs with no improvement to wait
        mode: 'min' or 'max' - whether lower or higher is better
        min_delta: Minimum change to qualify as an improvement
        verbose: Print when stopping early

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     mode='min'
        ... )
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        # Track state
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = -1

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Check if training should stop."""
        current_metric = logs.get(self.monitor)

        if current_metric is None:
            warnings.warn(f"Early stopping metric '{self.monitor}' not found in logs")
            return

        # Check if metric improved
        if self.mode == 'min':
            improved = current_metric < (self.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True

                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"No improvement in {self.monitor} for {self.patience} epochs")
                    print(f"Best {self.monitor}: {self.best_metric:.4f}")

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Print early stopping summary."""
        if self.verbose and self.stopped_epoch >= 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch+1}")


class TensorBoardCallback(Callback):
    """TensorBoard logging callback.

    Logs metrics and model graph to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs
        log_graph: Whether to log model graph
        log_histograms: Whether to log weight/gradient histograms
        histogram_freq: Log histograms every N epochs

    Example:
        >>> callback = TensorBoardCallback(
        ...     log_dir='./logs',
        ...     log_histograms=True
        ... )
    """

    def __init__(
        self,
        log_dir: str = './logs',
        log_graph: bool = True,
        log_histograms: bool = False,
        histogram_freq: int = 1
    ):
        self.log_dir = log_dir
        self.log_graph = log_graph
        self.log_histograms = log_histograms
        self.histogram_freq = histogram_freq
        self.writer = None

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir)

            # Log model graph
            if self.log_graph and hasattr(trainer, 'sample_input'):
                self.writer.add_graph(trainer.model, trainer.sample_input)

        except ImportError:
            warnings.warn("tensorboard not installed, skipping TensorBoard logging")

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        # Log scalar metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # Log histograms
        if self.log_histograms and (epoch + 1) % self.histogram_freq == 0:
            for name, param in trainer.model.named_parameters():
                self.writer.add_histogram(f'parameters/{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback.

    Wrapper for PyTorch learning rate schedulers that integrates with callbacks.

    Args:
        scheduler: PyTorch scheduler instance
        monitor: Metric to monitor (for ReduceLROnPlateau)

    Example:
        >>> from torch.optim.lr_scheduler import CosineAnnealingLR
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> callback = LearningRateSchedulerCallback(scheduler)
    """

    def __init__(
        self,
        scheduler: Any,
        monitor: Optional[str] = None
    ):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Step the scheduler."""
        # ReduceLROnPlateau needs a metric value
        if self.monitor is not None and hasattr(self.scheduler, 'step'):
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        elif hasattr(self.scheduler, 'step'):
            self.scheduler.step()


class EmbeddingVisualizationCallback(Callback):
    """Trajectory embedding visualization callback for training.

    Visualizes trajectory embeddings showing temporal progression through
    embedding space. Plots windows from selected trajectories.

    Can visualize different layers of the model and color by different attributes.

    Args:
        method: Visualization method ('pca' or 'tsne')
        num_trajectories: Number of trajectories to visualize per split
        max_windows_per_traj: Maximum windows per trajectory (None = all)
        layer: Which layer to visualize ('input', 'token_embed', 'proprio_encoded',
               'dino_encoded', 'combined', 'hidden1', 'hidden2', 'hidden3')
        color_by: How to color points ('timestep' or 'action')
        log_on_save: Log plot when saving best model (default)
        log_on_first: Also log on first epoch
        log_on_last: Also log on last epoch

    Example:
        >>> # Visualize token embeddings colored by action
        >>> callback = EmbeddingVisualizationCallback(
        ...     method='pca',
        ...     layer='token_embed',
        ...     color_by='action',
        ...     num_trajectories=3,
        ...     log_on_save=True
        ... )
    """

    def __init__(
        self,
        method: str = 'pca',
        num_trajectories: int = 3,
        max_windows_per_traj: Optional[int] = None,
        layer: str = 'input',
        color_by: str = 'timestep',
        log_on_save: bool = True,
        log_on_first: bool = True,
        log_on_last: bool = True
    ):
        self.method = method
        self.num_trajectories = num_trajectories
        self.max_windows_per_traj = max_windows_per_traj
        self.layer = layer
        self.color_by = color_by
        self.log_on_save = log_on_save
        self.log_on_first = log_on_first
        self.log_on_last = log_on_last

        # Track state
        self.train_trajectories = None
        self.val_trajectories = None
        self.train_dataset = None
        self.val_dataset = None
        self.has_logged_first = False
        self.best_val_loss = float('inf')

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """Store references to trajectories and datasets."""
        # Get datasets
        if hasattr(trainer, 'train_loader') and trainer.train_loader:
            self.train_dataset = trainer.train_loader.dataset
            # Try to get original trajectories from dataset
            if hasattr(self.train_dataset, 'trajectories'):
                self.train_trajectories = self.train_dataset.trajectories

        if hasattr(trainer, 'val_loader') and trainer.val_loader:
            self.val_dataset = trainer.val_loader.dataset
            if hasattr(self.val_dataset, 'trajectories'):
                self.val_trajectories = self.val_dataset.trajectories

        # Log on first epoch if requested
        if self.log_on_first:
            self._log_embeddings(trainer, epoch=0)
            self.has_logged_first = True

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Log embeddings when best model is saved."""
        # Check if this is a new best model
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

            if self.log_on_save:
                self._log_embeddings(trainer, epoch)

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Log embeddings at end of training if requested."""
        if self.log_on_last:
            epoch = logs.get('epoch', -1)
            self._log_embeddings(trainer, epoch, final=True)

    def _log_embeddings(self, trainer: 'Trainer', epoch: int, final: bool = False) -> None:
        """Generate and log embedding plots."""
        # Check if wandb is available
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        # Import visualizer
        from utils.trajectory_embedding_visualizer import create_embedding_plot_for_logging

        # Check if we have data
        if not (self.train_trajectories and self.train_dataset):
            warnings.warn("No trajectory data available for embedding visualization")
            return

        print(f"\nGenerating trajectory embedding visualization (epoch {epoch+1})...")
        print(f"  Layer: {self.layer}, Color by: {self.color_by}")

        try:
            # Create plots
            figures = create_embedding_plot_for_logging(
                self.train_trajectories,
                self.val_trajectories,
                self.train_dataset,
                self.val_dataset,
                method=self.method,
                num_trajectories_per_split=self.num_trajectories,
                max_windows_per_traj=self.max_windows_per_traj,
                model=trainer.model,
                layer=self.layer,
                color_by=self.color_by,
                device=trainer.device
            )

            # Log to wandb
            log_dict = {}
            layer_suffix = f'_{self.layer}' if self.layer != 'input' else ''
            color_suffix = f'_{self.color_by}' if self.color_by != 'timestep' else ''

            if 'train' in figures:
                log_dict[f'embeddings/train_trajectory{layer_suffix}{color_suffix}'] = wandb.Image(figures['train'])
                plt.close(figures['train'])
                print(f"  ✓ Logged training trajectory embedding plot")

            if 'val' in figures:
                log_dict[f'embeddings/val_trajectory{layer_suffix}{color_suffix}'] = wandb.Image(figures['val'])
                plt.close(figures['val'])
                print(f"  ✓ Logged validation trajectory embedding plot")

            if log_dict:
                log_dict['epoch'] = epoch
                wandb.log(log_dict)

        except Exception as e:
            warnings.warn(f"Failed to create embedding plots: {e}")
