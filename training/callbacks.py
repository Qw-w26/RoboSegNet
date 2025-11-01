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

            # Log multi-task configuration if available
            if 'data' in self.config:
                data_config = self.config['data']
                if 'train_task_ids' in data_config and data_config['train_task_ids']:
                    train_tasks = data_config['train_task_ids']
                    eval_tasks = data_config.get('eval_task_ids', train_tasks)

                    wandb.run.summary['train_tasks'] = train_tasks
                    wandb.run.summary['eval_tasks'] = eval_tasks
                    wandb.run.summary['num_train_tasks'] = len(train_tasks)
                    wandb.run.summary['num_eval_tasks'] = len(eval_tasks)
                    wandb.run.summary['is_cross_task_eval'] = (train_tasks != eval_tasks)

        except ImportError:
            warnings.warn("wandb not installed, skipping W&B logging")

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        """Log metrics and plots at epoch end."""
        if self.run is None:
            return

        import wandb

        # Organize metrics into categories for better WandB visualization
        metrics = {'epoch': epoch}

        # Core losses
        loss_metrics = {}
        if 'train_loss' in logs:
            loss_metrics['train/loss'] = logs['train_loss']
        if 'val_loss' in logs:
            loss_metrics['val/loss'] = logs['val_loss']

        # Learning rate
        if 'learning_rate' in logs:
            metrics['learning_rate'] = logs['learning_rate']

        # Separate tolerance-based metrics from regular metrics
        eval_metrics = {}
        task_metrics = {}
        tolerance_metrics = {
            'train': {},
            'val': {},
            'train_per_task': {},
            'val_per_task': {},
            'train_in_dist': {},
            'val_in_dist': {},
            'train_out_dist': {},
            'val_out_dist': {}
        }

        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue

            # Skip already logged metrics
            if key in ['epoch', 'train_loss', 'val_loss', 'learning_rate']:
                continue

            # Check if this is a tolerance-based metric (contains @)
            if '@' in key:
                # Parse different types of tolerance metrics
                if key.startswith('train_task'):
                    # Per-task training metric: train_task327_precision_@1s
                    parts = key[6:].split('_', 1)  # Remove 'train_' and split on first _
                    task_id = parts[0].replace('task', '')
                    rest = parts[1] if len(parts) > 1 else ''
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if task_id not in tolerance_metrics['train_per_task']:
                        tolerance_metrics['train_per_task'][task_id] = {}
                    if metric_name not in tolerance_metrics['train_per_task'][task_id]:
                        tolerance_metrics['train_per_task'][task_id][metric_name] = {}
                    tolerance_metrics['train_per_task'][task_id][metric_name][tol] = value
                    eval_metrics[f'train/task{task_id}_{metric_name}_@{tol}'] = value

                elif key.startswith('val_task'):
                    # Per-task validation metric: val_task327_precision_@1s
                    parts = key[4:].split('_', 1)  # Remove 'val_' and split on first _
                    task_id = parts[0].replace('task', '')
                    rest = parts[1] if len(parts) > 1 else ''
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if task_id not in tolerance_metrics['val_per_task']:
                        tolerance_metrics['val_per_task'][task_id] = {}
                    if metric_name not in tolerance_metrics['val_per_task'][task_id]:
                        tolerance_metrics['val_per_task'][task_id][metric_name] = {}
                    tolerance_metrics['val_per_task'][task_id][metric_name][tol] = value
                    eval_metrics[f'val/task{task_id}_{metric_name}_@{tol}'] = value

                elif key.startswith('train_in_dist_'):
                    # In-distribution training metric: train_in_dist_precision_@1s
                    rest = key[15:]  # Remove 'train_in_dist_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['train_in_dist']:
                        tolerance_metrics['train_in_dist'][metric_name] = {}
                    tolerance_metrics['train_in_dist'][metric_name][tol] = value
                    eval_metrics[f'train/in_dist_{metric_name}_@{tol}'] = value

                elif key.startswith('train_out_dist_'):
                    # Out-of-distribution training metric: train_out_dist_precision_@1s
                    rest = key[16:]  # Remove 'train_out_dist_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['train_out_dist']:
                        tolerance_metrics['train_out_dist'][metric_name] = {}
                    tolerance_metrics['train_out_dist'][metric_name][tol] = value
                    eval_metrics[f'train/out_dist_{metric_name}_@{tol}'] = value

                elif key.startswith('val_in_dist_'):
                    # In-distribution validation metric: val_in_dist_precision_@1s
                    rest = key[13:]  # Remove 'val_in_dist_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['val_in_dist']:
                        tolerance_metrics['val_in_dist'][metric_name] = {}
                    tolerance_metrics['val_in_dist'][metric_name][tol] = value
                    eval_metrics[f'val/in_dist_{metric_name}_@{tol}'] = value

                elif key.startswith('val_out_dist_'):
                    # Out-of-distribution validation metric: val_out_dist_precision_@1s
                    rest = key[14:]  # Remove 'val_out_dist_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['val_out_dist']:
                        tolerance_metrics['val_out_dist'][metric_name] = {}
                    tolerance_metrics['val_out_dist'][metric_name][tol] = value
                    eval_metrics[f'val/out_dist_{metric_name}_@{tol}'] = value

                elif key.startswith('train_'):
                    # Overall training metric: train_precision_@1s
                    rest = key[6:]  # Remove 'train_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['train']:
                        tolerance_metrics['train'][metric_name] = {}
                    tolerance_metrics['train'][metric_name][tol] = value
                    eval_metrics[f'train/{metric_name}_@{tol}'] = value

                elif key.startswith('val_'):
                    # Overall validation metric: val_precision_@1s
                    rest = key[4:]  # Remove 'val_'
                    metric_name, tol = rest.split('@', 1)
                    metric_name = metric_name.rstrip('_')

                    if metric_name not in tolerance_metrics['val']:
                        tolerance_metrics['val'][metric_name] = {}
                    tolerance_metrics['val'][metric_name][tol] = value
                    eval_metrics[f'val/{metric_name}_@{tol}'] = value
            else:
                # Regular metrics (non-tolerance)
                if 'train_' in key:
                    eval_metrics[f'train/{key.replace("train_", "")}'] = value
                elif 'val_' in key:
                    eval_metrics[f'val/{key.replace("val_", "")}'] = value
                elif 'task_' in key:
                    # Task-specific metrics for multi-task training
                    task_metrics[key] = value
                else:
                    metrics[key] = value

        # Log all metrics in organized groups
        all_metrics = {**metrics, **loss_metrics, **eval_metrics, **task_metrics}
        wandb.log(all_metrics)

        # Create and log merged tolerance plots
        self._log_merged_tolerance_plots(tolerance_metrics, epoch)

        # Create and log summary table every epoch for easy viewing
        if epoch % 5 == 0 or epoch == 0:  # Log every 5 epochs to reduce clutter
            self._log_metrics_summary_table(logs, epoch)

        # Log prediction plots if it's time - MERGED VERSION
        if (epoch + 1) % self.log_plots_every == 0:
            # Merge train and val prediction plots into single figure
            train_data = logs.get('train_trajectory_data', [])
            val_data = logs.get('val_trajectory_data', [])

            if train_data or val_data:
                self._log_merged_prediction_plots(
                    train_data=train_data,
                    val_data=val_data,
                    epoch=epoch
                )

        # Log trajectory timeline plots - MERGED VERSION
        train_plots = logs.get('train_trajectory_plots', [])
        val_plots = logs.get('val_trajectory_plots', [])

        if train_plots or val_plots:
            self._log_merged_trajectory_timeline_plots(
                train_plots=train_plots,
                val_plots=val_plots,
                epoch=epoch
            )

    def on_train_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        """Finish W&B run."""
        if self.run is not None:
            import wandb
            wandb.finish()

    def _log_merged_tolerance_plots(
        self,
        tolerance_metrics: Dict[str, Dict[str, Dict[str, float]]],
        epoch: int
    ) -> None:
        """Create and log merged plots showing all tolerances in one figure.

        Creates separate plots for:
        - Overall metrics (all tasks)
        - In-distribution metrics (training set tasks)
        - Out-of-distribution metrics (non-training set tasks)
        - Per-task metrics

        Args:
            tolerance_metrics: Nested dict with structure:
                {'train': {'precision': {'1s': 0.8, '2s': 0.9}, ...},
                 'val': {...},
                 'train_per_task': {'327': {'precision': {'1s': ...}}, ...},
                 'train_in_dist': {...},
                 'train_out_dist': {...}}
            epoch: Current epoch number
        """
        import wandb

        # Track history across epochs for line plots
        if not hasattr(self, '_tolerance_history'):
            self._tolerance_history = {
                'train': {}, 'val': {},
                'train_per_task': {}, 'val_per_task': {},
                'train_in_dist': {}, 'val_in_dist': {},
                'train_out_dist': {}, 'val_out_dist': {}
            }

        # Update history with current epoch's data
        # Overall metrics
        for split in ['train', 'val']:
            if split in tolerance_metrics:
                for metric_name, tol_values in tolerance_metrics[split].items():
                    if metric_name not in self._tolerance_history[split]:
                        self._tolerance_history[split][metric_name] = {}
                    for tol, value in tol_values.items():
                        if tol not in self._tolerance_history[split][metric_name]:
                            self._tolerance_history[split][metric_name][tol] = []
                        self._tolerance_history[split][metric_name][tol].append(value)

        # In-dist and out-dist metrics
        for group in ['in_dist', 'out_dist']:
            for split in ['train', 'val']:
                key = f'{split}_{group}'
                if key in tolerance_metrics:
                    for metric_name, tol_values in tolerance_metrics[key].items():
                        if metric_name not in self._tolerance_history[key]:
                            self._tolerance_history[key][metric_name] = {}
                        for tol, value in tol_values.items():
                            if tol not in self._tolerance_history[key][metric_name]:
                                self._tolerance_history[key][metric_name][tol] = []
                            self._tolerance_history[key][metric_name][tol].append(value)

        # Per-task metrics
        for split in ['train', 'val']:
            key = f'{split}_per_task'
            if key in tolerance_metrics:
                for task_id, task_data in tolerance_metrics[key].items():
                    if task_id not in self._tolerance_history[key]:
                        self._tolerance_history[key][task_id] = {}
                    for metric_name, tol_values in task_data.items():
                        if metric_name not in self._tolerance_history[key][task_id]:
                            self._tolerance_history[key][task_id][metric_name] = {}
                        for tol, value in tol_values.items():
                            if tol not in self._tolerance_history[key][task_id][metric_name]:
                                self._tolerance_history[key][task_id][metric_name][tol] = []
                            self._tolerance_history[key][task_id][metric_name][tol].append(value)

        # Create merged plots for key metrics
        key_metrics = ['precision', 'recall', 'f1_score', 'avg_iou']

        # 1. Overall metrics (all tasks)
        self._create_and_log_tolerance_plots(
            key_metrics, 'train', 'val', 'overall',
            'Overall Metrics (All Tasks)', wandb, epoch
        )

        # 2. In-distribution metrics (training set tasks)
        self._create_and_log_tolerance_plots(
            key_metrics, 'train_in_dist', 'val_in_dist', 'in_distribution',
            'In-Distribution (Training Set)', wandb, epoch
        )

        # 3. Out-of-distribution metrics (non-training set tasks)
        self._create_and_log_tolerance_plots(
            key_metrics, 'train_out_dist', 'val_out_dist', 'out_distribution',
            'Out-of-Distribution (Non-Training Set)', wandb, epoch
        )

        # 4. Per-task metrics
        self._create_per_task_tolerance_plots(key_metrics, wandb, epoch)

        print(f"Logged merged tolerance plots for epoch {epoch}")

    def _create_and_log_tolerance_plots(
        self,
        key_metrics: List[str],
        train_key: str,
        val_key: str,
        log_prefix: str,
        title_prefix: str,
        wandb,
        epoch: int
    ) -> None:
        """Helper to create and log tolerance plots for a specific group."""
        for metric_name in key_metrics:
            # Check if this metric exists in either split
            has_train = (train_key in self._tolerance_history and
                        metric_name in self._tolerance_history[train_key])
            has_val = (val_key in self._tolerance_history and
                      metric_name in self._tolerance_history[val_key])

            if not has_train and not has_val:
                continue

            # Create figure with subplots (one for train, one for val)
            num_plots = sum([has_train, has_val])
            fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

            if num_plots == 1:
                axes = [axes]

            plot_idx = 0

            # Plot training metrics
            if has_train:
                ax = axes[plot_idx]
                self._plot_tolerance_lines(
                    ax,
                    self._tolerance_history[train_key][metric_name],
                    title=f'{title_prefix} - Training {metric_name.replace("_", " ").title()}',
                    metric_name=metric_name
                )
                plot_idx += 1

            # Plot validation metrics
            if has_val:
                ax = axes[plot_idx]
                self._plot_tolerance_lines(
                    ax,
                    self._tolerance_history[val_key][metric_name],
                    title=f'{title_prefix} - Validation {metric_name.replace("_", " ").title()}',
                    metric_name=metric_name
                )

            plt.tight_layout()
            wandb.log({f'metrics_by_tolerance/{log_prefix}/{metric_name}': wandb.Image(fig)})
            plt.close(fig)

    def _create_per_task_tolerance_plots(
        self,
        key_metrics: List[str],
        wandb,
        epoch: int
    ) -> None:
        """Create separate tolerance plots for each task."""
        # Get all unique task IDs from both train and val
        all_task_ids = set()
        if 'train_per_task' in self._tolerance_history:
            all_task_ids.update(self._tolerance_history['train_per_task'].keys())
        if 'val_per_task' in self._tolerance_history:
            all_task_ids.update(self._tolerance_history['val_per_task'].keys())

        # Create plots for each task
        for task_id in sorted(all_task_ids):
            for metric_name in key_metrics:
                has_train = (task_id in self._tolerance_history.get('train_per_task', {}) and
                           metric_name in self._tolerance_history['train_per_task'][task_id])
                has_val = (task_id in self._tolerance_history.get('val_per_task', {}) and
                          metric_name in self._tolerance_history['val_per_task'][task_id])

                if not has_train and not has_val:
                    continue

                # Create figure with subplots
                num_plots = sum([has_train, has_val])
                fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))

                if num_plots == 1:
                    axes = [axes]

                plot_idx = 0

                # Plot training metrics for this task
                if has_train:
                    ax = axes[plot_idx]
                    self._plot_tolerance_lines(
                        ax,
                        self._tolerance_history['train_per_task'][task_id][metric_name],
                        title=f'Task {task_id} - Training {metric_name.replace("_", " ").title()}',
                        metric_name=metric_name
                    )
                    plot_idx += 1

                # Plot validation metrics for this task
                if has_val:
                    ax = axes[plot_idx]
                    self._plot_tolerance_lines(
                        ax,
                        self._tolerance_history['val_per_task'][task_id][metric_name],
                        title=f'Task {task_id} - Validation {metric_name.replace("_", " ").title()}',
                        metric_name=metric_name
                    )

                plt.tight_layout()
                wandb.log({f'metrics_by_tolerance/per_task/task{task_id}/{metric_name}': wandb.Image(fig)})
                plt.close(fig)

    def _plot_tolerance_lines(
        self,
        ax: plt.Axes,
        tolerance_data: Dict[str, List[float]],
        title: str,
        metric_name: str
    ) -> None:
        """Plot lines for each tolerance on the same axes.

        Args:
            ax: Matplotlib axes
            tolerance_data: Dict mapping tolerance (e.g., '1s') to list of values across epochs
            title: Plot title
            metric_name: Name of the metric being plotted
        """
        # Sort tolerances for consistent ordering
        sorted_tolerances = sorted(tolerance_data.keys(), key=lambda x: float(x.rstrip('s')))

        # Color map for different tolerances
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_tolerances)))

        for idx, tol in enumerate(sorted_tolerances):
            values = tolerance_data[tol]
            epochs = list(range(len(values)))

            ax.plot(
                epochs,
                values,
                marker='o',
                linewidth=2,
                markersize=6,
                label=f'@{tol}',
                color=colors[idx],
                alpha=0.8
            )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='Tolerance', fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set y-axis limits based on metric type
        if metric_name in ['precision', 'recall', 'f1_score']:
            ax.set_ylim(-0.05, 1.05)
        elif metric_name == 'avg_iou':
            ax.set_ylim(-0.05, 1.05)

    def _log_metrics_summary_table(self, logs: Dict[str, Any], epoch: int) -> None:
        """Create and log a summary table of key metrics."""
        import wandb

        # Extract key metrics for different tolerances
        summary_data = []

        # Get tolerance values if available
        tolerances = []
        for key in logs.keys():
            if 'tol_' in key and 'val_' in key:
                tol = key.split('tol_')[1].split('_')[0]
                if tol not in tolerances:
                    tolerances.append(tol)

        tolerances = sorted(tolerances, key=float)

        # Create row for each tolerance
        for tol in tolerances:
            row = {
                'Tolerance (s)': tol,
                'Train Precision': logs.get(f'train_precision_tol_{tol}', 0.0),
                'Train Recall': logs.get(f'train_recall_tol_{tol}', 0.0),
                'Train F1': logs.get(f'train_f1_tol_{tol}', 0.0),
                'Val Precision': logs.get(f'val_precision_tol_{tol}', 0.0),
                'Val Recall': logs.get(f'val_recall_tol_{tol}', 0.0),
                'Val F1': logs.get(f'val_f1_tol_{tol}', 0.0),
            }
            summary_data.append(row)

        # Add overall metrics row
        if summary_data:
            overall_row = {
                'Tolerance (s)': 'Overall',
                'Train Precision': '-',
                'Train Recall': '-',
                'Train F1': '-',
                'Val Precision': '-',
                'Val Recall': '-',
                'Val F1': '-',
            }

            # Add loss info
            if 'train_loss' in logs:
                overall_row['Train Loss'] = f"{logs['train_loss']:.4f}"
            if 'val_loss' in logs:
                overall_row['Val Loss'] = f"{logs['val_loss']:.4f}"

            summary_data.append(overall_row)

            # Create table
            table = wandb.Table(
                columns=list(summary_data[0].keys()),
                data=[list(row.values()) for row in summary_data]
            )

            wandb.log({f"metrics/summary_epoch_{epoch:03d}": table})

    def _log_merged_prediction_plots(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        epoch: int
    ) -> None:
        """Log merged train/val prediction plots to W&B in a single figure.

        Creates a combined figure with train plots on top and val plots on bottom.
        """
        import wandb

        if len(train_data) == 0 and len(val_data) == 0:
            print(f"No trajectory data available for plotting")
            return

        num_plots_per_split = self.num_plots ** 2

        # Select train data
        train_selected = []
        if len(train_data) > 0:
            train_selected = self._select_interesting_datapoints(
                train_data, num_plots_per_split, 'train'
            )

        # Select val data
        val_selected = []
        if len(val_data) > 0:
            val_selected = self._select_interesting_datapoints(
                val_data, num_plots_per_split, 'val'
            )

        # Create combined figure
        grid_size = self.num_plots
        total_rows = grid_size * (2 if train_selected and val_selected else 1)

        fig = plt.figure(figsize=(4 * grid_size, 3 * total_rows))

        # Add super titles for each section
        plot_idx = 0

        # Plot training data
        if train_selected:
            for i, idx in enumerate(train_selected):
                ax = plt.subplot(total_rows, grid_size, i + 1)
                traj_data = train_data[idx]
                self._create_prediction_plot(traj_data, ax)
                if i == 0:
                    ax.set_title(f'TRAIN - Task {traj_data["task_id"]}, Ep {traj_data["episode_id"]}',
                               fontweight='bold', fontsize=10)
                plot_idx = i + 1

        # Plot validation data
        if val_selected:
            start_idx = len(train_selected) if train_selected else 0
            for i, idx in enumerate(val_selected):
                ax = plt.subplot(total_rows, grid_size, start_idx + i + 1)
                traj_data = val_data[idx]
                self._create_prediction_plot(traj_data, ax)
                if i == 0 and not train_selected:
                    ax.set_title(f'VAL - Task {traj_data["task_id"]}, Ep {traj_data["episode_id"]}',
                               fontweight='bold', fontsize=10, color='darkgreen')
                elif i == 0:
                    # Add section divider
                    ax.text(0.5, 1.15, 'VALIDATION', transform=ax.transAxes,
                           ha='center', fontweight='bold', fontsize=12, color='darkgreen')

        plt.tight_layout()
        wandb.log({f"predictions/epoch_{epoch:03d}_train_val": wandb.Image(fig)})
        plt.close(fig)

        print(f"Logged {len(train_selected)} train + {len(val_selected)} val "
              f"prediction plots to wandb for epoch {epoch}")

    def _select_interesting_datapoints(
        self,
        trajectory_data: List[Dict[str, Any]],
        num_plots: int,
        data_type: str
    ) -> List[int]:
        """Select interesting datapoints for plotting."""
        if len(trajectory_data) == 0:
            return []

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

        print(f"  {data_type}: searched {search_size} datapoints, "
              f"found {len(interesting_indices)} interesting, selected {len(selected_indices)}")

        return selected_indices

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

    def _log_merged_trajectory_timeline_plots(
        self,
        train_plots: List[Dict[str, Any]],
        val_plots: List[Dict[str, Any]],
        epoch: int
    ) -> None:
        """Log merged train/val trajectory timeline plots to W&B in a single figure.

        Creates a combined figure with train plots on left column and val plots on right.
        """
        import wandb

        if len(train_plots) == 0 and len(val_plots) == 0:
            print(f"No trajectory timeline plots available")
            return

        # Select subsets
        max_plots_per_split = 4
        num_train = min(max_plots_per_split, len(train_plots))
        num_val = min(max_plots_per_split, len(val_plots))

        train_indices = random.sample(range(len(train_plots)), num_train) if num_train > 0 else []
        val_indices = random.sample(range(len(val_plots)), num_val) if num_val > 0 else []

        # Determine grid layout
        max_rows = max(num_train, num_val)
        num_cols = 2 if (num_train > 0 and num_val > 0) else 1

        fig, axes = plt.subplots(max_rows, num_cols, figsize=(10 * num_cols, 4 * max_rows))

        # Handle single plot case
        if max_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif max_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Add column headers
        if num_cols == 2:
            fig.text(0.25, 0.98, 'TRAINING', ha='center', fontsize=14,
                    fontweight='bold', color='blue')
            fig.text(0.75, 0.98, 'VALIDATION', ha='center', fontsize=14,
                    fontweight='bold', color='green')
        elif num_train > 0:
            fig.text(0.5, 0.98, 'TRAINING', ha='center', fontsize=14,
                    fontweight='bold', color='blue')
        else:
            fig.text(0.5, 0.98, 'VALIDATION', ha='center', fontsize=14,
                    fontweight='bold', color='green')

        # Plot training trajectories (left column or only column)
        col_idx = 0
        for row_idx, traj_idx in enumerate(train_indices):
            ax = axes[row_idx, col_idx] if num_cols > 1 else axes[row_idx, 0]
            self._create_trajectory_timeline_plot(train_plots[traj_idx], ax)

        # Hide unused train slots
        for row_idx in range(num_train, max_rows):
            ax = axes[row_idx, col_idx] if num_cols > 1 else axes[row_idx, 0]
            ax.axis('off')

        # Plot validation trajectories (right column or only column)
        if num_val > 0:
            col_idx = 1 if num_cols > 1 else 0
            for row_idx, traj_idx in enumerate(val_indices):
                ax = axes[row_idx, col_idx]
                self._create_trajectory_timeline_plot(val_plots[traj_idx], ax)

            # Hide unused val slots
            if num_cols > 1:
                for row_idx in range(num_val, max_rows):
                    ax = axes[row_idx, col_idx]
                    ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        wandb.log({f"trajectories/epoch_{epoch:03d}_train_val_timelines": wandb.Image(fig)})
        plt.close(fig)

        print(f"Logged {num_train} train + {num_val} val trajectory timeline plots to wandb "
              f"for epoch {epoch}")

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
