"""Training infrastructure for boundary prediction models.

This module provides a Trainer class that handles the training loop with
callback support, automatic metric tracking, and progress visualization.

Example:
    >>> from training import Trainer, WandBCallback, CheckpointCallback
    >>> from models import create_model, ModelConfig
    >>>
    >>> model = create_model(ModelConfig())
    >>> callbacks = [
    ...     WandBCallback(project='boundary-prediction'),
    ...     CheckpointCallback(save_dir='./checkpoints')
    ... ]
    >>> trainer = Trainer(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     criterion=criterion,
    ...     callbacks=callbacks
    ... )
    >>> trainer.fit(train_loader, val_loader, num_epochs=100)
"""

from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np

from training.callbacks import Callback


class Trainer:
    """Training manager with callback support.

    Handles the training loop, validation, metric tracking, and integrates
    with callbacks for logging, checkpointing, and early stopping.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use ('cuda' or 'cpu', 'auto' for auto-detection)
        callbacks: List of callback instances
        scheduler: Optional learning rate scheduler
        gradient_clip_val: Optional gradient clipping value
        collect_trajectory_data_fn: Optional function to collect trajectory data for plotting

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ...     device='cuda',
        ...     callbacks=[WandBCallback(), CheckpointCallback()]
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'auto',
        callbacks: Optional[List[Callback]] = None,
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip_val: Optional[float] = None,
        collect_trajectory_data_fn: Optional[Callable] = None
    ):
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Core components
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.gradient_clip_val = gradient_clip_val
        self.collect_trajectory_data_fn = collect_trajectory_data_fn

        # Callbacks
        self.callbacks = callbacks or []

        # Training state
        self.stop_training = False
        self.current_epoch = 0

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        log_every: int = 10,
        collect_plots_every: int = 3
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            log_every: Log progress every N batches
            collect_plots_every: Collect trajectory data for plots every N epochs

        Returns:
            Dictionary of training history (losses, metrics)
        """
        # Store data loaders for callback access
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_accuracy': [],
            'train_f1': [],
            'val_accuracy': [],
            'val_f1': []
        }

        # Callback: on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Training loop
        for epoch in range(num_epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch

            # Callback: on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Determine if we should collect trajectory data this epoch
            should_collect_plots = (epoch + 1) % collect_plots_every == 0

            # Train
            train_results = self._train_epoch(
                train_loader,
                log_every=log_every,
                collect_trajectory_data=should_collect_plots
            )
            train_loss = train_results['loss']
            train_trajectory_data = train_results.get('trajectory_data', [])
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_results.get('accuracy', 0.0))
            history['train_f1'].append(train_results.get('f1_score', 0.0))

            # Validate
            val_loss = None
            val_trajectory_data = []
            val_accuracy = 0.0
            val_f1 = 0.0
            if val_loader is not None:
                val_results = self._validate_epoch(
                    val_loader,
                    collect_trajectory_data=should_collect_plots
                )
                val_loss = val_results['loss']
                val_trajectory_data = val_results.get('trajectory_data', [])
                val_accuracy = val_results.get('accuracy', 0.0)
                val_f1 = val_results.get('f1_score', 0.0)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                history['val_f1'].append(val_f1)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            # Prepare logs for callbacks
            logs = {
                'train_loss': train_loss,
                'train_accuracy': train_results.get('accuracy', 0.0),
                'train_f1': train_results.get('f1_score', 0.0),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'learning_rate': current_lr,
                'epoch': epoch
            }

            # Add trajectory data if collected
            if should_collect_plots:
                logs['train_trajectory_data'] = train_trajectory_data
                logs['val_trajectory_data'] = val_trajectory_data

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_results.get('accuracy', 0.0):.4f}, F1: {train_results.get('f1_score', 0.0):.4f}", end='')
            if val_loss is not None:
                print(f" | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}", end='')
            print(f" | LR: {current_lr:.6f}")

            # Callback: on_epoch_end
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)

            # Check if training should stop (early stopping)
            if self.stop_training:
                print("\nTraining stopped early")
                break

        # Callback: on_train_end
        final_logs = {
            'epoch': self.current_epoch,
            'train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None
        }
        for callback in self.callbacks:
            callback.on_train_end(self, final_logs)

        print("\nTraining completed!")
        return history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        log_every: int = 10,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Train for one epoch.

        Returns:
            Dictionary with 'loss', 'trajectory_data', and optional metrics
        """
        self.model.train()
        total_loss = 0.0
        trajectory_data = []

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Callback: on_batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Forward pass
            loss, batch_trajectory_data = self._train_step(batch, collect_trajectory_data)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()

            # Collect trajectory data if requested
            if collect_trajectory_data:
                trajectory_data.extend(batch_trajectory_data)

            # Update progress bar
            if (batch_idx + 1) % log_every == 0 or batch_idx == len(train_loader) - 1:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Callback: on_batch_end
            batch_logs = {'batch_loss': loss.item()}
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_logs)

        avg_loss = total_loss / len(train_loader)

        return {
            'loss': avg_loss,
            'trajectory_data': trajectory_data
        }

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Validate for one epoch.

        Returns:
            Dictionary with 'loss', 'trajectory_data', and optional metrics
        """
        # Callback: on_validation_begin
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        self.model.eval()
        total_loss = 0.0
        trajectory_data = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Forward pass only
                loss, batch_trajectory_data = self._validation_step(
                    batch,
                    collect_trajectory_data
                )

                total_loss += loss.item()

                # Collect trajectory data if requested
                if collect_trajectory_data:
                    trajectory_data.extend(batch_trajectory_data)

        avg_loss = total_loss / len(val_loader)

        # Callback: on_validation_end
        val_logs = {'val_loss': avg_loss}
        for callback in self.callbacks:
            callback.on_validation_end(self, val_logs)

        return {
            'loss': avg_loss,
            'trajectory_data': trajectory_data
        }

    def _train_step(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one training step.

        This method should be overridden for custom batch processing logic.
        Default implementation assumes batch has keys: 'input', 'target'.

        Returns:
            Tuple of (loss, trajectory_data)
        """
        # Default implementation - override in subclass for custom logic
        if hasattr(self, '_process_batch'):
            return self._process_batch(batch, training=True, collect_trajectory_data=collect_trajectory_data)
        else:
            raise NotImplementedError(
                "Either override _train_step or set a _process_batch method"
            )

    def _validation_step(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one validation step.

        Returns:
            Tuple of (loss, trajectory_data)
        """
        # Default implementation - override in subclass for custom logic
        if hasattr(self, '_process_batch'):
            return self._process_batch(batch, training=False, collect_trajectory_data=collect_trajectory_data)
        else:
            raise NotImplementedError(
                "Either override _validation_step or set a _process_batch method"
            )

    def set_batch_processor(self, process_fn: Callable) -> None:
        """Set a custom batch processing function.

        The function should have signature:
            process_fn(batch, training=True, collect_trajectory_data=False) -> (loss, trajectory_data)

        Example:
            >>> def process_batch(batch, training=True, collect_trajectory_data=False):
            ...     inputs = batch['input'].to(device)
            ...     targets = batch['target'].to(device)
            ...     outputs = model(inputs)
            ...     loss = criterion(outputs, targets)
            ...     trajectory_data = []  # Collect if needed
            ...     return loss, trajectory_data
            >>> trainer.set_batch_processor(process_batch)
        """
        self._process_batch = process_fn


class BoundaryTrainer(Trainer):
    """Specialized trainer for boundary prediction models.

    Handles the specific batch format and data processing for boundary
    prediction tasks with DINO features and proprioceptive tokens.

    Args:
        Same as Trainer, plus:
        prediction_stride: Stride for label prediction
        crop_labels_fn: Function to crop labels to prediction window

    Example:
        >>> trainer = BoundaryTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ...     prediction_stride=10
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'auto',
        callbacks: Optional[List[Callback]] = None,
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip_val: Optional[float] = None,
        prediction_stride: int = 10,
        prediction_window_size: int = 18,
        crop_labels_fn: Optional[Callable] = None,
        classification_threshold: float = 0.5
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            callbacks=callbacks,
            scheduler=scheduler,
            gradient_clip_val=gradient_clip_val
        )
        self.prediction_stride = prediction_stride
        self.prediction_window_size = prediction_window_size
        self.crop_labels_fn = crop_labels_fn or self._default_crop_labels
        self.classification_threshold = classification_threshold

        print(f"Classification threshold: {self.classification_threshold}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        log_every: int = 10,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Train for one epoch with metric computation.

        Returns:
            Dictionary with 'loss', 'accuracy', 'f1_score', 'trajectory_data'
        """
        self.model.train()
        total_loss = 0.0
        trajectory_data = []

        # Accumulate predictions and labels for metrics
        all_predictions = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Callback: on_batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Forward pass
            loss, batch_trajectory_data, predictions, labels = self._train_step_with_metrics(
                batch, collect_trajectory_data
            )

            # Accumulate for metrics
            all_predictions.append(predictions)
            all_labels.append(labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()

            # Collect trajectory data if requested
            if collect_trajectory_data:
                trajectory_data.extend(batch_trajectory_data)

            # Update progress bar
            if (batch_idx + 1) % log_every == 0 or batch_idx == len(train_loader) - 1:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Callback: on_batch_end
            batch_logs = {'batch_loss': loss.item()}
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_logs)

        avg_loss = total_loss / len(train_loader)

        # Compute metrics from accumulated predictions
        metrics = self._compute_metrics(all_predictions, all_labels)

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'trajectory_data': trajectory_data
        }

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Validate for one epoch with metric computation.

        Returns:
            Dictionary with 'loss', 'accuracy', 'f1_score', 'trajectory_data'
        """
        # Callback: on_validation_begin
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        self.model.eval()
        total_loss = 0.0
        trajectory_data = []

        # Accumulate predictions and labels for metrics
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Forward pass only
                loss, batch_trajectory_data, predictions, labels = self._validation_step_with_metrics(
                    batch,
                    collect_trajectory_data
                )

                # Accumulate for metrics
                all_predictions.append(predictions)
                all_labels.append(labels)

                total_loss += loss.item()

                # Collect trajectory data if requested
                if collect_trajectory_data:
                    trajectory_data.extend(batch_trajectory_data)

        avg_loss = total_loss / len(val_loader)

        # Compute metrics from accumulated predictions
        metrics = self._compute_metrics(all_predictions, all_labels)

        # Callback: on_validation_end
        val_logs = {
            'val_loss': avg_loss,
            'val_accuracy': metrics['accuracy'],
            'val_f1': metrics['f1_score']
        }
        for callback in self.callbacks:
            callback.on_validation_end(self, val_logs)

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'trajectory_data': trajectory_data
        }

    def _compute_metrics(
        self,
        all_predictions: List[np.ndarray],
        all_labels: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute accuracy and F1 from accumulated predictions and labels using configurable threshold."""
        # Concatenate all predictions and labels
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Ensure arrays have the same shape
        if predictions.shape != labels.shape:
            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]

        # Convert to binary using configurable threshold
        binary_predictions = (predictions > self.classification_threshold).astype(int)
        binary_ground_truth = (labels > 0.5).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((binary_predictions == 1) & (binary_ground_truth == 1))
        fp = np.sum((binary_predictions == 1) & (binary_ground_truth == 0))
        fn = np.sum((binary_predictions == 0) & (binary_ground_truth == 1))
        tn = np.sum((binary_predictions == 0) & (binary_ground_truth == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }

    def _train_step_with_metrics(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one training step and return predictions/labels for metrics.

        Returns:
            Tuple of (loss, trajectory_data, predictions, labels)
        """
        # Process batch normally
        loss, trajectory_data = self._process_batch(batch, training=True, collect_trajectory_data=collect_trajectory_data)

        # Extract predictions and labels for metrics
        with torch.no_grad():
            tokenized_actions = batch['proprio_features'].to(self.device)
            dino_features = torch.cat([
                batch['dino_features']['begin_frame'],
                batch['dino_features']['middle_frame'],
                batch['dino_features']['end_frame']
            ], dim=1).to(self.device)
            labels = batch['train_labels'].to(self.device)

            # Crop labels to prediction window (same as in _process_batch)
            labels = self.crop_labels_fn(
                labels,
                self.prediction_window_size,
                self.prediction_stride
            )
            labels = labels[:, labels.shape[1] // 2 - self.prediction_window_size // 2:
                              labels.shape[1] // 2 + self.prediction_window_size // 2]

            # Get predictions
            logits = self.model(tokenized_actions, None, dino_features)
            probabilities = torch.sigmoid(logits)

            # Convert to numpy for metrics
            predictions_np = probabilities.cpu().detach().numpy().flatten()
            labels_np = labels.cpu().detach().numpy().flatten()

        return loss, trajectory_data, predictions_np, labels_np

    def _validation_step_with_metrics(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one validation step and return predictions/labels for metrics.

        Returns:
            Tuple of (loss, trajectory_data, predictions, labels)
        """
        # Process batch normally
        loss, trajectory_data = self._process_batch(batch, training=False, collect_trajectory_data=collect_trajectory_data)

        # Extract predictions and labels for metrics
        tokenized_actions = batch['proprio_features'].to(self.device)
        dino_features = torch.cat([
            batch['dino_features']['begin_frame'],
            batch['dino_features']['middle_frame'],
            batch['dino_features']['end_frame']
        ], dim=1).to(self.device)
        labels = batch['train_labels'].to(self.device)

        # Crop labels to prediction window (same as in _process_batch)
        labels = self.crop_labels_fn(
            labels,
            self.prediction_window_size,
            self.prediction_stride
        )
        labels = labels[:, labels.shape[1] // 2 - self.prediction_window_size // 2:
                          labels.shape[1] // 2 + self.prediction_window_size // 2]

        # Get predictions
        logits = self.model(tokenized_actions, None, dino_features)
        probabilities = torch.sigmoid(logits)

        # Convert to numpy for metrics
        predictions_np = probabilities.cpu().detach().numpy().flatten()
        labels_np = labels.cpu().detach().numpy().flatten()

        return loss, trajectory_data, predictions_np, labels_np

    def _default_crop_labels(
        self,
        labels: torch.Tensor,
        window_size: int,
        stride: int
    ) -> torch.Tensor:
        """Default label cropping function.

        Ported from andrew's code.
        """
        label_length = labels.size(1)
        center = label_length // 2
        start = center - ((stride * window_size) // 2)
        end = start + stride * window_size

        assert start >= 0 and end <= label_length, \
            f"Start index {start} is negative or end index {end} is greater than label length {label_length}"

        cropped_labels = labels[:, start:end:stride]

        assert cropped_labels.size(1) == window_size, \
            f"Cropped labels have {cropped_labels.size(1)} frames, expected {window_size}"

        return cropped_labels

    def _process_batch(
        self,
        batch: Dict[str, Any],
        training: bool = True,
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Process one batch for boundary prediction.

        Expected batch format (from andrew's TrainingDataset):
        - 'proprio_features': Tokenized proprioceptive actions
        - 'dino_features': Dict with 'begin_frame', 'middle_frame', 'end_frame'
        - 'train_labels': Smoothed boundary labels
        - 'task_id', 'episode_id', 'window', 'intervals': Metadata

        Returns:
            Tuple of (loss, trajectory_data)
        """
        # Move data to device
        tokenized_actions = batch['proprio_features'].to(self.device)
        dino_features = torch.cat([
            batch['dino_features']['begin_frame'],
            batch['dino_features']['middle_frame'],
            batch['dino_features']['end_frame']
        ], dim=1).to(self.device)
        labels = batch['train_labels'].to(self.device)

        # Crop labels to prediction window
        labels = self.crop_labels_fn(
            labels,
            self.prediction_window_size,
            self.prediction_stride
        )

        # Additional cropping to center window (from andrew's code)
        labels = labels[:, labels.shape[1] // 2 - self.prediction_window_size // 2:
                          labels.shape[1] // 2 + self.prediction_window_size // 2]

        # Forward pass
        logits = self.model(tokenized_actions, None, dino_features)

        # Calculate loss
        loss = self.criterion(logits, labels)

        # Collect trajectory data if requested
        trajectory_data = []
        if collect_trajectory_data:
            with torch.no_grad():
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()

                for i in range(len(batch['task_id'])):
                    trajectory_data.append({
                        'task_id': batch['task_id'][i],
                        'episode_id': batch['episode_id'][i],
                        'window': batch['window'][i],
                        'predictions': predictions[i].cpu().detach().numpy(),
                        'ground_truth': labels[i].cpu().detach().numpy(),
                        'confidence_scores': probabilities[i].cpu().detach().numpy(),
                        'intervals': batch['intervals'][i]
                    })

        return loss, trajectory_data
