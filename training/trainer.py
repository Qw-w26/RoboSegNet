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

from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from scipy.signal import find_peaks

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
            val_results = {}
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
                'train_precision': train_results.get('precision', 0.0),
                'train_recall': train_results.get('recall', 0.0),
                'train_avg_iou': train_results.get('avg_iou', 0.0),
                'train_avg_peak_distance': train_results.get('avg_peak_distance', 0.0),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_precision': val_results.get('precision', 0.0),
                'val_recall': val_results.get('recall', 0.0),
                'val_avg_iou': val_results.get('avg_iou', 0.0),
                'val_avg_peak_distance': val_results.get('avg_peak_distance', 0.0),
                'learning_rate': current_lr,
                'epoch': epoch
            }

            # Add trajectory data if collected (legacy window-level plots)
            if should_collect_plots:
                logs['train_trajectory_data'] = train_trajectory_data
                logs['val_trajectory_data'] = val_trajectory_data

            # Add trajectory-level plots (always include when available)
            train_trajectory_plots = train_results.get('trajectory_plots', [])
            val_trajectory_plots = val_results.get('trajectory_plots', []) if val_loader is not None else []

            if train_trajectory_plots:
                logs['train_trajectory_plots'] = train_trajectory_plots
            if val_trajectory_plots:
                logs['val_trajectory_plots'] = val_trajectory_plots

            # Print epoch summary with peak detection metrics
            print(f"Train Loss: {train_loss:.4f}, P: {train_results.get('precision', 0.0):.3f}, R: {train_results.get('recall', 0.0):.3f}, F1: {train_results.get('f1_score', 0.0):.3f}, IoU: {train_results.get('avg_iou', 0.0):.3f}", end='')
            if val_loss is not None:
                print(f" | Val Loss: {val_loss:.4f}, P: {val_results.get('precision', 0.0):.3f}, R: {val_results.get('recall', 0.0):.3f}, F1: {val_f1:.3f}, IoU: {val_results.get('avg_iou', 0.0):.3f}", end='')
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
        classification_threshold: float = 0.5,
        use_distribution_loss: bool = False,
        freq_multiplier: int = 1,
        peak_detection_distance: float = 5.0,
        iou_window_size: float = 5.0
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
        self.use_distribution_loss = use_distribution_loss
        self.freq_multiplier = freq_multiplier  # For mapping window fps to proprio fps

        # Evaluation parameters (in seconds, will be scaled to frames at proprio_fps)
        self.peak_detection_distance = peak_detection_distance  # Minimum seconds between peaks
        self.iou_window_size = iou_window_size  # Window size in seconds for IoU calculation

        if freq_multiplier > 1:
            print(f"Frequency multiplier: {freq_multiplier}x (window frames will be mapped to timeline)")

        print(f"Classification threshold: {self.classification_threshold}")
        print(f"Using distribution loss: {self.use_distribution_loss}")
        print(f"Evaluation parameters (will be scaled by freq_multiplier):")
        print(f"  Peak detection distance: {self.peak_detection_distance}s")
        print(f"  Matching: At DINO frequency (no tolerance)")
        print(f"  IoU window size: ±{self.iou_window_size}s")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        log_every: int = 10,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Train for one epoch with trajectory-level metric computation.

        Returns:
            Dictionary with 'loss', 'accuracy', 'f1_score', 'trajectory_data', 'trajectory_plots'
        """
        self.model.train()
        total_loss = 0.0
        trajectory_data = []

        # Accumulate batch metadata for trajectory-level metrics
        all_batch_data = []

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Callback: on_batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Forward pass
            loss, batch_trajectory_data, batch_metadata = self._train_step_with_metrics(
                batch, collect_trajectory_data
            )

            # Accumulate batch metadata for trajectory-level metrics
            all_batch_data.append(batch_metadata)

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

        # Compute trajectory-level metrics
        metrics = self._compute_metrics([], [], all_batch_data=all_batch_data)

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'trajectory_data': trajectory_data,
            'trajectory_plots': metrics.get('trajectory_plots', [])
        }

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        collect_trajectory_data: bool = False
    ) -> Dict[str, Any]:
        """Validate for one epoch with trajectory-level metric computation.

        Returns:
            Dictionary with 'loss', 'accuracy', 'f1_score', 'trajectory_data', 'trajectory_plots'
        """
        # Callback: on_validation_begin
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        self.model.eval()
        total_loss = 0.0
        trajectory_data = []

        # Accumulate batch metadata for trajectory-level metrics
        all_batch_data = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Forward pass only
                loss, batch_trajectory_data, batch_metadata = self._validation_step_with_metrics(
                    batch,
                    collect_trajectory_data
                )

                # Accumulate batch metadata for trajectory-level metrics
                all_batch_data.append(batch_metadata)

                total_loss += loss.item()

                # Collect trajectory data if requested
                if collect_trajectory_data:
                    trajectory_data.extend(batch_trajectory_data)

        avg_loss = total_loss / len(val_loader)

        # Compute trajectory-level metrics
        metrics = self._compute_metrics([], [], all_batch_data=all_batch_data)

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
            'trajectory_data': trajectory_data,
            'trajectory_plots': metrics.get('trajectory_plots', [])
        }

    def _accumulate_trajectory_predictions(
        self,
        all_batch_data: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Accumulate window predictions into full trajectory predictions.

        Handles both per-frame (sigmoid) and distribution (softmax) predictions differently:
        - Per-frame: Simple averaging of overlapping predictions
        - Distribution: Normalize predictions before accumulation to maintain relative scales

        Args:
            all_batch_data: List of batch data dictionaries containing predictions, labels, and metadata

        Returns:
            Dictionary mapping (task_id, episode_id) to trajectory data with:
                - 'timeline_scores': Accumulated prediction scores for each frame
                - 'timeline_counts': Number of windows contributing to each frame
                - 'ground_truth_transitions': List of ground truth transition frame indices
                - 'num_frames': Total number of frames in trajectory
        """
        # First pass: determine trajectory lengths
        # NOTE: Timeline is at proprio fps (30 Hz), not window fps (2 Hz)
        # Intervals are at proprio fps, so we use them to determine timeline size
        traj_max_frames = {}
        traj_intervals = {}

        for batch_data in all_batch_data:
            batch_size = len(batch_data['task_ids'])

            for i in range(batch_size):
                task_id = batch_data['task_ids'][i]
                episode_id = batch_data['episode_ids'][i]
                window = batch_data['windows'][i]
                intervals = batch_data['intervals'][i]

                traj_key = (task_id, episode_id)

                # Track maximum frame index seen
                if traj_key not in traj_max_frames:
                    traj_max_frames[traj_key] = 0
                    traj_intervals[traj_key] = intervals

                # Use interval endpoints to determine timeline size (at proprio fps)
                if intervals:
                    max_interval_frame = max([interval.end_frame for interval in intervals])
                    traj_max_frames[traj_key] = max(traj_max_frames[traj_key], max_interval_frame)

        # Second pass: initialize trajectories with correct sizes
        trajectories = {}
        for traj_key, max_frame in traj_max_frames.items():
            intervals = traj_intervals[traj_key]
            trajectories[traj_key] = {
                'timeline_scores': np.zeros(max_frame, dtype=np.float32),
                'timeline_counts': np.zeros(max_frame, dtype=np.int32),
                'ground_truth_transitions': [interval.start_frame for interval in intervals[1:]] if len(intervals) > 1 else [],
                'num_frames': max_frame
            }

        # Third pass: accumulate predictions
        for batch_data in all_batch_data:
            batch_size = len(batch_data['task_ids'])

            for i in range(batch_size):
                task_id = batch_data['task_ids'][i]
                episode_id = batch_data['episode_ids'][i]
                window = batch_data['windows'][i]
                predictions = batch_data['predictions'][i]  # Shape: (prediction_window_size,)

                traj_key = (task_id, episode_id)
                traj_data = trajectories[traj_key]

                # Map window frames from window_fps to proprio_fps
                # window.begin_frame and window.end_frame are at window_fps (e.g., 2 Hz)
                # timeline is at proprio_fps (e.g., 30 Hz)
                start_idx_window_fps = window.begin_frame
                end_idx_window_fps = window.end_frame

                # Map to proprio fps
                start_idx_proprio_fps = start_idx_window_fps * self.freq_multiplier
                end_idx_proprio_fps = end_idx_window_fps * self.freq_multiplier

                # Window length at proprio fps
                window_length_proprio = end_idx_proprio_fps - start_idx_proprio_fps

                # Upsample predictions from window_fps to proprio_fps
                # predictions has shape (prediction_window_size,) at window_fps
                # We need to upsample to window_length_proprio at proprio_fps
                if len(predictions) < window_length_proprio:
                    # Upsample predictions using linear interpolation
                    indices = np.linspace(0, len(predictions) - 1, window_length_proprio)
                    upsampled_predictions = np.interp(
                        indices,
                        np.arange(len(predictions)),
                        predictions
                    )
                else:
                    upsampled_predictions = predictions[:window_length_proprio]

                # Add predictions to timeline (accumulate overlapping windows)
                end_idx_clipped = min(end_idx_proprio_fps, traj_data['num_frames'])
                valid_length = end_idx_clipped - start_idx_proprio_fps

                if valid_length > 0 and start_idx_proprio_fps < traj_data['num_frames']:
                    traj_data['timeline_scores'][start_idx_proprio_fps:end_idx_clipped] += upsampled_predictions[:valid_length]
                    traj_data['timeline_counts'][start_idx_proprio_fps:end_idx_clipped] += 1

        # Average overlapping predictions
        for traj_key, traj_data in trajectories.items():
            counts = traj_data['timeline_counts']
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            traj_data['timeline_scores'] = traj_data['timeline_scores'] / counts

            # For distribution losses, the averaged softmax probabilities maintain their
            # probability interpretation in [0, 1] range. No additional normalization needed.
            # The classification threshold can now be interpreted as "probability > threshold"
            # consistently across all trajectories.

        return trajectories

    def _compute_trajectory_metrics(
        self,
        trajectories: Dict[Tuple[int, int], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute trajectory-level metrics with peak detection and IoU comparison.

        Args:
            trajectories: Dictionary from _accumulate_trajectory_predictions

        Returns:
            Dictionary with metrics and trajectory plots data
        """
        all_tp = 0
        all_fp = 0
        all_fn = 0

        all_ious = []
        all_peak_distances = []
        trajectory_plots = []

        for traj_key, traj_data in trajectories.items():
            task_id, episode_id = traj_key
            timeline_scores = traj_data['timeline_scores']
            ground_truth_transitions = traj_data['ground_truth_transitions']

            # Use peak detection instead of threshold-based approach
            # Find peaks with reasonable parameters for keyframe detection
            # NOTE: Timeline is at proprio_fps, convert time-based parameters to frames

            # Convert configurable parameters (in seconds) to frames at proprio_fps
            # proprio_fps ≈ window_fps * freq_multiplier
            # So: frames = seconds * window_fps * freq_multiplier

            # For simplicity, we directly compute frames from seconds knowing the timeline frequency
            # peak_detection_distance: minimum time between detected peaks (in seconds)
            # This is converted to frames assuming window_fps as the base frequency
            peak_distance = max(1, int(self.peak_detection_distance * self.freq_multiplier))

            predicted_peaks, peak_properties = find_peaks(
                timeline_scores,
                height=self.classification_threshold,
                distance=peak_distance,  # Minimum frames between boundaries at proprio fps
                prominence=0.01  # Peak must stand out by 1% probability
            )

            # Get peak scores for visualization
            peak_scores = timeline_scores[predicted_peaks] if len(predicted_peaks) > 0 else np.array([])

            # Match predictions to ground truth at DINO frequency (no tolerance)
            # Map both predictions and GT from proprio_fps to window_fps
            # Two predictions/GTs match if they map to the same window frame

            matched_gt = set()
            matched_pred = set()
            ious = []
            distances = []

            # Map GT transitions from proprio_fps to window_fps
            gt_at_window_fps = {}
            for gt_idx in ground_truth_transitions:
                gt_window_frame = gt_idx // self.freq_multiplier
                if gt_window_frame not in gt_at_window_fps:
                    gt_at_window_fps[gt_window_frame] = []
                gt_at_window_fps[gt_window_frame].append(gt_idx)

            # Map predicted peaks from proprio_fps to window_fps
            pred_at_window_fps = {}
            for i, pred_idx in enumerate(predicted_peaks):
                pred_window_frame = pred_idx // self.freq_multiplier
                if pred_window_frame not in pred_at_window_fps:
                    pred_at_window_fps[pred_window_frame] = []
                pred_at_window_fps[pred_window_frame].append((i, pred_idx))

            # Match at window_fps: if pred and GT are in same window frame, they match
            # iou_window_size: window size (in seconds) around each transition for IoU calculation
            iou_window_frames = max(1, int(self.iou_window_size * self.freq_multiplier))

            for window_frame, gt_indices in gt_at_window_fps.items():
                # Check if there's a prediction at this window frame
                if window_frame in pred_at_window_fps:
                    # Match: take the first (or closest) prediction at this window frame
                    for gt_idx in gt_indices:
                        if gt_idx in matched_gt:
                            continue  # Already matched

                        # Find best matching prediction at this window frame
                        best_iou = 0.0
                        best_distance = float('inf')
                        best_pred_idx = None

                        for pred_i, pred_idx in pred_at_window_fps[window_frame]:
                            if pred_i in matched_pred:
                                continue  # Already matched

                            distance = abs(pred_idx - gt_idx)

                            # Compute IoU using a window around each transition
                            gt_window = set(range(max(0, gt_idx - iou_window_frames),
                                                 min(len(timeline_scores), gt_idx + iou_window_frames + 1)))
                            pred_window = set(range(max(0, pred_idx - iou_window_frames),
                                                   min(len(timeline_scores), pred_idx + iou_window_frames + 1)))

                            intersection = len(gt_window & pred_window)
                            union = len(gt_window | pred_window)
                            iou = intersection / union if union > 0 else 0.0

                            if distance < best_distance:
                                best_distance = distance
                                best_iou = iou
                                best_pred_idx = pred_i

                        if best_pred_idx is not None:
                            matched_gt.add(gt_idx)
                            matched_pred.add(best_pred_idx)
                            ious.append(best_iou)
                            distances.append(best_distance)

            # Compute TP, FP, FN based on matches
            tp = len(matched_gt)
            fp = len(predicted_peaks) - len(matched_pred)
            fn = len(ground_truth_transitions) - len(matched_gt)

            all_tp += tp
            all_fp += fp
            all_fn += fn
            all_ious.extend(ious)
            all_peak_distances.extend(distances)

            # Store visualization data with predicted peaks
            trajectory_plots.append({
                'task_id': task_id,
                'episode_id': episode_id,
                'timeline_scores': timeline_scores,
                'ground_truth_transitions': ground_truth_transitions,
                'predicted_peaks': predicted_peaks,
                'peak_scores': peak_scores,
                'threshold': self.classification_threshold,
                'matched_predictions': [predicted_peaks[i] for i in matched_pred],
                'unmatched_predictions': [predicted_peaks[i] for i in range(len(predicted_peaks)) if i not in matched_pred]
            })

        # Calculate overall metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute average IoU and distance for matched transitions
        avg_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
        avg_distance = np.mean(all_peak_distances) if len(all_peak_distances) > 0 else 0.0

        return {
            'accuracy': float(precision),  # For keyframe detection, precision is more meaningful than frame-level accuracy
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'avg_iou': float(avg_iou),
            'avg_peak_distance': float(avg_distance),
            'num_predicted_peaks': int(all_tp + all_fp),
            'num_gt_transitions': int(all_tp + all_fn),
            'trajectory_plots': trajectory_plots
        }

    def _compute_metrics(
        self,
        all_predictions: List[np.ndarray],
        all_labels: List[np.ndarray],
        all_batch_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compute trajectory-level metrics with visualization data.

        Args:
            all_predictions: Legacy parameter (kept for backward compatibility)
            all_labels: Legacy parameter (kept for backward compatibility)
            all_batch_data: List of batch metadata for trajectory-level computation

        Returns:
            Dictionary with accuracy, F1 score, and trajectory plots
        """
        if all_batch_data is not None:
            # New trajectory-level metrics
            trajectories = self._accumulate_trajectory_predictions(all_batch_data)
            return self._compute_trajectory_metrics(trajectories)
        else:
            # Legacy fallback: per-window metrics (for backward compatibility)
            # This path should not be used in normal training
            warnings.warn(
                "Computing per-window metrics. This is deprecated. "
                "Use trajectory-level metrics instead.",
                DeprecationWarning
            )

            predictions = np.concatenate(all_predictions, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            if predictions.shape != labels.shape:
                min_len = min(len(predictions), len(labels))
                predictions = predictions[:min_len]
                labels = labels[:min_len]

            binary_predictions = (predictions > self.classification_threshold).astype(int)
            binary_ground_truth = (labels > 0.5).astype(int)

            tp = np.sum((binary_predictions == 1) & (binary_ground_truth == 1))
            fp = np.sum((binary_predictions == 1) & (binary_ground_truth == 0))
            fn = np.sum((binary_predictions == 0) & (binary_ground_truth == 1))
            tn = np.sum((binary_predictions == 0) & (binary_ground_truth == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

            return {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'trajectory_plots': []
            }

    def _train_step_with_metrics(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one training step and return batch metadata for trajectory-level metrics.

        Returns:
            Tuple of (loss, trajectory_data, batch_metadata)
        """
        # Process batch normally
        loss, trajectory_data = self._process_batch(batch, training=True, collect_trajectory_data=collect_trajectory_data)

        # Extract predictions and metadata for trajectory-level metrics
        with torch.no_grad():
            tokenized_actions = batch['proprio_features'].to(self.device)
            dino_features = torch.cat([
                batch['dino_features']['begin_frame'],
                batch['dino_features']['middle_frame'],
                batch['dino_features']['end_frame']
            ], dim=1).to(self.device)

            # Get predictions
            logits = self.model(tokenized_actions, None, dino_features)

            # Use softmax for distribution losses, sigmoid for per-frame losses
            if self.use_distribution_loss:
                probabilities = torch.softmax(logits, dim=1)
            else:
                probabilities = torch.sigmoid(logits)

            # Convert to numpy
            predictions_np = probabilities.cpu().detach().numpy()

            # Build batch metadata
            batch_metadata = {
                'task_ids': batch['task_id'],
                'episode_ids': batch['episode_id'],
                'windows': batch['window'],
                'predictions': [predictions_np[i] for i in range(len(predictions_np))],
                'intervals': batch['intervals']
            }

        return loss, trajectory_data, batch_metadata

    def _validation_step_with_metrics(
        self,
        batch: Dict[str, Any],
        collect_trajectory_data: bool = False
    ) -> tuple:
        """Execute one validation step and return batch metadata for trajectory-level metrics.

        Returns:
            Tuple of (loss, trajectory_data, batch_metadata)
        """
        # Process batch normally
        loss, trajectory_data = self._process_batch(batch, training=False, collect_trajectory_data=collect_trajectory_data)

        # Extract predictions and metadata for trajectory-level metrics
        tokenized_actions = batch['proprio_features'].to(self.device)
        dino_features = torch.cat([
            batch['dino_features']['begin_frame'],
            batch['dino_features']['middle_frame'],
            batch['dino_features']['end_frame']
        ], dim=1).to(self.device)

        # Get predictions
        logits = self.model(tokenized_actions, None, dino_features)

        # Use softmax for distribution losses, sigmoid for per-frame losses
        if self.use_distribution_loss:
            probabilities = torch.softmax(logits, dim=1)
        else:
            probabilities = torch.sigmoid(logits)

        # Convert to numpy
        predictions_np = probabilities.cpu().detach().numpy()

        # Build batch metadata
        batch_metadata = {
            'task_ids': batch['task_id'],
            'episode_ids': batch['episode_id'],
            'windows': batch['window'],
            'predictions': [predictions_np[i] for i in range(len(predictions_np))],
            'intervals': batch['intervals']
        }

        return loss, trajectory_data, batch_metadata

    def _default_crop_labels(
        self,
        labels: torch.Tensor,
        window_size: int,
        stride: int
    ) -> torch.Tensor:
        """Default label cropping function.

        Handles two cases:
        1. When prediction_window_size < data_window_size: Crops and downsamples
        2. When prediction_window_size == data_window_size: Returns labels as-is or with minimal cropping

        Args:
            labels: Input labels of shape (batch, data_window_size)
            window_size: Target prediction window size (model.prediction_window_size)
            stride: Downsampling stride
        """
        label_length = labels.size(1)

        # Case 1: Prediction window equals or exceeds label length - no complex cropping needed
        if window_size >= label_length:
            # Just return the labels, possibly downsampled
            if stride == 1 or label_length <= window_size:
                return labels
            else:
                # Downsample to fit window_size
                step = max(1, label_length // window_size)
                return labels[:, ::step][:, :window_size]

        # Case 2: Traditional cropping for hierarchical window setup
        # Extract a centered region and downsample
        center = label_length // 2
        required_length = stride * window_size  # Total length needed before downsampling

        # Check if we have enough frames
        if required_length > label_length:
            # Not enough frames - just center-crop what we can and downsample
            start = 0
            end = label_length
            cropped = labels[:, start:end:stride]
            # Pad if needed
            if cropped.size(1) < window_size:
                padding = window_size - cropped.size(1)
                cropped = torch.nn.functional.pad(cropped, (0, padding), mode='replicate')
            return cropped[:, :window_size]

        # Standard case: enough frames for proper cropping
        start = center - (required_length // 2)
        end = start + required_length

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
        # Only apply if labels are still longer than prediction window
        if labels.shape[1] > self.prediction_window_size:
            center = labels.shape[1] // 2
            start = center - self.prediction_window_size // 2
            end = center + self.prediction_window_size // 2
            labels = labels[:, start:end]

        # Ensure labels match prediction window size exactly
        assert labels.shape[1] == self.prediction_window_size, \
            f"Label size {labels.shape[1]} doesn't match prediction window size {self.prediction_window_size}"

        # Forward pass
        logits = self.model(tokenized_actions, None, dino_features)

        # Calculate loss
        loss = self.criterion(logits, labels)

        # Collect trajectory data if requested
        trajectory_data = []
        if collect_trajectory_data:
            with torch.no_grad():
                # Use softmax for distribution losses, sigmoid for per-frame losses
                if self.use_distribution_loss:
                    probabilities = torch.softmax(logits, dim=1)
                    # For distribution: find peaks (argmax gives single boundary per window)
                    predictions = torch.zeros_like(probabilities)
                    max_idx = probabilities.argmax(dim=1)
                    predictions[torch.arange(probabilities.size(0)), max_idx] = 1.0

                    # Normalize ground truth labels to sum to 1 (for fair visual comparison)
                    normalized_labels = labels.clone()
                    label_sum = normalized_labels.sum(dim=1, keepdim=True)
                    label_sum = torch.clamp(label_sum, min=1e-8)  # Avoid division by zero
                    normalized_labels = normalized_labels / label_sum
                else:
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > self.classification_threshold).float()
                    normalized_labels = labels  # Keep raw labels for per-frame losses

                for i in range(len(batch['task_id'])):
                    trajectory_data.append({
                        'task_id': batch['task_id'][i],
                        'episode_id': batch['episode_id'][i],
                        'window': batch['window'][i],
                        'predictions': predictions[i].cpu().detach().numpy(),
                        'ground_truth': normalized_labels[i].cpu().detach().numpy(),
                        'confidence_scores': probabilities[i].cpu().detach().numpy(),
                        'intervals': batch['intervals'][i]
                    })

        return loss, trajectory_data
