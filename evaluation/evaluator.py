"""Unified evaluation pipeline for boundary prediction models.

This module provides a complete evaluation pipeline that runs inference,
computes comprehensive metrics, and generates visualizations.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from .metrics import BoundaryMetrics, calculate_segment_metrics
from .visualizer import (
    plot_metrics_comparison,
    create_metrics_summary,
    plot_confusion_matrix,
    plot_boundary_predictions,
    save_metrics_to_csv
)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Thresholds
    iou_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    confidence_threshold: float = 0.5

    # Output options
    save_predictions: bool = True
    save_visualizations: bool = True
    save_metrics_csv: bool = True
    num_visualizations: int = 5

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelEvaluator:
    """
    Comprehensive model evaluator.

    This class provides a complete evaluation pipeline including:
    - Inference on test data
    - Comprehensive metrics computation
    - Visualization generation
    - Results saving

    Example:
        >>> evaluator = ModelEvaluator(model, config)
        >>> results = evaluator.evaluate(
        ...     test_loader,
        ...     save_dir='./evaluation_results'
        ... )
        >>> print(f"F1@IoU(0.5): {results['f1_iou_0.5']['f1']:.3f}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            config: Evaluation configuration
        """
        self.model = model
        self.config = config or EvaluationConfig()
        self.metrics_calculator = BoundaryMetrics(iou_thresholds=self.config.iou_thresholds)

        # Move model to device and set to eval mode
        self.model.to(self.config.device)
        self.model.eval()

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        save_dir: Optional[Path] = None,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            test_loader: DataLoader for test data
            save_dir: Directory to save results (creates if doesn't exist)
            visualize: Whether to generate and save visualizations

        Returns:
            Dictionary with all evaluation results
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("EVALUATION PIPELINE")
        print("="*80 + "\n")

        # Run inference
        print("Running inference on test set...")
        predictions_data = self._run_inference(test_loader)

        # Compute metrics
        print("\nComputing comprehensive metrics...")
        metrics = self._compute_all_metrics(predictions_data)

        # Save results
        if save_dir:
            print("\nSaving results...")
            self._save_results(metrics, predictions_data, save_dir)

        # Generate visualizations
        if visualize and save_dir:
            print("\nGenerating visualizations...")
            self._generate_visualizations(metrics, predictions_data, save_dir)

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        self._print_metrics_summary(metrics)

        return metrics

    def _run_inference(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Run inference on test set.

        Returns:
            Dictionary with predictions, ground truth, and confidence scores
        """
        all_predictions = []
        all_ground_truth = []
        all_confidence_scores = []
        all_timestamps = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference"):
                # Get inputs
                tokens = batch.get('proprio_features', None)
                dino = batch.get('dino_features', None)
                labels = batch['labels']

                # Move to device
                if tokens is not None:
                    tokens = tokens.to(self.config.device)
                if dino is not None:
                    dino = dino.to(self.config.device)
                labels = labels.to(self.config.device)

                # Forward pass
                logits = self.model(tokens, None, dino)

                # Get probabilities
                probs = torch.sigmoid(logits)

                # Get predictions (threshold at 0.5)
                preds = (probs > self.config.confidence_threshold).float()

                # Collect results
                all_predictions.append(preds.cpu().numpy())
                all_ground_truth.append(labels.cpu().numpy())
                all_confidence_scores.append(probs.cpu().numpy())

                # Timestamps if available
                if 'timestamps' in batch:
                    all_timestamps.append(batch['timestamps'].cpu().numpy())

        # Concatenate all batches
        predictions_data = {
            'predictions': np.concatenate(all_predictions, axis=0),
            'ground_truth': np.concatenate(all_ground_truth, axis=0),
            'confidence_scores': np.concatenate(all_confidence_scores, axis=0),
        }

        if all_timestamps:
            predictions_data['timestamps'] = np.concatenate(all_timestamps, axis=0)

        return predictions_data

    def _compute_all_metrics(
        self,
        predictions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary with all metrics
        """
        # Get data
        predictions = predictions_data['predictions']
        ground_truth = predictions_data['ground_truth']
        confidence_scores = predictions_data['confidence_scores']
        timestamps = predictions_data.get('timestamps', None)

        # Flatten if needed (for window-based predictions)
        if predictions.ndim > 1:
            # Take center frame of prediction window
            center_idx = predictions.shape[1] // 2
            predictions = predictions[:, center_idx]
            ground_truth = ground_truth[:, center_idx]
            confidence_scores = confidence_scores[:, center_idx]
            if timestamps is not None:
                timestamps = timestamps[:, center_idx]

        # Compute comprehensive metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            timestamps=timestamps
        )

        # Compute segment-level metrics
        segment_metrics = calculate_segment_metrics(predictions, ground_truth)
        metrics.update(segment_metrics)

        return metrics

    def _save_results(
        self,
        metrics: Dict[str, Any],
        predictions_data: Dict[str, Any],
        save_dir: Path
    ) -> None:
        """Save evaluation results to disk."""

        # Save metrics as JSON
        metrics_path = save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_metrics = self._convert_to_json_serializable(metrics)
            json.dump(json_metrics, f, indent=2)
        print(f"  Metrics saved to: {metrics_path}")

        # Save metrics summary as text
        summary_path = save_dir / 'metrics_summary.txt'
        summary = create_metrics_summary({'model': metrics}, output_path=str(summary_path))
        print(f"  Summary saved to: {summary_path}")

        # Save metrics as CSV
        if self.config.save_metrics_csv:
            csv_path = save_dir / 'metrics.csv'
            save_metrics_to_csv({'model': metrics}, str(csv_path))

        # Save predictions
        if self.config.save_predictions:
            pred_path = save_dir / 'predictions.npz'
            np.savez(pred_path, **predictions_data)
            print(f"  Predictions saved to: {pred_path}")

    def _generate_visualizations(
        self,
        metrics: Dict[str, Any],
        predictions_data: Dict[str, Any],
        save_dir: Path
    ) -> None:
        """Generate and save visualizations."""

        viz_dir = save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # Confusion matrix
        try:
            plot_confusion_matrix(
                tp=metrics.get('tp', 0),
                fp=metrics.get('fp', 0),
                fn=metrics.get('fn', 0),
                tn=metrics.get('tn', 0),
                save_path=str(viz_dir / 'confusion_matrix.png')
            )
        except Exception as e:
            print(f"  Warning: Could not generate confusion matrix: {e}")

        # Sample predictions
        if self.config.num_visualizations > 0:
            self._plot_sample_predictions(
                predictions_data,
                viz_dir,
                num_samples=self.config.num_visualizations
            )

    def _plot_sample_predictions(
        self,
        predictions_data: Dict[str, Any],
        viz_dir: Path,
        num_samples: int = 5
    ) -> None:
        """Plot sample predictions."""

        predictions = predictions_data['predictions']
        ground_truth = predictions_data['ground_truth']
        confidence_scores = predictions_data['confidence_scores']

        # Flatten if needed
        if predictions.ndim > 1:
            center_idx = predictions.shape[1] // 2
            predictions = predictions[:, center_idx]
            ground_truth = ground_truth[:, center_idx]
            confidence_scores = confidence_scores[:, center_idx]

        # Sample random indices
        num_samples = min(num_samples, len(predictions))
        sample_indices = np.random.choice(len(predictions), num_samples, replace=False)

        for i, idx in enumerate(sample_indices):
            # Get a window around this index
            window_size = min(200, len(predictions))
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(predictions), start_idx + window_size)

            frames = np.arange(start_idx, end_idx)
            pred_window = predictions[start_idx:end_idx]
            gt_window = ground_truth[start_idx:end_idx]
            conf_window = confidence_scores[start_idx:end_idx]

            try:
                plot_boundary_predictions(
                    frames=frames,
                    predictions=pred_window,
                    ground_truth=gt_window,
                    confidence_scores=conf_window,
                    save_path=str(viz_dir / f'prediction_sample_{i+1}.png')
                )
            except Exception as e:
                print(f"  Warning: Could not generate sample prediction {i+1}: {e}")

    def _print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Print formatted metrics summary to console."""

        print("\nKey Metrics:")
        print("-" * 40)

        # Classification metrics
        if 'precision' in metrics:
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1 Score:   {metrics['f1_score']:.4f}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")

        # F1@IoU metrics
        print("\n  F1@IoU:")
        for threshold in self.config.iou_thresholds:
            key = f'f1_iou_{threshold}'
            if key in metrics:
                f1_value = metrics[key].get('f1', 0) if isinstance(metrics[key], dict) else metrics[key]
                print(f"    @{threshold:.1f}: {f1_value:.4f}")

        # Temporal metrics
        if 'temporal_f1' in metrics:
            print(f"\n  Temporal F1: {metrics['temporal_f1']:.4f}")

        # Confidence metrics
        if 'auc_roc' in metrics:
            print(f"\n  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")

        print()

    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""

        if isinstance(obj, dict):
            return {k: ModelEvaluator._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelEvaluator._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
