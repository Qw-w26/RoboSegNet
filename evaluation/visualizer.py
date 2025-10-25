"""Visualization utilities for evaluation metrics.

This module provides functions for plotting and summarizing evaluation
metrics, including comparison plots and formatted summaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from pathlib import Path


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize_per_metric: tuple = (5, 4)
) -> None:
    """
    Plot comparison of metrics across different models/configurations.

    Creates a grid of bar plots, one for each metric, comparing values
    across different models.

    Args:
        metrics_dict: Dictionary with model names as keys and metrics dicts as values
        save_path: Optional path to save the plot
        figsize_per_metric: Figure size per metric subplot (width, height)

    Example:
        >>> metrics = {
        ...     'model_v1': {'precision': 0.85, 'recall': 0.80, 'f1': 0.82},
        ...     'model_v2': {'precision': 0.88, 'recall': 0.84, 'f1': 0.86}
        ... }
        >>> plot_metrics_comparison(metrics, save_path='comparison.png')
    """
    # Extract all metric names
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())

    # Filter to numeric metrics only
    numeric_metrics = []
    for metric in all_metrics:
        first_value = list(metrics_dict.values())[0].get(metric, 0)
        if isinstance(first_value, (int, float)) and not isinstance(first_value, bool):
            numeric_metrics.append(metric)

    if not numeric_metrics:
        print("No numeric metrics to plot")
        return

    # Create subplots
    n_metrics = len(numeric_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_metric[0] * n_cols, figsize_per_metric[1] * n_rows)
    )

    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each metric
    for i, metric in enumerate(sorted(numeric_metrics)):
        ax = axes[i]

        # Extract values for this metric
        model_names = list(metrics_dict.keys())
        values = [metrics_dict[model].get(metric, 0) for model in model_names]

        # Create bar plot
        bars = ax.bar(range(len(model_names)), values, color=sns.color_palette()[0])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Score')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01 * max(values) if max(values) > 0 else 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9
            )

        # Rotate x-axis labels if needed
        max_name_len = max(len(name) for name in model_names)
        if max_name_len > 8 or len(model_names) > 3:
            ax.tick_params(axis='x', rotation=45)

        # Set y-axis limits
        if max(values) > 0:
            ax.set_ylim(0, max(values) * 1.15)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def create_metrics_summary(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
) -> str:
    """
    Create a formatted summary of metrics.

    Organizes metrics by category (classification, F1@IoU, temporal, etc.)
    and creates a nicely formatted text summary.

    Args:
        metrics_dict: Dictionary with model names as keys and metrics dicts as values
        output_path: Optional path to save the summary

    Returns:
        Formatted string summary

    Example:
        >>> metrics = {'model_v1': {'precision': 0.85, 'f1_iou_0.5': 0.82}}
        >>> summary = create_metrics_summary(metrics)
        >>> print(summary)
    """
    summary = []
    summary.append("=" * 80)
    summary.append("EVALUATION METRICS SUMMARY")
    summary.append("=" * 80)
    summary.append("")

    for model_name, metrics in metrics_dict.items():
        summary.append(f"Model: {model_name}")
        summary.append("-" * 80)

        # Define metric categories
        categories = {
            'Classification Metrics': ['precision', 'recall', 'f1_score', 'accuracy'],
            'Confusion Matrix': ['tp', 'fp', 'fn', 'tn'],
            'F1@IoU Metrics': [k for k in metrics.keys() if k.startswith('f1_iou_')],
            'Temporal Metrics': [k for k in metrics.keys() if k.startswith('temporal_')],
            'Confidence Metrics': ['auc_roc', 'auc_pr', 'optimal_threshold', 'max_f1_at_optimal'],
            'Segment Metrics': [k for k in metrics.keys() if k.startswith('segment_')]
        }

        # Print each category
        for category_name, metric_names in categories.items():
            # Check if any metrics in this category exist
            category_metrics = {k: v for k, v in metrics.items() if k in metric_names}

            if category_metrics:
                summary.append(f"\n{category_name}:")

                for metric_name, value in category_metrics.items():
                    formatted_name = metric_name.replace('_', ' ').title()

                    # Format value based on type
                    if isinstance(value, dict):
                        summary.append(f"  {formatted_name}:")
                        for sub_key, sub_value in value.items():
                            summary.append(f"    {sub_key}: {sub_value:.4f}")
                    elif isinstance(value, (int, np.integer)):
                        summary.append(f"  {formatted_name}: {value}")
                    elif isinstance(value, (float, np.floating)):
                        summary.append(f"  {formatted_name}: {value:.4f}")
                    else:
                        summary.append(f"  {formatted_name}: {value}")

        summary.append("")
        summary.append("=" * 80)
        summary.append("")

    result = "\n".join(summary)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)
        print(f"Summary saved to {output_path}")

    return result


def plot_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives
        normalize: Whether to normalize to show proportions
        save_path: Optional path to save the plot

    Example:
        >>> plot_confusion_matrix(tp=85, fp=15, fn=20, tn=80)
    """
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auc_pr: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        auc_pr: Area under PR curve
        save_path: Optional path to save the plot

    Example:
        >>> from sklearn.metrics import precision_recall_curve
        >>> p, r, _ = precision_recall_curve(y_true, y_scores)
        >>> plot_precision_recall_curve(p, r, auc_pr=0.85)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.3f}', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")

    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_roc: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_roc: Area under ROC curve
        save_path: Optional path to save the plot

    Example:
        >>> from sklearn.metrics import roc_curve
        >>> fpr, tpr, _ = roc_curve(y_true, y_scores)
        >>> plot_roc_curve(fpr, tpr, auc_roc=0.92)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()


def plot_boundary_predictions(
    frames: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot boundary predictions vs ground truth over time.

    Args:
        frames: Frame indices or timestamps
        predictions: Binary predictions
        ground_truth: Ground truth labels
        confidence_scores: Optional confidence scores to plot
        save_path: Optional path to save the plot

    Example:
        >>> plot_boundary_predictions(
        ...     frames=np.arange(len(pred)),
        ...     predictions=pred,
        ...     ground_truth=gt,
        ...     confidence_scores=conf
        ... )
    """
    fig, axes = plt.subplots(2 if confidence_scores is not None else 1, 1,
                             figsize=(12, 6 if confidence_scores is None else 9))

    if confidence_scores is None:
        axes = [axes]

    # Plot binary predictions vs ground truth
    ax = axes[0]
    ax.plot(frames, ground_truth, 'g-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(frames, predictions, 'r--', label='Predictions', linewidth=2, alpha=0.7)
    ax.set_xlabel('Frame / Time')
    ax.set_ylabel('Boundary (0/1)')
    ax.set_title('Boundary Predictions vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot confidence scores if provided
    if confidence_scores is not None:
        ax = axes[1]
        ax.plot(frames, confidence_scores, 'b-', label='Confidence Score', linewidth=2)
        ax.fill_between(frames, 0, confidence_scores, alpha=0.3)
        ax.axhline(y=0.5, color='k', linestyle='--', label='Threshold (0.5)', alpha=0.5)
        ax.set_xlabel('Frame / Time')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Prediction Confidence Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boundary predictions plot saved to {save_path}")

    plt.show()


def save_metrics_to_csv(
    metrics_dict: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save metrics to CSV file for further analysis.

    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        output_path: Path to save CSV file

    Example:
        >>> metrics = {'model_v1': {'precision': 0.85, 'recall': 0.80}}
        >>> save_metrics_to_csv(metrics, 'metrics.csv')
    """
    import pandas as pd

    # Flatten nested dictionaries
    flattened = {}
    for model_name, metrics in metrics_dict.items():
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}.{sub_key}"] = sub_value
            else:
                flat_metrics[key] = value
        flattened[model_name] = flat_metrics

    # Create DataFrame
    df = pd.DataFrame(flattened).T
    df.index.name = 'model'

    # Save to CSV
    df.to_csv(output_path)
    print(f"Metrics saved to CSV: {output_path}")
