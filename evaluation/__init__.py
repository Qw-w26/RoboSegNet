"""Comprehensive evaluation metrics for task segmentation."""

from .metrics import BoundaryMetrics, calculate_segment_metrics
from .f1_iou import (
    f1_at_iou,
    calculate_iou_matrix,
    extract_intervals_from_binary,
    f1_at_multiple_ious
)
from .visualizer import (
    plot_metrics_comparison,
    create_metrics_summary,
    plot_confusion_matrix,
    plot_boundary_predictions,
    save_metrics_to_csv
)
from .evaluator import ModelEvaluator, EvaluationConfig

__all__ = [
    # Metrics
    'BoundaryMetrics',
    'calculate_segment_metrics',

    # F1@IoU
    'f1_at_iou',
    'calculate_iou_matrix',
    'extract_intervals_from_binary',
    'f1_at_multiple_ious',

    # Visualization
    'plot_metrics_comparison',
    'create_metrics_summary',
    'plot_confusion_matrix',
    'plot_boundary_predictions',
    'save_metrics_to_csv',

    # Evaluator
    'ModelEvaluator',
    'EvaluationConfig',
]
