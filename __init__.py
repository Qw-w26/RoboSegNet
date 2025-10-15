"""
Task Segmentation Framework

A comprehensive framework for task segmentation in robotics trajectories.

This package provides:
- Data loading and management (data/)
- Embedding generation (embedding/)
- Benchmark framework (benchmark/)
- Model architectures (model/)
- Training utilities (training/)

Quick Start:
    >>> from data import AgiBotWorldDataset
    >>> from embedding import create_embedding_function
    >>> from benchmark import BenchmarkRunner
    >>> from model import create_model
    >>> from training import Trainer
"""

__version__ = "0.1.0"

# Re-export commonly used classes and functions
from data import (
    AgiBotWorldDataset,
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation
)

from embedding import (
    FastDROIDEmbedder,
    create_embedding_function
)

from benchmark import (
    BenchmarkRunner,
    SegmentationMetrics,
    evaluate_segmentation
)

from model import (
    BoundaryPredictionMLP,
    FocalLoss,
    create_model
)

from training import (
    SegmentBoundaryDataset,
    Trainer
)

__all__ = [
    # Data
    'AgiBotWorldDataset',
    'BaseSegmentationDataset',
    'TrajectoryData',
    'SegmentAnnotation',
    # Embedding
    'FastDROIDEmbedder',
    'create_embedding_function',
    # Benchmark
    'BenchmarkRunner',
    'SegmentationMetrics',
    'evaluate_segmentation',
    # Model
    'BoundaryPredictionMLP',
    'FocalLoss',
    'create_model',
    # Training
    'SegmentBoundaryDataset',
    'Trainer',
]
