"""
Task Segmentation Framework

A comprehensive framework for task segmentation in robotics trajectories.

This package provides:
- Data loading and management (data/)
- Model architectures (models/)
- Training utilities (training/)
- Evaluation metrics (evaluation/)

Quick Start:
    >>> from data import load_multi_task_split
    >>> from models import create_model
    >>> from training import BoundaryTrainer
"""

__version__ = "0.1.0"

# Re-export commonly used classes and functions
from data import (
    load_parquet_split,
    load_multi_task_split,
    ParquetTrajectory,
)

from models import (
    BoundaryPredictionMLP,
    BoundaryDETR,
    create_model,
)

from training import (
    BoundaryTrainer,
    Trainer,
)

__all__ = [
    # Data
    'load_parquet_split',
    'load_multi_task_split',
    'ParquetTrajectory',
    # Models
    'BoundaryPredictionMLP',
    'BoundaryDETR',
    'create_model',
    # Training
    'BoundaryTrainer',
    'Trainer',
]
