"""Training module for segment boundary prediction."""

from .dataset_generator import SegmentBoundaryDataset, create_dataloaders
from .train import Trainer
from .label_smoothing import (
    LabelSmoother,
    GaussianSmoother,
    TriangularSmoother,
    UniformWindowSmoother,
    ExponentialSmoother,
    NoSmoother,
    create_smoother
)

__all__ = [
    'SegmentBoundaryDataset',
    'create_dataloaders',
    'Trainer',
    'LabelSmoother',
    'GaussianSmoother',
    'TriangularSmoother',
    'UniformWindowSmoother',
    'ExponentialSmoother',
    'NoSmoother',
    'create_smoother'
]
