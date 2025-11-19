"""Training module for segment boundary prediction."""

from .training_dataset import (
    BoundaryDataset,
    TrainingDatasetConfig,
    Window,
    collate_dict
)
from .lazy_dataset import LazyBoundaryDataset
from .trainer import Trainer, BoundaryTrainer
from .callbacks import (
    Callback,
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    LearningRateSchedulerCallback,
)
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
    # Dataset
    'BoundaryDataset',
    'LazyBoundaryDataset',
    'TrainingDatasetConfig',
    'Window',
    'collate_dict',

    # Training
    'Trainer',
    'BoundaryTrainer',

    # Callbacks
    'Callback',
    'WandBCallback',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    'TensorBoardCallback',
    'LearningRateSchedulerCallback',

    # Label smoothing
    'LabelSmoother',
    'GaussianSmoother',
    'TriangularSmoother',
    'UniformWindowSmoother',
    'ExponentialSmoother',
    'NoSmoother',
    'create_smoother'
]
