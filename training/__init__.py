"""Training module for segment boundary prediction."""

from .dataset_generator import SegmentBoundaryDataset, create_dataloaders
from .training_dataset import (
    BoundaryDataset,
    TrainingDatasetConfig,
    Window,
    collate_dict
)
from .trainer import Trainer, BoundaryTrainer
from .callbacks import (
    Callback,
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    LearningRateSchedulerCallback
)
from .config import (
    TrainingConfig,
    WandBConfig,
    TensorBoardConfig,
    DataConfig
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
    'SegmentBoundaryDataset',
    'create_dataloaders',
    'BoundaryDataset',
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

    # Configuration
    'TrainingConfig',
    'WandBConfig',
    'TensorBoardConfig',
    'DataConfig',

    # Label smoothing
    'LabelSmoother',
    'GaussianSmoother',
    'TriangularSmoother',
    'UniformWindowSmoother',
    'ExponentialSmoother',
    'NoSmoother',
    'create_smoother'
]
