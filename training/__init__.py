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
# Legacy config moved to config/config_schema.py
# Import from there instead:
# from config import TrainingConfig, WandBConfig, DataConfig
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

    # Configuration (moved to config/ module)
    # Use: from config import TrainingConfig, WandBConfig, DataConfig

    # Label smoothing
    'LabelSmoother',
    'GaussianSmoother',
    'TriangularSmoother',
    'UniformWindowSmoother',
    'ExponentialSmoother',
    'NoSmoother',
    'create_smoother'
]
