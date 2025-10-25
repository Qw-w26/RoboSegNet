"""Training configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=100,
        ...     batch_size=512,
        ...     learning_rate=3e-3
        ... )
    """

    # Training parameters
    num_epochs: int = 100
    """Number of training epochs"""

    batch_size: int = 512
    """Batch size for training"""

    learning_rate: float = 3e-3
    """Initial learning rate"""

    weight_decay: float = 1e-4
    """Weight decay for optimizer"""

    # Optimizer
    optimizer: str = 'adam'
    """Optimizer type: 'adam', 'adamw', 'sgd'"""

    # Learning rate scheduling
    scheduler: Optional[str] = 'cosine'
    """Learning rate scheduler: 'cosine', 'step', 'plateau', None"""

    scheduler_patience: int = 5
    """Patience for ReduceLROnPlateau scheduler"""

    scheduler_factor: float = 0.5
    """Factor for ReduceLROnPlateau scheduler"""

    min_lr: float = 1e-6
    """Minimum learning rate"""

    # Loss function
    loss_type: str = 'soft_focal'
    """Loss function: 'focal', 'soft_focal', 'bce', 'weighted_bce', 'mse'"""

    loss_alpha: float = 0.25
    """Alpha parameter for focal loss"""

    loss_gamma: float = 2.0
    """Gamma parameter for focal loss"""

    loss_pos_weight: float = 10.0
    """Positive class weight for BCE loss"""

    # Early stopping
    early_stopping: bool = True
    """Whether to use early stopping"""

    early_stopping_patience: int = 10
    """Patience for early stopping (epochs)"""

    early_stopping_metric: str = 'val_loss'
    """Metric to monitor for early stopping"""

    early_stopping_mode: str = 'min'
    """Mode for early stopping: 'min' or 'max'"""

    # Checkpointing
    save_checkpoints: bool = True
    """Whether to save checkpoints"""

    checkpoint_every: int = 10
    """Save checkpoint every N epochs"""

    save_best_only: bool = True
    """Whether to save only the best checkpoint"""

    checkpoint_metric: str = 'val_loss'
    """Metric to monitor for best checkpoint"""

    checkpoint_mode: str = 'min'
    """Mode for checkpoint: 'min' or 'max'"""

    # Logging
    log_every: int = 10
    """Log metrics every N steps"""

    log_plots_every: int = 3
    """Log prediction plots every N epochs"""

    num_plots: int = 3
    """Number of prediction plots to log"""

    # Device
    device: str = 'cuda'
    """Device for training: 'cuda' or 'cpu'"""

    # Output
    output_dir: str = './outputs'
    """Output directory for checkpoints and logs"""

    experiment_name: Optional[str] = None
    """Name for this experiment (auto-generated if None)"""

    # Reproducibility
    seed: int = 42
    """Random seed for reproducibility"""

    # Gradient clipping
    gradient_clip_val: Optional[float] = None
    """Gradient clipping value (None to disable)"""

    # Mixed precision
    use_amp: bool = False
    """Whether to use automatic mixed precision training"""

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class WandBConfig:
    """
    Configuration for Weights & Biases logging.

    Example:
        >>> config = WandBConfig(
        ...     project='task-segmentation',
        ...     entity='my-team',
        ...     name='experiment-1'
        ... )
    """

    # Project settings
    project: str = 'task-segmentation'
    """WandB project name"""

    entity: Optional[str] = None
    """WandB entity (team name)"""

    name: Optional[str] = None
    """Run name (auto-generated if None)"""

    tags: List[str] = field(default_factory=list)
    """Tags for this run"""

    notes: Optional[str] = None
    """Notes about this run"""

    # Logging settings
    log_model: bool = True
    """Whether to log model checkpoints to WandB"""

    log_gradients: bool = False
    """Whether to log gradient histograms"""

    log_frequency: int = 100
    """Log frequency in steps"""

    # Mode
    mode: str = 'online'
    """WandB mode: 'online', 'offline', 'disabled'"""

    # Resume
    resume: Optional[str] = None
    """Resume from run ID"""


@dataclass
class TensorBoardConfig:
    """
    Configuration for TensorBoard logging.

    Example:
        >>> config = TensorBoardConfig(log_dir='./logs/experiment-1')
    """

    log_dir: str = './logs'
    """Directory for TensorBoard logs"""

    flush_secs: int = 120
    """How often to flush logs to disk (seconds)"""

    log_histograms: bool = False
    """Whether to log weight/gradient histograms"""


@dataclass
class DataConfig:
    """
    Configuration for data loading.

    Example:
        >>> config = DataConfig(
        ...     train_split_path='./data/splits/task_327_split.json',
        ...     batch_size=512,
        ...     num_workers=4
        ... )
    """

    # Data paths
    data_dir: str = './data'
    """Base data directory"""

    train_split_path: Optional[str] = None
    """Path to train split manifest"""

    val_split_path: Optional[str] = None
    """Path to validation split manifest"""

    # DataLoader settings
    num_workers: int = 4
    """Number of workers for data loading"""

    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer"""

    persistent_workers: bool = True
    """Whether to keep workers alive between epochs"""

    # Sampling
    weighted_sampling: bool = False
    """Whether to use weighted sampling for class imbalance"""

    positive_weight: float = 0.5
    """Weight for positive samples in weighted sampling"""

    # Data augmentation
    augmentation: bool = False
    """Whether to use data augmentation"""

    augmentation_prob: float = 0.5
    """Probability of applying augmentation"""
