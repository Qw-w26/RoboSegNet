"""Unified configuration schema for boundary prediction experiments.

This module provides a single ExperimentConfig that consolidates all
configuration dataclasses used throughout the codebase. Configs can be
loaded from YAML files and easily serialized.

Example:
    >>> from config import ExperimentConfig, load_config
    >>> config = load_config('configs/experiments/task_357.yaml')
    >>> print(config.training.learning_rate)
    >>> config.model.vocab_size = 4096
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        vocab_size: Tokenizer vocabulary size
        token_embed_dim: Dimension of token embeddings
        dino_feature_dim: DINO feature dimension (3 frames * 1024 = 3072)
        hidden_dims: List of hidden layer dimensions for MLP
        prediction_window_size: Number of frames to predict boundaries for
        use_proprio: Whether to use proprioceptive features
        use_dino: Whether to use DINO visual features
        transformer_dropout: Dropout for transformer layers
        embedding_dropout: Dropout for embedding layers
        dino_encoder_dropout: Dropout for DINO encoder
        mlp_dropout: Dropout for MLP layers
        feature_noise_std: Standard deviation for feature noise augmentation
    """
    vocab_size: int = 2048
    token_embed_dim: int = 128
    dino_feature_dim: int = 3072  # 3 frames * 1024
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    prediction_window_size: int = 18

    use_proprio: bool = True
    use_dino: bool = True

    transformer_dropout: float = 0.3
    embedding_dropout: float = 0.1
    dino_encoder_dropout: float = 0.3
    mlp_dropout: float = 0.3
    feature_noise_std: float = 0.05

    def __post_init__(self):
        if not (self.use_proprio or self.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration.

    Combines dataset, dataloader, and feature extraction settings.

    Attributes:
        # Paths
        splits_dir: Directory containing train/val/test splits
        processed_dataset_dir: Directory with processed parquet files
        cache_dir: Directory for feature caching

        # Dataset parameters
        window_size: Window size for boundary prediction (frames)
        stride: Stride for window sampling (frames)
        fps: Target FPS for data (should match preprocessing)
        action_dim: Action dimension for proprioception

        # DataLoader
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        pin_memory: Pin memory for faster GPU transfer

        # Feature extraction
        vocab_size: Tokenizer vocabulary size (must match model)
        label_smoother_type: Type of label smoother ('gaussian', 'triangular', etc.)
        label_smoother_sigma: Sigma for Gaussian smoother
        label_smoother_peak_value: Peak value for smoothed labels

        # Processing
        ray_num_workers: Number of Ray workers for parallel processing
        max_samples: Maximum samples to load (None = all)
    """
    # Paths
    splits_dir: str = './splits'
    processed_dataset_dir: Optional[str] = None  # Auto-determined from splits_dir if None
    cache_dir: str = './cache'

    # Dataset parameters
    window_size: int = 180
    stride: int = 10
    fps: float = 5.0
    action_dim: int = 14

    # DataLoader
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    # Feature extraction
    vocab_size: int = 2048
    label_smoother_type: str = 'gaussian'
    label_smoother_sigma: float = 5.0
    label_smoother_peak_value: float = 1.0

    # Processing
    ray_num_workers: int = 4
    max_samples: Optional[int] = None

    def __post_init__(self):
        """Set defaults for derived paths."""
        if self.processed_dataset_dir is None:
            self.processed_dataset_dir = str(Path(self.splits_dir) / 'processed_dataset')


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')

        # Scheduler
        scheduler: Learning rate scheduler ('cosine', 'step', 'plateau', None)
        min_lr: Minimum learning rate

        # Loss
        loss_type: Loss function ('bce', 'focal', 'soft_focal', 'weighted_bce')
        loss_pos_weight: Positive class weight for BCE loss
        loss_alpha: Alpha for focal loss
        loss_gamma: Gamma for focal loss

        # Metrics
        classification_threshold: Threshold for binary classification (default: 0.5)

        # Regularization
        gradient_clip_val: Gradient clipping value (None to disable)
        use_amp: Use automatic mixed precision

        # Checkpointing
        checkpoint_dir: Directory for checkpoints
        save_every: Save checkpoint every N epochs
        save_best_only: Only save best model

        # Early stopping
        early_stopping: Enable early stopping
        patience: Early stopping patience

        # Logging
        log_every: Log metrics every N batches
        log_plots_every: Log plots every N epochs
        num_plots: Number of plots to log

        # System
        device: Device ('cuda', 'cpu', 'auto')
        seed: Random seed
    """
    # Training
    num_epochs: int = 100
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'

    # Scheduler
    scheduler: Optional[str] = 'cosine'
    min_lr: float = 1e-6

    # Loss
    loss_type: str = 'bce'
    loss_pos_weight: float = 10.0
    loss_alpha: float = 0.25
    loss_gamma: float = 2.0

    # Metrics
    classification_threshold: float = 0.5

    # Regularization
    gradient_clip_val: Optional[float] = None
    use_amp: bool = False

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10
    save_best_only: bool = True

    # Early stopping
    early_stopping: bool = False
    patience: int = 10

    # Logging
    log_every: int = 10
    log_plots_every: int = 3
    num_plots: int = 3

    # System
    device: str = 'auto'
    seed: int = 42


@dataclass
class WandBConfig:
    """Weights & Biases logging configuration.

    Attributes:
        enabled: Whether to enable WandB logging
        project: WandB project name
        entity: WandB entity (team name)
        name: Run name (auto-generated if None)
        tags: Tags for this run
        notes: Notes about this run
        mode: WandB mode ('online', 'offline', 'disabled')
        log_model: Whether to log model checkpoints
    """
    enabled: bool = False
    project: str = 'task-segmentation'
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: str = 'online'
    log_model: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    This is the root configuration that contains all sub-configurations.

    Attributes:
        experiment_name: Name of this experiment
        model: Model configuration
        data: Data configuration
        training: Training configuration
        wandb: WandB configuration

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name='task_357_baseline',
        ...     model=ModelConfig(vocab_size=2048),
        ...     data=DataConfig(batch_size=512),
        ...     training=TrainingConfig(num_epochs=100)
        ... )
    """
    experiment_name: str = 'boundary_prediction'
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        """Ensure nested configs are proper dataclass instances."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.wandb, dict):
            self.wandb = WandBConfig(**self.wandb)

        # Sync vocab_size between model and data
        if self.model.vocab_size != self.data.vocab_size:
            self.data.vocab_size = self.model.vocab_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# For backward compatibility with existing code
__all__ = [
    'ExperimentConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'WandBConfig',
]
