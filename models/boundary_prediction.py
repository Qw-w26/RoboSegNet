"""Boundary prediction model combining multi-modal features.

This module provides the main neural network architecture for predicting
task segment boundaries from proprioceptive and visual features.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

from .encoders import TinyFASTEncoder, DinoFeatureEncoder, DINO_MODEL_FEATURE_DIM


# Constants
DEFAULT_TOKEN_EMBED_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_OUTPUT_DIM_PER_FRAME = 32
DEFAULT_EMBEDDING_DROPOUT = 0.1
DEFAULT_FEATURE_NOISE_STD = 0.05
DEFAULT_PAD_TOKEN_OFFSET = 1


@dataclass
class ModelConfig:
    """
    Configuration for the boundary prediction model.

    This dataclass stores all hyperparameters needed to construct
    the BoundaryPredictionMLP model.

    Attributes:
        vocab_size: Tokenizer vocabulary size
        token_embed_dim: Token embedding dimension
        dino_feature_dim: DINO feature dimension (typically 3 frames × 1024)
        hidden_dims: List of hidden layer dimensions for MLP
        prediction_window_size: Number of frames to predict
        transformer_dropout: Dropout for transformer layers
        embedding_dropout: Dropout for embedding layers
        dino_encoder_dropout: Dropout for DINO encoder
        mlp_dropout: Dropout for MLP layers
        feature_noise_std: Std for feature noise augmentation
        use_proprio: Whether to use proprioceptive features
        use_dino: Whether to use DINO visual features
    """

    # Vocabulary and embedding settings
    vocab_size: int = 2048
    token_embed_dim: int = DEFAULT_TOKEN_EMBED_DIM

    # DINO feature settings
    dino_feature_dim: int = 3 * DINO_MODEL_FEATURE_DIM  # 3 frames × 1024

    # Model architecture
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    prediction_window_size: int = 10

    # Dropout settings (separated for different components)
    transformer_dropout: float = 0.3
    embedding_dropout: float = DEFAULT_EMBEDDING_DROPOUT
    dino_encoder_dropout: float = 0.3
    mlp_dropout: float = 0.3

    # Feature processing
    feature_noise_std: float = DEFAULT_FEATURE_NOISE_STD

    # Model components
    use_proprio: bool = True
    use_dino: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not (self.use_proprio or self.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        if self.dino_feature_dim % DINO_MODEL_FEATURE_DIM != 0:
            raise ValueError(
                f"DINO feature dimension ({self.dino_feature_dim}) must be "
                f"multiple of {DINO_MODEL_FEATURE_DIM}"
            )


class BoundaryPredictionMLP(nn.Module):
    """
    Multi-modal MLP for predicting task segment boundaries.

    This model combines proprioceptive token sequences (via TinyFASTEncoder)
    and visual DINO features (via DinoFeatureEncoder), then applies an MLP
    to predict boundary probabilities for a window of frames.

    Architecture:
        Tokenized Actions → TinyFASTEncoder →  ┐
                                               ├→ Concat → MLP → Predictions
        DINO Features → DinoFeatureEncoder →   ┘

    The output is a set of logits for each frame in the prediction window,
    indicating the probability of a boundary at that frame.

    Attributes:
        vocab_size: Total vocabulary size (including padding)
        token_embed_dim: Embedding dimension for tokens
        dino_feature_dim: Dimension of DINO features
        prediction_window_size: Number of frames to predict
        device: Device to run model on
        use_proprio: Whether proprioceptive features are used
        use_dino: Whether DINO features are used
        proprio_encoder: Encoder for proprioceptive tokens (if enabled)
        dino_encoder: Encoder for DINO features (if enabled)
        mlp: MLP for final prediction
    """

    def __init__(
        self,
        config: ModelConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize boundary prediction model.

        Args:
            config: Model configuration
            device: Device to run model on

        Example:
            >>> config = ModelConfig()
            >>> model = BoundaryPredictionMLP(config)
            >>> # Forward pass
            >>> tokens = torch.randint(0, 2048, (32, 100))
            >>> dino = torch.randn(32, 3072)
            >>> logits = model(tokens, None, dino)
            >>> print(logits.shape)  # (32, 10)
        """
        super().__init__()

        # Validate configuration
        if not (config.use_proprio or config.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        # Store configuration
        self.vocab_size = config.vocab_size + DEFAULT_PAD_TOKEN_OFFSET
        self.token_embed_dim = config.token_embed_dim
        self.dino_feature_dim = config.dino_feature_dim
        self.prediction_window_size = config.prediction_window_size
        self.device = device
        self.use_proprio = config.use_proprio
        self.use_dino = config.use_dino

        # Create encoders
        if config.use_proprio:
            self.proprio_encoder = TinyFASTEncoder(
                vocab_size=self.vocab_size,
                embed_dim=config.token_embed_dim,
                num_heads=DEFAULT_NUM_HEADS,
                transformer_dropout=config.transformer_dropout,
                embedding_dropout=config.embedding_dropout
            )

        if config.use_dino:
            self.dino_encoder = DinoFeatureEncoder(
                feature_dim=config.dino_feature_dim,
                dino_encoder_dropout=config.dino_encoder_dropout,
                output_dim_per_frame=DEFAULT_OUTPUT_DIM_PER_FRAME,
                feature_noise_std=config.feature_noise_std
            )

        # Calculate total input dimension
        total_input_dim = 0
        if config.use_proprio:
            total_input_dim += self.proprio_encoder.output_dim
        if config.use_dino:
            total_input_dim += self.dino_encoder.output_dim

        # Build MLP
        layers = []
        prev_dim = total_input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.mlp_dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.prediction_window_size))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            tokenized_actions: Token IDs of shape (batch, seq_len).
                              Required if use_proprio=True, can be None otherwise.
            proprio_stats: Proprioception statistics (currently unused, kept for API compatibility).
            dino_features: DINO features of shape (batch, dino_feature_dim).
                          Required if use_dino=True, can be None otherwise.

        Returns:
            Logits of shape (batch, prediction_window_size)

        Raises:
            ValueError: If required inputs are missing based on model configuration
        """
        # Validate inputs
        if self.use_proprio and tokenized_actions is None:
            raise ValueError("tokenized_actions is required when use_proprio=True")
        if self.use_dino and dino_features is None:
            raise ValueError("dino_features is required when use_dino=True")

        encoded_features = []

        # Encode proprioceptive features
        if self.use_proprio:
            proprio_encoded = self.proprio_encoder(tokenized_actions)
            encoded_features.append(proprio_encoded)

        # Encode visual features
        if self.use_dino:
            dino_encoded = self.dino_encoder(dino_features)
            encoded_features.append(dino_encoded)

        # Concatenate all features
        combined_features = torch.cat(encoded_features, dim=1)

        # Apply MLP
        logits = self.mlp(combined_features)  # (batch, prediction_window_size)

        return logits

    def predict_probabilities(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor],
        use_softmax: bool = False
    ) -> torch.Tensor:
        """
        Get boundary probabilities from logits.

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Proprioception statistics (unused)
            dino_features: DINO features
            use_softmax: If True, apply softmax (for distribution-based losses).
                        If False, apply sigmoid (for per-frame BCE losses).
                        Default: False for backward compatibility.

        Returns:
            Probabilities of shape (batch, prediction_window_size)

        Example:
            >>> # For distribution-based losses (dist_ce, kl_div, js_div)
            >>> probs = model.predict_probabilities(..., use_softmax=True)
            >>>
            >>> # For per-frame losses (bce, weighted_bce)
            >>> probs = model.predict_probabilities(..., use_softmax=False)
        """
        logits = self.forward(tokenized_actions, proprio_stats, dino_features)

        if use_softmax:
            # Distribution-based: probabilities sum to 1 across window
            return torch.softmax(logits, dim=1)
        else:
            # Per-frame binary: independent probabilities per frame
            return torch.sigmoid(logits)

    def predict_distribution(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get boundary probability distribution (softmax of logits).

        This is the recommended prediction method when using distribution-based
        losses (dist_ce, kl_div, js_div, focal_dist_ce).

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Proprioception statistics (unused)
            dino_features: DINO features

        Returns:
            Probability distribution of shape (batch, prediction_window_size)
            where probabilities sum to 1 across the window dimension.

        Example:
            >>> dist = model.predict_distribution(tokens, None, dino)
            >>> # Find most likely boundary position
            >>> boundary_frame = dist.argmax(dim=1)
        """
        return self.predict_probabilities(
            tokenized_actions,
            proprio_stats,
            dino_features,
            use_softmax=True
        )

    def get_num_parameters(self) -> dict:
        """
        Get parameter counts for the model.

        Returns:
            Dictionary with parameter statistics
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        stats = {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }

        # Add per-component counts
        if self.use_proprio:
            stats['proprio_encoder'] = sum(
                p.numel() for p in self.proprio_encoder.parameters()
            )

        if self.use_dino:
            stats['dino_encoder'] = sum(
                p.numel() for p in self.dino_encoder.parameters()
            )

        stats['mlp'] = sum(p.numel() for p in self.mlp.parameters())

        return stats


def create_model(
    config: ModelConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BoundaryPredictionMLP:
    """
    Factory function to create and initialize a model.

    Args:
        config: Model configuration
        device: Device to place model on

    Returns:
        Initialized model on specified device

    Example:
        >>> config = ModelConfig(vocab_size=2048, hidden_dims=[512, 256])
        >>> model = create_model(config, device='cuda')
        >>> print(f"Parameters: {model.get_num_parameters()['total']:,}")
    """
    model = BoundaryPredictionMLP(config=config, device=device)
    model = model.to(device)
    return model


def collate_dict(batch: list) -> dict:
    """
    Collate function for DataLoader that handles dictionary batches.

    This function stacks dictionary values (tensors or numpy arrays) across
    a batch while preserving the dictionary structure.

    Args:
        batch: List of dictionaries with same keys

    Returns:
        Dictionary with stacked values

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MyDataset()
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_dict)
    """
    if not batch:
        return {}

    collated = {}

    for key in batch[0]:
        # Handle nested dictionaries recursively
        if isinstance(batch[0][key], dict):
            collated[key] = collate_dict([d[key] for d in batch])
            continue

        # Collect values for this key
        values = [d[key] for d in batch]

        # Stack based on type
        if isinstance(values[0], np.ndarray):
            # Convert numpy arrays to tensors and stack
            collated[key] = torch.stack([
                torch.from_numpy(v.copy()) for v in values
            ])
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors directly
            collated[key] = torch.stack(values)
        else:
            # Keep lists as-is for non-numeric data
            collated[key] = values

    return collated


def print_model_summary(model: BoundaryPredictionMLP):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: Model to summarize
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)

    # Configuration
    print("\nConfiguration:")
    print(f"  Use Proprio: {model.use_proprio}")
    print(f"  Use DINO: {model.use_dino}")
    print(f"  Vocab Size: {model.vocab_size}")
    print(f"  Token Embed Dim: {model.token_embed_dim}")
    print(f"  DINO Feature Dim: {model.dino_feature_dim}")
    print(f"  Prediction Window: {model.prediction_window_size}")
    print(f"  Device: {model.device}")

    # Parameters
    params = model.get_num_parameters()
    print("\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    if model.use_proprio:
        print(f"  Proprio Encoder: {params['proprio_encoder']:,}")
    if model.use_dino:
        print(f"  DINO Encoder: {params['dino_encoder']:,}")
    print(f"  MLP: {params['mlp']:,}")

    print("="*80 + "\n")
