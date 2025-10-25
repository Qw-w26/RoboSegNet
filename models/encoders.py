"""Encoder modules for multi-modal feature processing.

This module provides encoder architectures for processing proprioceptive
tokens and visual features for boundary prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants
DINO_MODEL_FEATURE_DIM = 1024
DEFAULT_OUTPUT_DIM_PER_FRAME = 32


class TinyFASTEncoder(nn.Module):
    """
    Encoder for Fast-DROID tokenized proprioceptive sequences.

    Uses an embedding layer + single-layer transformer to encode variable-length
    token sequences into fixed-dimensional representations. Proper masking ensures
    padding tokens are ignored during attention and pooling.

    Architecture:
        Token IDs → Embedding → Transformer Encoder → Mean Pooling (non-pad)

    Attributes:
        pad_token_idx: Index of padding token in vocabulary
        embedding: Token embedding layer
        embedding_dropout: Dropout after embedding
        transformer: Single-layer transformer encoder
        output_dim: Dimension of output features
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        transformer_dropout: float = 0.3,
        embedding_dropout: float = 0.1
    ):
        """
        Initialize encoder.

        Args:
            vocab_size: Size of token vocabulary (includes padding token)
            embed_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            transformer_dropout: Dropout probability in transformer
            embedding_dropout: Dropout probability after embedding

        Example:
            >>> encoder = TinyFASTEncoder(vocab_size=2049, embed_dim=128)
            >>> tokens = torch.randint(0, 2048, (32, 100))  # batch_size=32, seq_len=100
            >>> output = encoder(tokens)  # (32, 128)
        """
        super().__init__()

        # Padding token is last in vocabulary
        self.pad_token_idx = vocab_size - 1

        # Token embedding with padding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.pad_token_idx
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Single-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=transformer_dropout,
            batch_first=True  # Input/output format: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.output_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token sequences.

        Args:
            token_ids: Token ID tensor of shape (batch_size, seq_len)

        Returns:
            Encoded features of shape (batch_size, embed_dim)

        Raises:
            AssertionError: If token IDs are out of vocabulary range
        """
        # Validate token IDs
        assert torch.max(token_ids) < self.embedding.num_embeddings, \
            f"Max token ID {torch.max(token_ids)} exceeds vocab size {self.embedding.num_embeddings}"
        assert torch.min(token_ids) >= 0, \
            f"Min token ID {torch.min(token_ids)} is negative"

        # Embed tokens
        x = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        x = self.embedding_dropout(x)

        # Create padding mask (True where padding)
        key_padding_mask = (token_ids == self.pad_token_idx)

        # Apply transformer with masking
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Pool over non-padding tokens only
        # Count non-padding tokens per sequence
        lengths = (~key_padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # (batch, 1)

        # Zero out padding positions and sum
        x_masked = x * (~key_padding_mask).unsqueeze(-1)  # (batch, seq_len, embed_dim)
        x_pooled = x_masked.sum(dim=1) / lengths  # (batch, embed_dim)

        return x_pooled


class DinoFeatureEncoder(nn.Module):
    """
    Encoder for DINOv3 visual features.

    Processes concatenated multi-frame DINO features with normalization,
    augmentation, and per-frame encoding.

    Architecture:
        DINO features → Reshape → L2 Normalize → Noise (train) →
        Per-frame Linear+ReLU+Dropout → Flatten

    Attributes:
        encoder: Per-frame encoder network
        num_frames: Number of frames in input
        output_dim: Dimension of output features
        dropout: Dropout probability
        feature_noise_std: Standard deviation for training-time noise
    """

    def __init__(
        self,
        feature_dim: int = 3072,  # 3 frames × 1024
        dino_encoder_dropout: float = 0.3,
        output_dim_per_frame: int = DEFAULT_OUTPUT_DIM_PER_FRAME,
        feature_noise_std: float = 0.05
    ):
        """
        Initialize encoder.

        Args:
            feature_dim: Total DINO feature dimension (num_frames × 1024)
            dino_encoder_dropout: Dropout probability
            output_dim_per_frame: Output dimension per frame
            feature_noise_std: Std of Gaussian noise added during training

        Example:
            >>> encoder = DinoFeatureEncoder(feature_dim=3072)
            >>> dino_feats = torch.randn(32, 3072)  # 32 samples, 3 frames
            >>> output = encoder(dino_feats)  # (32, 96) = 3 frames × 32 dims
        """
        super().__init__()

        # Validate feature dimension
        assert feature_dim % DINO_MODEL_FEATURE_DIM == 0, \
            f"Feature dim ({feature_dim}) must be multiple of {DINO_MODEL_FEATURE_DIM}"

        self.num_frames = feature_dim // DINO_MODEL_FEATURE_DIM

        # Per-frame encoder
        self.encoder = nn.Sequential(
            nn.Linear(DINO_MODEL_FEATURE_DIM, output_dim_per_frame),
            nn.ReLU(),
            nn.Dropout(dino_encoder_dropout)
        )

        self.output_dim = output_dim_per_frame * self.num_frames
        self.dropout = dino_encoder_dropout
        self.feature_noise_std = feature_noise_std

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode DINO features.

        Args:
            features: DINO features of shape (batch, num_frames * 1024)

        Returns:
            Encoded features of shape (batch, num_frames * output_dim_per_frame)
        """
        batch_size = features.size(0)

        # Reshape to (batch, num_frames, 1024)
        features = features.view(batch_size, self.num_frames, DINO_MODEL_FEATURE_DIM)

        # L2 normalize features for stability
        features = F.normalize(features, p=2, dim=-1)

        # Training-time augmentation
        if self.training:
            # Add Gaussian noise
            features = features + torch.randn_like(features) * self.feature_noise_std

            # Random dropout of features (set to zero)
            dropout_mask = torch.rand_like(features) < self.dropout
            features = features.masked_fill(dropout_mask, 0)

        # Encode each frame
        encoded = self.encoder(features)  # (batch, num_frames, output_dim_per_frame)

        # Flatten to (batch, num_frames * output_dim_per_frame)
        encoded = encoded.view(batch_size, -1)

        return encoded


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts

    Example:
        >>> encoder = TinyFASTEncoder(vocab_size=2049)
        >>> stats = count_parameters(encoder)
        >>> print(f"Total: {stats['total']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
