"""Cross-modal transformer architecture for boundary prediction.

This module implements an enhanced architecture that uses:
1. Temporal modeling for visual features
2. Cross-modal attention between proprioceptive and visual features
3. Temporal transformer for sequence modeling
4. Better multi-modal fusion compared to simple concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

from .encoders import TinyFASTEncoder, DINO_MODEL_FEATURE_DIM


# Constants
DEFAULT_TOKEN_EMBED_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_EMBEDDING_DROPOUT = 0.1
DEFAULT_PAD_TOKEN_OFFSET = 1


@dataclass
class TransformerModelConfig:
    """Configuration for the cross-modal transformer model.

    Attributes:
        architecture: Architecture type identifier (always 'cross_modal_transformer')
        vocab_size: Tokenizer vocabulary size
        token_embed_dim: Token embedding dimension for proprioceptive features
        dino_feature_dim: DINO feature dimension (typically 3 frames × 1024)
        prediction_window_size: Number of frames to predict

        # Visual encoder settings
        visual_hidden_dim: Hidden dimension for visual temporal encoder
        visual_dropout: Dropout for visual encoder
        num_visual_frames: Number of DINO frames (default 3)

        # Cross-modal settings
        cross_modal_dim: Hidden dimension for cross-modal attention
        attention_dropout: Dropout for attention layers
        num_attention_heads: Number of attention heads

        # Temporal transformer settings
        temporal_layers: Number of temporal transformer layers
        temporal_dropout: Dropout for temporal transformer
        temporal_feedforward_dim: Feedforward dimension in temporal transformer

        # Output settings
        output_dropout: Dropout before final prediction
        use_cls_token: Whether to use CLS token for final prediction

        # Component flags
        use_proprio: Whether to use proprioceptive features
        use_dino: Whether to use DINO visual features

        # Original encoder settings (for TinyFASTEncoder)
        transformer_dropout: Dropout for proprioceptive transformer
        embedding_dropout: Dropout for embedding layers
    """

    # Architecture identifier
    architecture: str = 'cross_modal_transformer'

    # Vocabulary and embedding
    vocab_size: int = 2048
    token_embed_dim: int = 1024

    # DINO settings
    num_dino_frames: int = 3  # For compatibility with config schema
    dino_feature_dim: int = 3 * DINO_MODEL_FEATURE_DIM
    num_visual_frames: int = 3

    # MLP settings (unused in transformer, for compatibility with config schema)
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    dino_encoder_dropout: float = 0.3  # Unused
    mlp_dropout: float = 0.3  # Unused
    feature_noise_std: float = 0.05  # Unused

    # Visual encoder
    visual_hidden_dim: int = 256
    visual_dropout: float = 0.3

    # Cross-modal attention
    cross_modal_dim: int = 512
    attention_dropout: float = 0.1
    num_attention_heads: int = 8

    # Temporal transformer
    temporal_layers: int = 3
    temporal_dropout: float = 0.1
    temporal_feedforward_dim: int = 2048

    # Output
    prediction_window_size: int = 360
    output_dropout: float = 0.3
    use_cls_token: bool = True

    # Component flags
    use_proprio: bool = True
    use_dino: bool = True

    # Legacy settings for TinyFASTEncoder
    transformer_dropout: float = 0.3
    embedding_dropout: float = DEFAULT_EMBEDDING_DROPOUT

    def __post_init__(self):
        """Validate configuration."""
        if not (self.use_proprio or self.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        if self.dino_feature_dim != self.num_visual_frames * DINO_MODEL_FEATURE_DIM:
            raise ValueError(
                f"DINO feature dimension ({self.dino_feature_dim}) must equal "
                f"num_visual_frames ({self.num_visual_frames}) × {DINO_MODEL_FEATURE_DIM}"
            )


class TemporalVisualEncoder(nn.Module):
    """Temporal encoder for DINO visual features.

    Replaces the per-frame independent processing with temporal self-attention
    to model relationships between frames.

    Architecture:
        DINO Features → Reshape → L2 Norm → Linear Projection
        → + Positional Encoding → Temporal Self-Attention (2 layers)
        → Output: (sequence_features, pooled_features)
    """

    def __init__(self, config: TransformerModelConfig):
        super().__init__()

        self.num_frames = config.num_visual_frames
        self.dino_dim = DINO_MODEL_FEATURE_DIM
        self.hidden_dim = config.visual_hidden_dim

        # Project DINO features to hidden dimension
        self.frame_projection = nn.Linear(self.dino_dim, self.hidden_dim)

        # Learnable positional encoding for frames
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_frames, self.hidden_dim) * 0.02
        )

        # Temporal self-attention layers
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=config.visual_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            ) for _ in range(2)  # 2 layers for efficiency
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.visual_dropout)

    def forward(self, dino_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode visual features with temporal modeling.

        Args:
            dino_features: (batch, num_frames * 1024) concatenated DINO features

        Returns:
            sequence_features: (batch, num_frames, hidden_dim) for cross-attention
            pooled_features: (batch, hidden_dim) mean-pooled features
        """
        batch_size = dino_features.shape[0]

        # Reshape to (batch, num_frames, 1024)
        features = dino_features.view(batch_size, self.num_frames, self.dino_dim)

        # L2 normalize per frame (important for DINO features)
        features = F.normalize(features, p=2, dim=-1)

        # Project to hidden dimension
        x = self.frame_projection(features)  # (batch, num_frames, hidden_dim)
        x = self.dropout(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Apply temporal self-attention
        for layer in self.temporal_layers:
            x = layer(x)

        x = self.layer_norm(x)

        # Return both sequence and pooled representations
        sequence_features = x  # (batch, num_frames, hidden_dim)
        pooled_features = x.mean(dim=1)  # (batch, hidden_dim)

        return sequence_features, pooled_features


class CrossModalAttention(nn.Module):
    """Bi-directional cross-attention between proprioceptive and visual modalities.

    This module allows the model to learn joint representations by attending
    from one modality to another, rather than simple concatenation.

    Architecture:
        Proprio Features ↔ Cross-Attention ↔ Visual Sequence
        → Gated Fusion → Output Features
    """

    def __init__(self, config: TransformerModelConfig):
        super().__init__()

        self.proprio_dim = config.token_embed_dim
        self.visual_dim = config.visual_hidden_dim
        self.hidden_dim = config.cross_modal_dim

        # Project to common dimension
        self.proprio_projection = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Bi-directional cross-attention
        self.proprio_to_visual = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        self.visual_to_proprio = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # Gating mechanism for fusion
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # Output projection with feedforward network
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

    def forward(self,
                proprio_features: torch.Tensor,
                visual_sequence: torch.Tensor) -> torch.Tensor:
        """Fuse proprioceptive and visual features via cross-attention.

        Args:
            proprio_features: (batch, proprio_dim) pooled proprio features
            visual_sequence: (batch, num_frames, visual_dim) visual sequence

        Returns:
            fused_features: (batch, num_frames, hidden_dim) fused multi-modal features
        """
        batch_size = proprio_features.shape[0]

        # Project to common dimension
        proprio_proj = self.proprio_projection(proprio_features)  # (batch, hidden_dim)
        visual_proj = self.visual_projection(visual_sequence)  # (batch, num_frames, hidden_dim)

        # Expand proprio to sequence for attention
        proprio_seq = proprio_proj.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Bi-directional cross-attention
        # Proprio attending to visual (what visual info is relevant for proprio?)
        proprio_attended, _ = self.proprio_to_visual(
            query=proprio_seq,
            key=visual_proj,
            value=visual_proj
        )  # (batch, 1, hidden_dim)

        # Visual attending to proprio (how does proprio context affect visual?)
        visual_attended, _ = self.visual_to_proprio(
            query=visual_proj,
            key=proprio_seq,
            value=proprio_seq
        )  # (batch, num_frames, hidden_dim)

        # Expand proprio_attended to match visual sequence length
        proprio_expanded = proprio_attended.expand(-1, visual_proj.shape[1], -1)

        # Gated fusion: learn to balance proprio vs visual information
        concat_features = torch.cat([proprio_expanded, visual_attended], dim=-1)
        gate_weights = self.gate(concat_features)

        fused = gate_weights * proprio_expanded + (1 - gate_weights) * visual_attended

        # Apply residual connection and feedforward
        fused = fused + visual_proj  # Residual from visual
        output = self.output_projection(fused)  # (batch, num_frames, hidden_dim)
        output = output + fused  # Residual connection

        return output


class TemporalTransformer(nn.Module):
    """Temporal transformer for processing fused multi-modal features.

    This module processes the fused features over time to capture temporal
    dependencies for boundary prediction.

    Architecture:
        Fused Features → + Positional Encoding → + CLS Token (optional)
        → Transformer Encoder Layers → Output Features
    """

    def __init__(self, config: TransformerModelConfig):
        super().__init__()

        self.hidden_dim = config.cross_modal_dim
        self.num_layers = config.temporal_layers
        self.use_cls_token = config.use_cls_token

        # Learnable CLS token for global representation
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

        # Temporal transformer layers
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.temporal_feedforward_dim,
                dropout=config.temporal_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            ),
            num_layers=self.num_layers
        )

        # Window-aware sinusoidal positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_sinusoidal_encoding(100, self.hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Process fused features temporally.

        Args:
            fused_features: (batch, seq_len, hidden_dim) fused features

        Returns:
            temporal_features: (batch, seq_len+1, hidden_dim) if use_cls_token
                             (batch, seq_len, hidden_dim) otherwise
        """
        batch_size, seq_len = fused_features.shape[:2]

        # Add positional encoding
        x = fused_features + self.pos_encoding[:, :seq_len, :]

        # Prepend CLS token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Apply temporal transformer
        output = self.temporal_encoder(x)
        output = self.layer_norm(output)

        return output


class BoundaryPredictionTransformer(nn.Module):
    """Cross-modal transformer for boundary prediction.

    This model improves upon the baseline MLP by:
    1. Adding temporal modeling to visual features
    2. Using cross-modal attention instead of concatenation
    3. Processing fused features with a temporal transformer

    Architecture:
        Tokenized Actions → TinyFASTEncoder → Proprio Features
        DINO Features → TemporalVisualEncoder → Visual Sequence
        (Proprio, Visual) → CrossModalAttention → Fused Features
        Fused Features → TemporalTransformer → Temporal Features
        Temporal Features → Prediction Head → Boundary Predictions

    Attributes:
        vocab_size: Total vocabulary size
        prediction_window_size: Number of frames to predict
        use_proprio: Whether proprioceptive features are used
        use_dino: Whether visual features are used
        device: Device to run model on
    """

    def __init__(
        self,
        config: TransformerModelConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize cross-modal transformer model.

        Args:
            config: Model configuration
            device: Device to run model on

        Example:
            >>> config = TransformerModelConfig()
            >>> model = BoundaryPredictionTransformer(config)
            >>> tokens = torch.randint(0, 2048, (32, 100))
            >>> dino = torch.randn(32, 3072)
            >>> logits = model(tokens, None, dino)
            >>> print(logits.shape)  # (32, 360)
        """
        super().__init__()

        # Validate configuration
        if not (config.use_proprio or config.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        # Store configuration
        self.vocab_size = config.vocab_size + DEFAULT_PAD_TOKEN_OFFSET
        self.prediction_window_size = config.prediction_window_size
        self.use_proprio = config.use_proprio
        self.use_dino = config.use_dino
        self.use_cls_token = config.use_cls_token
        self.device = device

        # Proprioceptive encoder (keep existing architecture)
        if config.use_proprio:
            self.proprio_encoder = TinyFASTEncoder(
                vocab_size=self.vocab_size,
                embed_dim=config.token_embed_dim,
                num_heads=DEFAULT_NUM_HEADS,
                transformer_dropout=config.transformer_dropout,
                embedding_dropout=config.embedding_dropout
            )

        # Temporal visual encoder (new)
        if config.use_dino:
            self.visual_encoder = TemporalVisualEncoder(config)

        # Cross-modal attention (new)
        if config.use_proprio and config.use_dino:
            self.cross_modal = CrossModalAttention(config)

        # Temporal transformer (new)
        self.temporal_transformer = TemporalTransformer(config)

        # Prediction head
        self.boundary_head = nn.Sequential(
            nn.Linear(config.cross_modal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.output_dropout),
            nn.Linear(256, config.prediction_window_size)
        )

    def forward(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            tokenized_actions: (batch, seq_len) token IDs
            proprio_stats: Unused, kept for API compatibility
            dino_features: (batch, num_frames * 1024) DINO features

        Returns:
            logits: (batch, prediction_window_size) boundary predictions

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate inputs
        if self.use_proprio and tokenized_actions is None:
            raise ValueError("tokenized_actions required when use_proprio=True")
        if self.use_dino and dino_features is None:
            raise ValueError("dino_features required when use_dino=True")

        # Encode proprioceptive features
        if self.use_proprio:
            proprio_features = self.proprio_encoder(tokenized_actions)

        # Encode visual features with temporal modeling
        if self.use_dino:
            visual_sequence, visual_pooled = self.visual_encoder(dino_features)

        # Cross-modal fusion
        if self.use_proprio and self.use_dino:
            fused_features = self.cross_modal(proprio_features, visual_sequence)
        elif self.use_proprio:
            # Proprio only: expand to sequence
            fused_features = proprio_features.unsqueeze(1).expand(-1, 3, -1)
        else:
            # Visual only
            fused_features = visual_sequence

        # Temporal transformer processing
        temporal_features = self.temporal_transformer(fused_features)

        # Use CLS token for prediction or pool all tokens
        if self.use_cls_token:
            output_features = temporal_features[:, 0, :]  # (batch, hidden_dim)
        else:
            output_features = temporal_features.mean(dim=1)  # (batch, hidden_dim)

        # Generate boundary predictions
        predictions = self.boundary_head(output_features)

        return predictions

    def predict_probabilities(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor],
        use_softmax: bool = False
    ) -> torch.Tensor:
        """Get boundary probabilities from logits.

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Unused
            dino_features: DINO features
            use_softmax: If True, apply softmax. If False, apply sigmoid.

        Returns:
            Probabilities of shape (batch, prediction_window_size)
        """
        logits = self.forward(tokenized_actions, proprio_stats, dino_features)

        if use_softmax:
            return torch.softmax(logits, dim=1)
        else:
            return torch.sigmoid(logits)

    def predict_distribution(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get boundary probability distribution (softmax of logits).

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Unused
            dino_features: DINO features

        Returns:
            Probability distribution of shape (batch, prediction_window_size)
        """
        return self.predict_probabilities(
            tokenized_actions,
            proprio_stats,
            dino_features,
            use_softmax=True
        )

    def get_num_parameters(self) -> dict:
        """Get parameter counts for the model.

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

        # Per-component counts
        if self.use_proprio:
            stats['proprio_encoder'] = sum(
                p.numel() for p in self.proprio_encoder.parameters()
            )

        if self.use_dino:
            stats['visual_encoder'] = sum(
                p.numel() for p in self.visual_encoder.parameters()
            )

        if self.use_proprio and self.use_dino:
            stats['cross_modal'] = sum(
                p.numel() for p in self.cross_modal.parameters()
            )

        stats['temporal_transformer'] = sum(
            p.numel() for p in self.temporal_transformer.parameters()
        )

        stats['boundary_head'] = sum(
            p.numel() for p in self.boundary_head.parameters()
        )

        return stats


def create_transformer_model(
    config: TransformerModelConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BoundaryPredictionTransformer:
    """Factory function to create transformer model.

    Args:
        config: Model configuration
        device: Device to place model on

    Returns:
        Initialized model on specified device
    """
    model = BoundaryPredictionTransformer(config=config, device=device)
    model = model.to(device)
    return model
