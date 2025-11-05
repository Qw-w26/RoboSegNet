"""DETR-style boundary detection model.

This module implements a Detection Transformer (DETR) inspired architecture
for boundary detection in robotic manipulation tasks. Instead of dense
per-frame predictions, this model directly predicts a set of boundaries
using learned queries and Hungarian matching.

Key components:
- TrajectoryEncoder: Encodes temporal features with bidirectional attention
- BoundaryQueryDecoder: Learned queries that attend to trajectory to find boundaries
- BoundaryPredictionHeads: Predict position and confidence for each query
- SetCriterion: Hungarian matching loss for set-based supervision

Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math

from .encoders import TinyFASTEncoder, DINO_MODEL_FEATURE_DIM


@dataclass
class BoundaryDETRConfig:
    """Configuration for DETR-style boundary detection model.

    Attributes:
        # Architecture
        architecture: Model architecture identifier
        vocab_size: Tokenizer vocabulary size
        token_embed_dim: Token embedding dimension for proprioceptive

        # Visual encoding (reuse existing)
        num_dino_frames: Number of DINO frames
        dino_feature_dim: DINO feature dimension
        visual_hidden_dim: Hidden dimension for visual encoder
        visual_dropout: Dropout for visual encoder

        # Cross-modal (reuse existing)
        cross_modal_dim: Hidden dimension for cross-modal attention
        attention_dropout: Dropout for attention
        num_attention_heads: Number of attention heads

        # DETR-specific
        d_model: Transformer hidden dimension
        num_encoder_layers: Number of trajectory encoder layers
        num_decoder_layers: Number of boundary decoder layers
        num_boundary_queries: Number of boundary queries (max boundaries)
        detr_dropout: Dropout for DETR transformer
        detr_feedforward_dim: Feedforward dimension in DETR transformer

        # Prediction
        prediction_window_size: Window size in frames

        # Component flags
        use_proprio: Whether to use proprioceptive features
        use_dino: Whether to use DINO visual features

        # Legacy (for compatibility with existing encoders)
        transformer_dropout: Dropout for TinyFAST
        embedding_dropout: Dropout for embeddings
    """

    # Architecture
    architecture: str = 'boundary_detr'
    vocab_size: int = 2048
    token_embed_dim: int = 1024

    # Visual encoding
    num_dino_frames: int = 6
    dino_feature_dim: int = 6 * DINO_MODEL_FEATURE_DIM
    visual_hidden_dim: int = 256
    visual_dropout: float = 0.3

    # Cross-modal
    cross_modal_dim: int = 512
    attention_dropout: float = 0.1
    num_attention_heads: int = 8

    # DETR-specific
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_boundary_queries: int = 9  # Max 9 boundaries per window
    detr_dropout: float = 0.1
    detr_feedforward_dim: int = 2048

    # Prediction
    prediction_window_size: int = 18

    # Component flags
    use_proprio: bool = True
    use_dino: bool = True

    # Legacy
    transformer_dropout: float = 0.3
    embedding_dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration."""
        if not (self.use_proprio or self.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        if self.dino_feature_dim != self.num_dino_frames * DINO_MODEL_FEATURE_DIM:
            raise ValueError(
                f"DINO feature dimension ({self.dino_feature_dim}) must equal "
                f"num_dino_frames ({self.num_dino_frames}) × {DINO_MODEL_FEATURE_DIM}"
            )


class TrajectoryEncoder(nn.Module):
    """Encode temporal features with bidirectional transformer.

    This encoder processes the fused multi-modal features (from cross-modal
    attention) and adds full temporal context using bidirectional self-attention.
    Unlike causal models, this sees the full sequence at once.

    Architecture:
        Input Features → Projection → + Positional Encoding
        → Bidirectional Transformer (6 layers) → Encoded Trajectory
    """

    def __init__(self, config: BoundaryDETRConfig):
        super().__init__()

        self.d_model = config.d_model
        self.max_seq_len = 200  # Support up to 200 frames

        # Project input features to d_model
        self.input_proj = nn.Linear(config.cross_modal_dim, self.d_model)

        # Fixed sinusoidal positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(self.max_seq_len, self.d_model)
        )

        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.detr_feedforward_dim,
            dropout=config.detr_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create fixed sinusoidal positional encoding.

        Uses the standard sinusoidal encoding from "Attention is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode trajectory features.

        Args:
            features: (batch, seq_len, feature_dim) input features
            mask: (batch, seq_len) optional padding mask (True = padding)

        Returns:
            encoded: (batch, seq_len, d_model) encoded trajectory
        """
        batch_size, seq_len, _ = features.shape

        # Project to d_model
        x = self.input_proj(features)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)

        # Bidirectional encoding (no causal mask)
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)

        return encoded


class BoundaryQueryDecoder(nn.Module):
    """DETR-style decoder with learned boundary queries.

    This decoder uses learned query embeddings that "ask" about boundaries.
    Queries attend to the encoded trajectory via cross-attention to localize
    boundaries. Self-attention between queries helps avoid duplicate predictions.

    Key concepts:
    - Query embeddings: Learned vectors that specialize to detect different boundaries
    - Query positional embeddings: Help queries specialize (e.g., query 0 for early boundaries)
    - Cross-attention: Queries attend to trajectory to find boundaries
    - Self-attention: Queries communicate to avoid overlaps
    """

    def __init__(self, config: BoundaryDETRConfig):
        super().__init__()

        self.d_model = config.d_model
        self.num_queries = config.num_boundary_queries

        # Learned query embeddings (content)
        # These are randomly initialized and learned during training
        self.query_embed = nn.Embedding(self.num_queries, self.d_model)

        # Learned query positional embeddings
        # These provide positional priors for queries
        self.query_pos = nn.Embedding(self.num_queries, self.d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.detr_feedforward_dim,
            dropout=config.detr_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )

    def forward(
        self,
        encoded_trajectory: torch.Tensor,
        trajectory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode boundary queries by attending to trajectory.

        Args:
            encoded_trajectory: (batch, seq_len, d_model) from encoder
            trajectory_mask: (batch, seq_len) padding mask

        Returns:
            decoded_queries: (batch, num_queries, d_model)
        """
        batch_size = encoded_trajectory.shape[0]

        # Get query embeddings and expand for batch
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine content and positional embeddings
        # Note: In DETR, these are added. Alternative: concatenate
        tgt = query_embed + query_pos

        # Decode: queries attend to encoded trajectory
        decoded_queries = self.transformer_decoder(
            tgt=tgt,
            memory=encoded_trajectory,
            memory_key_padding_mask=trajectory_mask
        )

        return decoded_queries


class BoundaryPredictionHeads(nn.Module):
    """Prediction heads for boundary position and confidence.

    Given decoded query features, predict:
    1. Position: Normalized position [0, 1] in the window
    2. Confidence: Probability that this query detected a real boundary
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()

        # Position head: predict normalized position [0, 1]
        self.position_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Ensure output in [0, 1]
        )

        # Confidence head: probability of real boundary
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Probability in [0, 1]
        )

    def forward(self, decoded_queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict positions and confidences.

        Args:
            decoded_queries: (batch, num_queries, d_model)

        Returns:
            positions: (batch, num_queries) in [0, 1]
            confidences: (batch, num_queries) in [0, 1]
        """
        positions = self.position_head(decoded_queries).squeeze(-1)
        confidences = self.confidence_head(decoded_queries).squeeze(-1)

        return positions, confidences


class BoundaryDETR(nn.Module):
    """Complete DETR-style boundary detection model.

    This model integrates existing feature encoders (proprioceptive + visual)
    with DETR components (trajectory encoder, query decoder, prediction heads)
    to directly predict a set of boundaries.

    Architecture flow:
        1. Feature Extraction (reuse existing encoders)
           - TinyFASTEncoder for proprioceptive
           - TemporalVisualEncoder for DINO
           - CrossModalAttention for fusion

        2. Trajectory Encoding (NEW)
           - Bidirectional transformer over fused features

        3. Boundary Query Decoding (NEW)
           - Learned queries attend to trajectory

        4. Prediction (NEW)
           - Position and confidence for each query

    Output:
        Set of N boundary predictions: {(position_i, confidence_i)} for i=1..N
        where N = num_boundary_queries
    """

    def __init__(
        self,
        config: BoundaryDETRConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        # Store config
        self.vocab_size = config.vocab_size
        self.num_queries = config.num_boundary_queries
        self.window_size = config.prediction_window_size
        self.use_proprio = config.use_proprio
        self.use_dino = config.use_dino
        self.device = device

        # Reuse existing encoders (from baseline model)
        if config.use_proprio:
            from .encoders import TinyFASTEncoder
            self.proprio_encoder = TinyFASTEncoder(
                vocab_size=config.vocab_size + 1,  # +1 for padding
                embed_dim=config.token_embed_dim,
                num_heads=4,
                transformer_dropout=config.transformer_dropout,
                embedding_dropout=config.embedding_dropout
            )

        if config.use_dino:
            from .boundary_prediction_transformer import TemporalVisualEncoder
            self.visual_encoder = TemporalVisualEncoder(
                type('Config', (), {
                    'num_visual_frames': config.num_dino_frames,
                    'visual_hidden_dim': config.visual_hidden_dim,
                    'visual_dropout': config.visual_dropout
                })()
            )

        # Cross-modal fusion (reuse if both modalities present)
        if config.use_proprio and config.use_dino:
            from .boundary_prediction_transformer import CrossModalAttention
            self.cross_modal = CrossModalAttention(
                type('Config', (), {
                    'token_embed_dim': config.token_embed_dim,
                    'visual_hidden_dim': config.visual_hidden_dim,
                    'cross_modal_dim': config.cross_modal_dim,
                    'num_attention_heads': config.num_attention_heads,
                    'attention_dropout': config.attention_dropout
                })()
            )

        # DETR components (NEW)
        self.trajectory_encoder = TrajectoryEncoder(config)
        self.boundary_decoder = BoundaryQueryDecoder(config)
        self.prediction_heads = BoundaryPredictionHeads(
            d_model=config.d_model,
            dropout=config.detr_dropout
        )

    def forward(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through DETR model.

        Args:
            tokenized_actions: (batch, seq_len) token IDs
            proprio_stats: Unused, kept for API compatibility
            dino_features: (batch, num_frames * 1024) concatenated DINO features

        Returns:
            Dictionary with:
                - 'pred_positions': (batch, num_queries) normalized positions [0, 1]
                - 'pred_confidences': (batch, num_queries) confidence scores [0, 1]
        """
        # Validate inputs
        if self.use_proprio and tokenized_actions is None:
            raise ValueError("tokenized_actions required when use_proprio=True")
        if self.use_dino and dino_features is None:
            raise ValueError("dino_features required when use_dino=True")

        # 1. Feature encoding (reuse existing pipeline)
        if self.use_proprio:
            proprio_features = self.proprio_encoder(tokenized_actions)
            # proprio_features: (batch, hidden_dim)

        if self.use_dino:
            visual_sequence, _ = self.visual_encoder(dino_features)
            # visual_sequence: (batch, num_visual_frames, hidden_dim)

        # 2. Cross-modal fusion
        if self.use_proprio and self.use_dino:
            fused_features = self.cross_modal(proprio_features, visual_sequence)
            # fused_features: (batch, num_visual_frames, cross_modal_dim)
        elif self.use_proprio:
            # Proprio only: expand to sequence
            fused_features = proprio_features.unsqueeze(1)
        else:
            # Visual only: use visual features directly
            # But need to project to cross_modal_dim
            fused_features = visual_sequence

        # 3. Trajectory encoding (bidirectional context)
        encoded_trajectory = self.trajectory_encoder(fused_features)
        # encoded_trajectory: (batch, seq_len, d_model)

        # 4. Boundary query decoding
        decoded_queries = self.boundary_decoder(encoded_trajectory)
        # decoded_queries: (batch, num_queries, d_model)

        # 5. Predict positions and confidences
        positions, confidences = self.prediction_heads(decoded_queries)
        # positions: (batch, num_queries) in [0, 1]
        # confidences: (batch, num_queries) in [0, 1]

        return {
            'pred_positions': positions,
            'pred_confidences': confidences
        }

    def predict_boundaries(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> List[List[int]]:
        """Inference: extract boundary frame indices from predictions.

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Unused
            dino_features: DINO features
            confidence_threshold: Minimum confidence to keep prediction

        Returns:
            List of boundary frame lists (one per batch sample)
        """
        predictions = self.forward(tokenized_actions, proprio_stats, dino_features)

        positions = predictions['pred_positions']  # (batch, num_queries)
        confidences = predictions['pred_confidences']

        boundaries_batch = []

        for b in range(positions.shape[0]):
            # Filter by confidence threshold
            valid_mask = confidences[b] > confidence_threshold
            valid_positions = positions[b][valid_mask]

            if len(valid_positions) == 0:
                boundaries_batch.append([])
                continue

            # Convert normalized positions to frame indices
            frame_indices = (valid_positions * self.window_size).long()

            # Sort temporally and convert to list
            frame_indices = sorted(frame_indices.cpu().tolist())

            boundaries_batch.append(frame_indices)

        return boundaries_batch

    def get_num_parameters(self) -> Dict[str, int]:
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

        stats['trajectory_encoder'] = sum(
            p.numel() for p in self.trajectory_encoder.parameters()
        )

        stats['boundary_decoder'] = sum(
            p.numel() for p in self.boundary_decoder.parameters()
        )

        stats['prediction_heads'] = sum(
            p.numel() for p in self.prediction_heads.parameters()
        )

        return stats


def create_boundary_detr(
    config: BoundaryDETRConfig,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BoundaryDETR:
    """Factory function to create DETR model.

    Args:
        config: Model configuration
        device: Device to place model on

    Returns:
        Initialized model on specified device
    """
    model = BoundaryDETR(config=config, device=device)
    model = model.to(device)
    return model
