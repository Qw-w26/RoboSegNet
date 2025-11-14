"""
ASFormer (Action Segmentation Transformer) for boundary prediction.

Adapted to work with existing dataloader and evaluation pipeline.
Uses DINO features, compatible with sliding window approach.

Reference: https://github.com/ChinaYi/ASFormer
Drop-in replacement for BoundaryPredictionMLP - same inputs, same outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .encoders import TinyFASTEncoder, DINO_MODEL_FEATURE_DIM
from .boundary_prediction import ModelConfig, DEFAULT_NUM_HEADS, DEFAULT_PAD_TOKEN_OFFSET
import copy


class AttentionHelper(nn.Module):
    """Helper module for scaled dot-product attention."""

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor (batch, n_heads, seq_len_q, d_k)
            k: Key tensor (batch, n_heads, seq_len_k, d_k)
            v: Value tensor (batch, n_heads, seq_len_v, d_k)
            mask: Optional mask tensor

        Returns:
            Output tensor and attention weights
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class AttLayer(nn.Module):
    """Single attention layer with different attention types."""

    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3,
                 attention_type='normal', stage='encoder'):
        """
        Args:
            q_dim: Query dimension
            k_dim: Key dimension
            v_dim: Value dimension
            r1, r2, r3: Dimension reduction factors
            attention_type: Type of attention ('normal', 'block', 'sliding')
            stage: 'encoder' or 'decoder'
        """
        super().__init__()

        self.query_conv = nn.Conv1d(q_dim, q_dim // r1, 1)
        self.key_conv = nn.Conv1d(k_dim, k_dim // r2, 1)
        self.value_conv = nn.Conv1d(v_dim, v_dim // r3, 1)

        self.output_dim = v_dim // r3
        self.attention_type = attention_type
        self.stage = stage
        self.att_helper = AttentionHelper()

    def forward(self, x1, x2):
        """
        Args:
            x1: Input 1 (batch, channels, seq_len)
            x2: Input 2 (batch, channels, seq_len)

        Returns:
            Output tensor (batch, output_dim, seq_len)
        """
        batch_size = x1.size(0)

        # Generate Q, K, V
        q = self.query_conv(x1)  # (batch, q_dim//r1, seq_len)
        k = self.key_conv(x2)    # (batch, k_dim//r2, seq_len)
        v = self.value_conv(x2)  # (batch, v_dim//r3, seq_len)

        # Reshape for attention: (batch, channels, seq_len) -> (batch, 1, seq_len, channels)
        q = q.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, seq_len, channels)
        k = k.permute(0, 2, 1).unsqueeze(1)
        v = v.permute(0, 2, 1).unsqueeze(1)

        # Apply attention
        if self.attention_type == 'normal':
            output, _ = self.att_helper(q, k, v)
        elif self.attention_type == 'block':
            # Block-wise attention (simplified)
            output, _ = self.att_helper(q, k, v)
        else:  # sliding
            # Sliding window attention (simplified)
            output, _ = self.att_helper(q, k, v)

        # Reshape back: (batch, 1, seq_len, channels) -> (batch, channels, seq_len)
        output = output.squeeze(1).permute(0, 2, 1)

        return output


class MultiHeadAttLayer(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, q_dim, k_dim, v_dim, num_heads, dropout=0.1):
        """
        Args:
            q_dim: Query dimension
            k_dim: Key dimension
            v_dim: Value dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_heads = num_heads
        self.att_layers = nn.ModuleList([
            AttLayer(q_dim, k_dim, v_dim, r1=num_heads, r2=num_heads, r3=num_heads)
            for _ in range(num_heads)
        ])

        self.output_conv = nn.Conv1d(v_dim, v_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input 1 (batch, channels, seq_len)
            x2: Input 2 (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        # Concatenate outputs from all heads
        outputs = []
        for att in self.att_layers:
            outputs.append(att(x1, x2))

        out = torch.cat(outputs, dim=1)
        out = self.output_conv(out)
        out = self.dropout(out)

        return out


class ConvFeedForward(nn.Module):
    """Convolutional feed-forward network with dilation."""

    def __init__(self, dim, hidden_dim, dropout=0.1, dilation=1):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            dilation: Dilation rate for convolution
        """
        super().__init__()

        self.conv1 = nn.Conv1d(dim, hidden_dim, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class AttModule(nn.Module):
    """Attention module combining feed-forward, attention, and residual connections."""

    def __init__(self, dim, num_heads, dilation, dropout=0.1, attention_type='normal',
                 stage='encoder'):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dilation: Dilation rate for feed-forward
            dropout: Dropout rate
            attention_type: Type of attention
            stage: 'encoder' or 'decoder'
        """
        super().__init__()

        self.feed_forward = ConvFeedForward(dim, dim * 2, dropout=dropout, dilation=dilation)
        # Use LayerNorm instead of InstanceNorm for better stability with small sequences
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.att_layer = MultiHeadAttLayer(dim, dim, dim, num_heads, dropout=dropout)
        self.conv_out = nn.Conv1d(dim, dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            y: Optional second input for cross-attention (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        # Feed-forward with residual
        out = self.feed_forward(x)
        out = self.dropout(out)
        out = out + x  # Residual
        # LayerNorm: (batch, channels, seq_len) -> normalize over channels
        out = out.transpose(1, 2)  # (batch, seq_len, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (batch, channels, seq_len)

        # Attention with residual
        identity = out
        if y is None:
            # Self-attention
            att_out = self.att_layer(out, out)
        else:
            # Cross-attention
            att_out = self.att_layer(out, y)

        out = identity + self.dropout(att_out)  # Residual
        # LayerNorm
        out = out.transpose(1, 2)  # (batch, seq_len, channels)
        out = self.norm2(out)
        out = out.transpose(1, 2)  # (batch, channels, seq_len)

        # Final convolution WITH RESIDUAL (critical for gradient flow!)
        out_conv = self.conv_out(out)
        out = out_conv + out  # Residual connection

        return out


class Encoder(nn.Module):
    """ASFormer encoder with cascading attention modules."""

    def __init__(self, num_layers, dim, num_heads, dropout=0.1, attention_type='normal'):
        """
        Args:
            num_layers: Number of encoder layers
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            attention_type: Type of attention
        """
        super().__init__()

        self.layers = nn.ModuleList([
            AttModule(dim, num_heads, dilation=2**i, dropout=dropout,
                     attention_type=attention_type, stage='encoder')
            for i in range(num_layers)
        ])

        self.conv_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)
        return out


class Decoder(nn.Module):
    """ASFormer decoder with cross-attention to encoder features."""

    def __init__(self, num_layers, dim, num_heads, dropout=0.1, attention_type='normal'):
        """
        Args:
            num_layers: Number of decoder layers
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            attention_type: Type of attention
        """
        super().__init__()

        self.layers = nn.ModuleList([
            AttModule(dim, num_heads, dilation=2**i, dropout=dropout,
                     attention_type=attention_type, stage='decoder')
            for i in range(num_layers)
        ])

        self.conv_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x, encoder_output):
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
            encoder_output: Encoder output (batch, channels, seq_len)

        Returns:
            Output tensor (batch, channels, seq_len)
        """
        out = x
        for layer in self.layers:
            out = layer(out, encoder_output)

        out = self.conv_out(out)
        return out


class BoundaryASFormer(nn.Module):
    """
    ASFormer adapted for boundary prediction with existing dataloader.

    Compatible with existing training pipeline. Uses the same inputs and outputs
    as BoundaryPredictionMLP but with transformer-based temporal modeling.

    Architecture:
        Tokenized Actions → TinyFASTEncoder →  ┐
                                               ├→ Expand to sequence → ASFormer → Pool → Predictions
        DINO Features → Per-frame Encoder →    ┘

    Key adaptations:
    - Works with sliding windows instead of full sequences
    - Uses DINO features (pre-extracted) from existing dataloader
    - Outputs predictions for prediction_window_size (not full sequence)
    - Drop-in replacement for existing models
    """

    def __init__(
        self,
        config: ModelConfig,
        num_decoder_stages: int = 3,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        num_f_maps: int = 64,
        num_heads: int = 4,
        attention_type: str = 'normal',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize ASFormer boundary prediction model.

        Args:
            config: Model configuration (same as other models)
            num_decoder_stages: Number of decoder refinement stages (default: 3)
            num_encoder_layers: Number of encoder layers (default: 6)
            num_decoder_layers: Number of decoder layers per stage (default: 3)
            num_f_maps: Number of feature maps (default: 64)
            num_heads: Number of attention heads (default: 4)
            attention_type: Type of attention ('normal', 'block', 'sliding')
            device: Device to run on
        """
        super().__init__()

        if not (config.use_proprio or config.use_dino):
            raise ValueError("At least one of use_proprio or use_dino must be True")

        self.vocab_size = config.vocab_size + DEFAULT_PAD_TOKEN_OFFSET
        self.prediction_window_size = config.prediction_window_size
        self.device = device
        self.use_proprio = config.use_proprio
        self.use_dino = config.use_dino
        self.num_decoder_stages = num_decoder_stages

        # Encoders (same as existing models)
        if config.use_proprio:
            self.proprio_encoder = TinyFASTEncoder(
                vocab_size=self.vocab_size,
                embed_dim=config.token_embed_dim,
                num_heads=DEFAULT_NUM_HEADS,
                transformer_dropout=config.transformer_dropout,
                embedding_dropout=config.embedding_dropout
            )

        if config.use_dino:
            # Extract per-frame DINO features
            num_dino_frames = config.dino_feature_dim // DINO_MODEL_FEATURE_DIM
            self.num_dino_frames = num_dino_frames

            # Process each DINO frame separately
            self.dino_frame_encoder = nn.Sequential(
                nn.Linear(DINO_MODEL_FEATURE_DIM, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(config.dino_encoder_dropout)
            )

        # Calculate feature dimension for temporal processing
        temporal_dim = 0
        if config.use_proprio:
            temporal_dim += self.proprio_encoder.output_dim
        if config.use_dino:
            temporal_dim += 128  # Per-frame DINO encoding

        # Project to num_f_maps
        self.input_projection = nn.Conv1d(temporal_dim, num_f_maps, 1)

        # Single encoder stage
        self.encoder = Encoder(
            num_encoder_layers, num_f_maps, num_heads,
            dropout=config.mlp_dropout, attention_type=attention_type
        )

        # Multiple decoder stages
        self.decoders = nn.ModuleList([
            Decoder(
                num_decoder_layers, num_f_maps, num_heads,
                dropout=config.mlp_dropout, attention_type=attention_type
            )
            for _ in range(num_decoder_stages)
        ])

        # Output prediction head for each stage
        self.output_layers = nn.ModuleList([
            nn.Conv1d(num_f_maps, self.prediction_window_size, 1)
            for _ in range(num_decoder_stages)
        ])

        # Temporal pooling: use max pooling which is more stable than attention
        # Max pooling captures the strongest boundary signal across temporal positions

    def forward(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass (same signature as BoundaryPredictionMLP).

        Args:
            tokenized_actions: Token IDs (batch, seq_len) or None
            proprio_stats: Unused, for API compatibility
            dino_features: DINO features (batch, num_dino_frames * 1024) or None

        Returns:
            Logits (batch, prediction_window_size)
        """
        if self.use_proprio and tokenized_actions is None:
            raise ValueError("tokenized_actions required when use_proprio=True")
        if self.use_dino and dino_features is None:
            raise ValueError("dino_features required when use_dino=True")

        batch_size = tokenized_actions.shape[0] if tokenized_actions is not None else dino_features.shape[0]

        # Build temporal sequence from features
        temporal_features = []

        # Encode proprioception (single feature vector)
        if self.use_proprio:
            proprio_encoded = self.proprio_encoder(tokenized_actions)  # (batch, dim)
            # Expand to match DINO temporal dimension
            if self.use_dino:
                proprio_expanded = proprio_encoded.unsqueeze(2).expand(-1, -1, self.num_dino_frames)
                temporal_features.append(proprio_expanded)
            else:
                # If only proprio, create a pseudo-temporal dimension
                proprio_expanded = proprio_encoded.unsqueeze(2).expand(-1, -1, self.prediction_window_size)
                temporal_features.append(proprio_expanded)

        # Encode DINO features per frame
        if self.use_dino:
            # Reshape DINO features: (batch, num_frames * 1024) -> (batch, num_frames, 1024)
            dino_reshaped = dino_features.view(batch_size, self.num_dino_frames, DINO_MODEL_FEATURE_DIM)

            # Encode each frame
            dino_encoded_list = []
            for i in range(self.num_dino_frames):
                frame_features = self.dino_frame_encoder(dino_reshaped[:, i, :])  # (batch, 128)
                dino_encoded_list.append(frame_features)

            # Stack to (batch, 128, num_frames)
            dino_temporal = torch.stack(dino_encoded_list, dim=2)
            temporal_features.append(dino_temporal)

        # Concatenate features (batch, feature_dim, seq_len)
        combined = torch.cat(temporal_features, dim=1)

        # Project to num_f_maps
        x = self.input_projection(combined)

        # Encoder
        encoder_output = self.encoder(x)

        # Multi-stage decoder with progressive refinement
        outputs = []
        decoder_input = encoder_output

        for i, (decoder, output_layer) in enumerate(zip(self.decoders, self.output_layers)):
            # Decode with cross-attention to encoder
            decoder_output = decoder(decoder_input, encoder_output)

            # Predict
            stage_logits = output_layer(decoder_output)  # (batch, prediction_window_size, seq_len)
            outputs.append(stage_logits)

            # Prepare input for next stage: keep decoder refinement + encoder context
            if i < len(self.decoders) - 1:
                # Keep the decoder output (refinement) and add encoder (context)
                # This preserves information from previous refinement stages
                decoder_input = decoder_output

        # Use final stage output
        final_logits = outputs[-1]  # (batch, prediction_window_size, seq_len)

        # Use mean pooling across temporal dimension (like MS-TCN)
        # Mean pooling better preserves information and generalizes better
        pooled = final_logits.mean(dim=2)  # (batch, prediction_window_size)

        return pooled

    def predict_probabilities(
        self,
        tokenized_actions: Optional[torch.Tensor],
        proprio_stats: Optional[torch.Tensor],
        dino_features: Optional[torch.Tensor],
        use_softmax: bool = False
    ) -> torch.Tensor:
        """
        Predict probabilities (same signature as BoundaryPredictionMLP).

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Unused
            dino_features: DINO features
            use_softmax: Whether to apply softmax

        Returns:
            Probabilities (batch, prediction_window_size)
        """
        logits = self.forward(tokenized_actions, proprio_stats, dino_features)

        if use_softmax:
            return F.softmax(logits, dim=-1)
        else:
            return torch.sigmoid(logits)


def build_boundary_asformer(
    config: ModelConfig,
    num_decoder_stages: int = 3,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 3,
    num_f_maps: int = 64,
    num_heads: int = 4,
    attention_type: str = 'normal',
    device: str = 'cuda'
) -> BoundaryASFormer:
    """
    Factory function to build ASFormer boundary prediction model.

    Args:
        config: Model configuration (same as other models)
        num_decoder_stages: Number of decoder refinement stages
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers per stage
        num_f_maps: Number of feature maps
        num_heads: Number of attention heads
        attention_type: Type of attention ('normal', 'block', 'sliding')
        device: Device to run on

    Returns:
        Configured ASFormer model

    Example:
        >>> from models.boundary_prediction import ModelConfig
        >>> config = ModelConfig(vocab_size=2048, prediction_window_size=10)
        >>> model = build_boundary_asformer(config, num_decoder_stages=3, num_encoder_layers=6)
    """
    return BoundaryASFormer(
        config=config,
        num_decoder_stages=num_decoder_stages,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_f_maps=num_f_maps,
        num_heads=num_heads,
        attention_type=attention_type,
        device=device
    )


if __name__ == "__main__":
    # Test ASFormer with same interface as existing models
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.encoders import DINO_MODEL_FEATURE_DIM
    from models.boundary_prediction import ModelConfig

    print("Testing BoundaryASFormer with existing interface...")

    config = ModelConfig(
        vocab_size=2048,
        token_embed_dim=128,
        dino_feature_dim=3 * DINO_MODEL_FEATURE_DIM,  # 3 DINO frames
        prediction_window_size=10,
        use_proprio=True,
        use_dino=True
    )

    model = build_boundary_asformer(config, num_decoder_stages=3, num_encoder_layers=6)

    # Test forward pass
    batch_size = 16
    seq_len = 100
    tokenized_actions = torch.randint(0, 2048, (batch_size, seq_len))
    dino_features = torch.randn(batch_size, 3 * DINO_MODEL_FEATURE_DIM)

    logits = model(tokenized_actions, None, dino_features)

    print(f"  Input shapes:")
    print(f"    tokenized_actions: {tokenized_actions.shape}")
    print(f"    dino_features: {dino_features.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: (batch={batch_size}, prediction_window_size={config.prediction_window_size})")

    assert logits.shape == (batch_size, config.prediction_window_size)
    print("\n✓ BoundaryASFormer test passed!")
    print("\nUsage:")
    print("  - Drop-in replacement for BoundaryPredictionMLP")
    print("  - Use with existing dataloader and training scripts")
    print("  - Same inputs/outputs as other models")
