"""
MS-TCN (Multi-Stage Temporal Convolutional Network) for boundary prediction.

Adapted to work with existing dataloader and evaluation pipeline.
Uses DINO features, compatible with sliding window approach.

Drop-in replacement for BoundaryPredictionMLP - same inputs, same outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .encoders import TinyFASTEncoder, DINO_MODEL_FEATURE_DIM
from .boundary_prediction import ModelConfig, DEFAULT_NUM_HEADS, DEFAULT_PAD_TOKEN_OFFSET


class DilatedResidualLayer(nn.Module):
    """Dilated residual convolutional layer for temporal modeling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.5):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class TemporalStage(nn.Module):
    """Single stage of temporal convolution network."""

    def __init__(self, num_layers: int, num_f_maps: int, in_dim: int,
                 out_classes: int, dropout: float = 0.5):
        super().__init__()

        self.conv_in = nn.Conv1d(in_dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                num_f_maps, num_f_maps,
                kernel_size=3,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.conv_out = nn.Conv1d(num_f_maps, out_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class BoundaryMSTCN(nn.Module):
    """
    MS-TCN adapted for boundary prediction with existing dataloader.

    Compatible with existing training pipeline. Uses the same inputs and outputs
    as BoundaryPredictionMLP but with multi-stage temporal convolutions.

    Architecture:
        Tokenized Actions → TinyFASTEncoder →  ┐
                                               ├→ Expand to sequence → MS-TCN → Pool → Predictions
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
        num_stages: int = 2,
        num_layers: int = 5,
        num_f_maps: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize MS-TCN boundary prediction model.

        Args:
            config: Model configuration (same as other models)
            num_stages: Number of refinement stages (default: 2 for sliding windows)
            num_layers: Number of dilated layers per stage (default: 5)
            num_f_maps: Number of feature maps (default: 64)
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
        self.num_stages = num_stages

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
            # Extract per-frame DINO features instead of concatenating
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

        # Multi-stage temporal convolution
        self.stage1 = TemporalStage(
            num_layers, num_f_maps, temporal_dim,
            self.prediction_window_size, config.mlp_dropout
        )

        self.stages = nn.ModuleList([
            TemporalStage(
                num_layers, num_f_maps,
                temporal_dim + self.prediction_window_size,
                self.prediction_window_size, config.mlp_dropout
            )
            for _ in range(num_stages - 1)
        ])

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

        # Multi-stage temporal convolution
        out = self.stage1(combined)  # (batch, prediction_window_size, seq_len)

        for stage in self.stages:
            # Concatenate input features with previous stage predictions (with sigmoid)
            # Use sigmoid for binary classification, not softmax
            out_sigmoid = torch.sigmoid(out)
            stage_input = torch.cat([combined, out_sigmoid], dim=1)
            out = stage(stage_input)

        # Extract predictions for the final temporal position
        # out shape: (batch, prediction_window_size, seq_len) where seq_len = num_dino_frames
        # We take the last frame (most recent observation) instead of averaging
        # This preserves temporal peaks for boundary detection
        out = out[:, :, -1]  # (batch, prediction_window_size)

        return out

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


def build_boundary_mstcn(
    config: ModelConfig,
    num_stages: int = 2,
    num_layers: int = 5,
    num_f_maps: int = 64,
    device: str = 'cuda'
) -> BoundaryMSTCN:
    """
    Factory function to build MS-TCN boundary prediction model.

    Args:
        config: Model configuration (same as other models)
        num_stages: Number of refinement stages
        num_layers: Number of dilated layers per stage
        num_f_maps: Number of feature maps
        device: Device to run on

    Returns:
        Configured MS-TCN model

    Example:
        >>> from models.boundary_prediction import ModelConfig
        >>> config = ModelConfig(vocab_size=2048, prediction_window_size=10)
        >>> model = build_boundary_mstcn(config, num_stages=2, num_layers=5)
    """
    return BoundaryMSTCN(
        config=config,
        num_stages=num_stages,
        num_layers=num_layers,
        num_f_maps=num_f_maps,
        device=device
    )


if __name__ == "__main__":
    # Test MS-TCN with same interface as existing models
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.encoders import DINO_MODEL_FEATURE_DIM
    from models.boundary_prediction import ModelConfig

    print("Testing BoundaryMSTCN with existing interface...")

    config = ModelConfig(
        vocab_size=2048,
        token_embed_dim=128,
        dino_feature_dim=3 * DINO_MODEL_FEATURE_DIM,  # 3 DINO frames
        prediction_window_size=10,
        use_proprio=True,
        use_dino=True
    )

    model = build_boundary_mstcn(config, num_stages=2, num_layers=5)

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
    print("\n✓ BoundaryMSTCN test passed!")
    print("\nUsage:")
    print("  - Drop-in replacement for BoundaryPredictionMLP")
    print("  - Use with existing dataloader and training scripts")
    print("  - Same inputs/outputs as other models")
