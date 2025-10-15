"""
Models for segment boundary prediction.

This module provides neural network architectures for predicting segment boundaries
from multi-modal features (DINO visual + Fast-DROID tokenized proprio).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BoundaryPredictionMLP(nn.Module):
    """
    MLP for segment boundary prediction.

    Takes as input:
    - Frozen DINO features from begin, middle, end frames (3 * 1024 = 3072 dims)
    - Fast-DROID tokenized proprio (embedded via embedding layer)
    - Statistical proprio features (84 dims)

    Outputs:
    - Binary logits for boundary prediction
    """

    def __init__(
        self,
        vocab_size: int = 32000,  # Fast-DROID tokenizer vocab size
        token_embed_dim: int = 128,
        max_token_length: int = 600,
        proprio_stats_dim: int = 84,  # 6 stats * 14 dims
        dino_feature_dim: int = 3072,  # 3 frames * 1024 dims
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3
    ):
        """
        Initialize the boundary prediction MLP.

        Args:
            vocab_size: Size of tokenizer vocabulary
            token_embed_dim: Embedding dimension for tokens
            max_token_length: Maximum token sequence length
            proprio_stats_dim: Dimension of statistical proprio features
            dino_feature_dim: Dimension of DINO features (3 * 1024)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.max_token_length = max_token_length
        self.proprio_stats_dim = proprio_stats_dim
        self.dino_feature_dim = dino_feature_dim

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim, padding_idx=0)

        # Token sequence encoder (simple average pooling or 1D conv)
        self.token_encoder = nn.Sequential(
            nn.Conv1d(token_embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Compute total input dimension
        # Token features: 128 (from conv encoder)
        # Proprio stats: proprio_stats_dim
        # DINO features: dino_feature_dim (frozen)
        total_input_dim = 128 + proprio_stats_dim + dino_feature_dim

        # MLP layers
        layers = []
        prev_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        tokenized_actions: torch.Tensor,
        proprio_stats: torch.Tensor,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokenized_actions: (batch_size, max_token_length) - token IDs
            proprio_stats: (batch_size, proprio_stats_dim) - statistical features
            dino_features: (batch_size, dino_feature_dim) - frozen DINO features

        Returns:
            logits: (batch_size, 2) - binary classification logits
        """
        # Embed tokens
        token_embeds = self.token_embedding(tokenized_actions)  # (B, L, E)

        # Encode token sequence
        token_embeds = token_embeds.transpose(1, 2)  # (B, E, L) for conv1d
        token_features = self.token_encoder(token_embeds)  # (B, 128, 1)
        token_features = token_features.squeeze(-1)  # (B, 128)

        # Concatenate all features
        combined = torch.cat([
            token_features,
            proprio_stats,
            dino_features  # Already frozen
        ], dim=1)

        # MLP forward
        logits = self.mlp(combined)

        return logits

    def predict_proba(
        self,
        tokenized_actions: torch.Tensor,
        proprio_stats: torch.Tensor,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probabilities.

        Returns:
            probs: (batch_size, 2) - class probabilities
        """
        logits = self.forward(tokenized_actions, proprio_stats, dino_features)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        tokenized_actions: torch.Tensor,
        proprio_stats: torch.Tensor,
        dino_features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict binary labels.

        Args:
            tokenized_actions: Token IDs
            proprio_stats: Statistical features
            dino_features: DINO features
            threshold: Threshold for positive class

        Returns:
            predictions: (batch_size,) - binary predictions (0 or 1)
        """
        probs = self.predict_proba(tokenized_actions, proprio_stats, dino_features)
        return (probs[:, 1] > threshold).long()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    where pt is the probability of the correct class.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (batch_size, 2) - model logits
            targets: (batch_size,) - binary targets (0 or 1)

        Returns:
            loss: Scalar loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance.
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: (batch_size, 2) - model logits
            targets: (batch_size,) - binary targets (0 or 1)

        Returns:
            loss: Scalar loss value
        """
        # Convert logits to probabilities for positive class
        probs = F.softmax(logits, dim=1)[:, 1]

        # BCE with logits (more numerically stable)
        # We need to convert to binary logits
        binary_logits = logits[:, 1] - logits[:, 0]

        loss = F.binary_cross_entropy_with_logits(
            binary_logits,
            targets.float(),
            pos_weight=self.pos_weight.to(logits.device)
        )

        return loss


def create_model(
    vocab_size: int = 32000,
    max_token_length: int = 600,
    hidden_dims: list = [512, 256, 128],
    dropout: float = 0.3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BoundaryPredictionMLP:
    """
    Create and initialize a boundary prediction model.

    Args:
        vocab_size: Size of tokenizer vocabulary
        max_token_length: Maximum token sequence length
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        device: Device to place model on

    Returns:
        Initialized model
    """
    model = BoundaryPredictionMLP(
        vocab_size=vocab_size,
        max_token_length=max_token_length,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    model = model.to(device)

    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model(device=device)
    print(f"Model created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy batch
    batch_size = 16
    max_token_length = 600

    tokenized_actions = torch.randint(0, 32000, (batch_size, max_token_length)).to(device)
    proprio_stats = torch.randn(batch_size, 84).to(device)
    dino_features = torch.randn(batch_size, 3072).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)

    # Forward pass
    logits = model(tokenized_actions, proprio_stats, dino_features)
    print(f"\nLogits shape: {logits.shape}")

    # Test loss
    criterion = FocalLoss()
    loss = criterion(logits, labels)
    print(f"Focal loss: {loss.item():.4f}")

    # Test predictions
    preds = model.predict(tokenized_actions, proprio_stats, dino_features)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions: {preds}")
