"""Loss functions for boundary prediction.

This module provides various loss functions suitable for segment boundary
prediction, including support for both hard and soft (probability) labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)

    The focal loss focuses training on hard examples by down-weighting
    easy examples. The gamma parameter controls how much to down-weight.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (>=0). Higher values focus more on hard examples.
            reduction: Reduction method: 'mean', 'sum', or 'none'

        Example:
            >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            >>> logits = model(inputs)
            >>> loss = loss_fn(logits, targets)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model logits of shape (batch_size, 2)
            targets: Binary targets of shape (batch_size,) with values in {0, 1}

        Returns:
            Scalar loss value (or tensor if reduction='none')
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


class SoftFocalLoss(nn.Module):
    """
    Focal Loss adapted for soft (continuous) probability targets.

    This extends FocalLoss to work with label smoothing and soft targets
    in the range [0, 1] rather than hard {0, 1} labels.

    Particularly useful when using Gaussian or other smoothing strategies
    for boundary labels.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Soft Focal Loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: Reduction method: 'mean', 'sum', or 'none'

        Example:
            >>> loss_fn = SoftFocalLoss(alpha=0.25, gamma=2.0)
            >>> logits = model(inputs)
            >>> smooth_labels = gaussian_smooth(hard_labels)
            >>> loss = loss_fn(logits, smooth_labels)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute soft focal loss.

        Args:
            logits: Model logits of shape (batch_size, ...) for boundary/non-boundary.
                   Can be shape (batch, 2) for 2-class or (batch, prediction_window) for direct probability.
            targets: Soft probability targets in [0, 1] of shape matching logits' last dimension

        Returns:
            Scalar loss value (or tensor if reduction='none')
        """
        # Handle different input shapes
        if logits.ndim == 2 and logits.size(1) == 2:
            # 2-class logits (batch, 2)
            probs = F.softmax(logits, dim=1)
            p_pos = probs[:, 1]
            p_neg = probs[:, 0]
        else:
            # Direct boundary logits (batch, prediction_window)
            p_pos = torch.sigmoid(logits)
            p_neg = 1 - p_pos

        # Compute weighted cross-entropy with soft targets
        # CE = -[y*log(p_pos) + (1-y)*log(p_neg)]
        ce_pos = -torch.log(p_pos + 1e-8) * targets
        ce_neg = -torch.log(p_neg + 1e-8) * (1 - targets)
        ce_loss = ce_pos + ce_neg

        # Compute focal weight
        # pt represents how much the prediction agrees with the target
        pt = targets * p_pos + (1 - targets) * p_neg
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting (more weight to positive class)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossBCE(nn.Module):
    """
    Focal Loss using BCE with logits for numerical stability.

    This variant directly works with binary cross-entropy and is suitable
    for single-output boundary prediction.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        """
        Initialize Focal Loss BCE.

        Args:
            gamma: Focusing parameter
            alpha: Optional class-balancing weight
            reduction: Reduction method: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss using BCE.

        Args:
            logits: Model logits (before sigmoid)
            targets: Binary or soft targets in [0, 1]

        Returns:
            Scalar loss value
        """
        # BCE with logits is more numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Apply focal weighting
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        # Optional class-balancing alpha
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftBCELoss(nn.Module):
    """
    Soft Binary Cross-Entropy Loss for smooth probability targets.

    Standard BCE that supports both hard labels (0 or 1) and soft labels
    (continuous values in [0, 1]).
    """

    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Soft BCE Loss.

        Args:
            pos_weight: Weight for positive class to handle imbalance
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute soft BCE loss.

        Args:
            logits: Model logits. Can be (batch, 2) or (batch, prediction_window)
            targets: Probability targets in [0, 1]

        Returns:
            Scalar loss value
        """
        # Handle 2-class logits
        if logits.ndim == 2 and logits.size(1) == 2:
            # Convert 2-class logits to binary logits (difference)
            binary_logits = logits[:, 1] - logits[:, 0]
        else:
            binary_logits = logits

        # Use BCEWithLogitsLoss for numerical stability
        if self.pos_weight != 1.0:
            loss = F.binary_cross_entropy_with_logits(
                binary_logits,
                targets,
                pos_weight=torch.tensor([self.pos_weight], device=logits.device),
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                binary_logits,
                targets,
                reduction=self.reduction
            )

        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance.

    Simple BCE with configurable positive class weight.
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight: Weight for positive class

        Example:
            >>> # Give 10x weight to boundary frames
            >>> loss_fn = WeightedBCELoss(pos_weight=10.0)
        """
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Model logits. Can be (batch, 2) or (batch, prediction_window)
            targets: Binary or soft targets

        Returns:
            Scalar loss value
        """
        # Handle 2-class logits
        if logits.ndim == 2 and logits.size(1) == 2:
            binary_logits = logits[:, 1] - logits[:, 0]
        else:
            binary_logits = logits

        loss = F.binary_cross_entropy_with_logits(
            binary_logits,
            targets.float(),
            pos_weight=self.pos_weight.to(logits.device)
        )

        return loss


class MSEProbabilityLoss(nn.Module):
    """
    Mean Squared Error loss for probability prediction.

    Treats boundary prediction as regression problem. Simple but can
    work well with smooth labels.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSE Probability Loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on predicted probabilities.

        Args:
            logits: Model logits
            targets: Probability targets in [0, 1]

        Returns:
            Scalar loss value
        """
        # Get predicted probability
        if logits.ndim == 2 and logits.size(1) == 2:
            probs = F.softmax(logits, dim=1)
            pred_prob = probs[:, 1]
        else:
            pred_prob = torch.sigmoid(logits)

        # MSE between predicted and target probability
        loss = (pred_prob - targets) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions by name.

    Args:
        loss_name: Name of loss function
        **kwargs: Arguments to pass to loss constructor

    Returns:
        Loss function module

    Supported losses:
        - 'focal': FocalLoss
        - 'soft_focal': SoftFocalLoss
        - 'focal_bce': FocalLossBCE
        - 'bce': SoftBCELoss
        - 'weighted_bce': WeightedBCELoss
        - 'mse': MSEProbabilityLoss

    Example:
        >>> loss_fn = get_loss_function('soft_focal', alpha=0.25, gamma=2.0)
    """
    loss_map = {
        'focal': FocalLoss,
        'soft_focal': SoftFocalLoss,
        'focal_bce': FocalLossBCE,
        'bce': SoftBCELoss,
        'weighted_bce': WeightedBCELoss,
        'mse': MSEProbabilityLoss,
    }

    if loss_name not in loss_map:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: {list(loss_map.keys())}"
        )

    return loss_map[loss_name](**kwargs)
