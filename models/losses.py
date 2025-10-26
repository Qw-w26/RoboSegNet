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


# ============================================================================
# Distribution-Based Losses (for window-based boundary prediction)
# ============================================================================


class DistributionCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy loss treating the prediction window as a probability distribution.

    RECOMMENDED for boundary prediction with Gaussian-smoothed labels.

    This loss treats the model outputs as a distribution over frame positions
    in the prediction window, rather than independent binary classifications.
    This is fundamentally more appropriate when:
    - Labels are continuous probability distributions (Gaussian smoothed)
    - The task is to predict WHERE a boundary occurs (localization)
    - Multiple boundaries may exist in a window (multi-modal distribution)

    Key advantages over BCE:
    - Enforces probability mass conservation across the window
    - Handles multiple boundaries naturally (multi-modal distributions)
    - Invariant to the exact shape of label smoothing kernel
    - Theoretically sound for localization tasks

    Mathematical formulation:
        1. Normalize labels to sum to 1: Q[i] = L[i] / Σⱼ L[j]
        2. Apply softmax to logits: P[i] = exp(z[i]) / Σⱼ exp(z[j])
        3. Cross-entropy: ℒ = -Σᵢ Q[i] * log(P[i])

    Reference:
        Standard cross-entropy loss for probability distributions
    """

    def __init__(self, epsilon: float = 1e-8, uniform_on_zero: bool = True, reduction: str = 'mean'):
        """
        Initialize Distribution Cross-Entropy Loss.

        Args:
            epsilon: Small constant for numerical stability
            uniform_on_zero: If True, use uniform distribution when labels sum to ~0
                           (windows with no boundaries). If False, skip those samples.
            reduction: 'mean', 'sum', or 'none'

        Example:
            >>> loss_fn = DistributionCrossEntropyLoss()
            >>> logits = model(inputs)  # (batch, 120)
            >>> labels = gaussian_smoothed_labels  # (batch, 120)
            >>> loss = loss_fn(logits, labels)
        """
        super().__init__()
        self.epsilon = epsilon
        self.uniform_on_zero = uniform_on_zero
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution cross-entropy loss.

        Args:
            logits: Model logits of shape (batch, window_size)
            targets: Smoothed probability targets of shape (batch, window_size)

        Returns:
            Scalar loss value (or tensor if reduction='none')
        """
        # Normalize labels to sum to 1 (convert to probability distribution)
        label_sum = targets.sum(dim=1, keepdim=True)

        # Handle windows with no boundaries
        if self.uniform_on_zero:
            # Use uniform distribution when sum is near zero
            mask_zero = (label_sum < self.epsilon).squeeze(1)
            label_sum = torch.clamp(label_sum, min=self.epsilon)
            target_dist = targets / label_sum

            # For zero-boundary windows, use uniform distribution
            if mask_zero.any():
                uniform = torch.ones_like(targets) / targets.size(1)
                target_dist = torch.where(
                    mask_zero.unsqueeze(1),
                    uniform,
                    target_dist
                )
        else:
            # Skip samples with no boundaries
            mask_nonzero = (label_sum >= self.epsilon).squeeze(1)
            if not mask_nonzero.any():
                # All samples have zero boundaries, return zero loss
                return torch.tensor(0.0, device=logits.device)

            label_sum = label_sum[mask_nonzero]
            target_dist = targets[mask_nonzero] / label_sum
            logits = logits[mask_nonzero]

        # Compute log softmax (numerically stable)
        log_probs = F.log_softmax(logits, dim=1)

        # Cross-entropy: -Σ Q[i] * log(P[i])
        loss = -(target_dist * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for distribution matching.

    Computes KL(target || prediction) where both are treated as probability
    distributions over the prediction window.

    Note: For optimization purposes, KL divergence and cross-entropy are
    equivalent (differ only by a constant H(target)). This is provided
    as a separate option for clarity and compatibility.

    Mathematical formulation:
        KL(Q || P) = Σᵢ Q[i] * log(Q[i] / P[i])
                   = Σᵢ Q[i] * log(Q[i]) - Σᵢ Q[i] * log(P[i])
                   = H(Q) + CE(Q, P)
    """

    def __init__(self, epsilon: float = 1e-8, uniform_on_zero: bool = True, reduction: str = 'batchmean'):
        """
        Initialize KL Divergence Loss.

        Args:
            epsilon: Small constant for numerical stability
            uniform_on_zero: Use uniform distribution for zero-boundary windows
            reduction: 'batchmean' (default), 'mean', 'sum', or 'none'
                     'batchmean' divides by batch size (PyTorch KL convention)

        Example:
            >>> loss_fn = KLDivergenceLoss()
            >>> logits = model(inputs)
            >>> loss = loss_fn(logits, labels)
        """
        super().__init__()
        self.epsilon = epsilon
        self.uniform_on_zero = uniform_on_zero
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            logits: Model logits of shape (batch, window_size)
            targets: Smoothed probability targets of shape (batch, window_size)

        Returns:
            Scalar loss value
        """
        # Normalize labels to probability distribution
        label_sum = targets.sum(dim=1, keepdim=True)

        if self.uniform_on_zero:
            mask_zero = (label_sum < self.epsilon).squeeze(1)
            label_sum = torch.clamp(label_sum, min=self.epsilon)
            target_dist = targets / label_sum

            if mask_zero.any():
                uniform = torch.ones_like(targets) / targets.size(1)
                target_dist = torch.where(
                    mask_zero.unsqueeze(1),
                    uniform,
                    target_dist
                )
        else:
            mask_nonzero = (label_sum >= self.epsilon).squeeze(1)
            if not mask_nonzero.any():
                return torch.tensor(0.0, device=logits.device)

            label_sum = label_sum[mask_nonzero]
            target_dist = targets[mask_nonzero] / label_sum
            logits = logits[mask_nonzero]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # KL divergence using PyTorch's built-in
        # F.kl_div expects (log_predictions, targets)
        loss = F.kl_div(log_probs, target_dist, reduction=self.reduction)

        return loss


class JensenShannonLoss(nn.Module):
    """
    Jensen-Shannon Divergence loss for symmetric distribution matching.

    JS divergence is a symmetric and bounded variant of KL divergence:
        JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Properties:
    - Symmetric: JS(P || Q) = JS(Q || P)
    - Bounded: JS ∈ [0, 1] (when using log base 2)
    - Smooth: No issues when P or Q have zeros

    Useful when you want:
    - Symmetric penalty (prediction vs target treated equally)
    - Robust gradients (better behaved than KL at distribution boundaries)
    """

    def __init__(self, epsilon: float = 1e-8, uniform_on_zero: bool = True, reduction: str = 'mean'):
        """
        Initialize Jensen-Shannon Divergence Loss.

        Args:
            epsilon: Small constant for numerical stability
            uniform_on_zero: Use uniform distribution for zero-boundary windows
            reduction: 'mean', 'sum', or 'none'

        Example:
            >>> loss_fn = JensenShannonLoss()
            >>> logits = model(inputs)
            >>> loss = loss_fn(logits, labels)
        """
        super().__init__()
        self.epsilon = epsilon
        self.uniform_on_zero = uniform_on_zero
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence loss.

        Args:
            logits: Model logits of shape (batch, window_size)
            targets: Smoothed probability targets of shape (batch, window_size)

        Returns:
            Scalar loss value
        """
        # Get prediction probabilities
        pred_probs = F.softmax(logits, dim=1)

        # Normalize target labels
        label_sum = targets.sum(dim=1, keepdim=True)

        if self.uniform_on_zero:
            mask_zero = (label_sum < self.epsilon).squeeze(1)
            label_sum = torch.clamp(label_sum, min=self.epsilon)
            target_dist = targets / label_sum

            if mask_zero.any():
                uniform = torch.ones_like(targets) / targets.size(1)
                target_dist = torch.where(
                    mask_zero.unsqueeze(1),
                    uniform,
                    target_dist
                )
        else:
            mask_nonzero = (label_sum >= self.epsilon).squeeze(1)
            if not mask_nonzero.any():
                return torch.tensor(0.0, device=logits.device)

            label_sum = label_sum[mask_nonzero]
            target_dist = targets[mask_nonzero] / label_sum
            pred_probs = pred_probs[mask_nonzero]

        # Compute mixture distribution
        mixture = 0.5 * (pred_probs + target_dist)
        mixture_log = torch.log(mixture + self.epsilon)

        # KL(P || M)
        kl_pred_mixture = (pred_probs * (torch.log(pred_probs + self.epsilon) - mixture_log)).sum(dim=1)

        # KL(Q || M)
        kl_target_mixture = (target_dist * (torch.log(target_dist + self.epsilon) - mixture_log)).sum(dim=1)

        # JS divergence
        js_div = 0.5 * (kl_pred_mixture + kl_target_mixture)

        if self.reduction == 'mean':
            return js_div.mean()
        elif self.reduction == 'sum':
            return js_div.sum()
        else:
            return js_div


class FocalDistributionCELoss(nn.Module):
    """
    Focal variant of Distribution Cross-Entropy Loss.

    Combines distribution-based cross-entropy with focal loss weighting
    to focus on hard-to-predict boundary positions.

    Useful when:
    - Some boundary positions are consistently harder to predict
    - You want to down-weight easy examples
    - Training needs to focus on ambiguous temporal regions
    """

    def __init__(
        self,
        gamma: float = 2.0,
        epsilon: float = 1e-8,
        uniform_on_zero: bool = True,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Distribution CE Loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            epsilon: Numerical stability constant
            uniform_on_zero: Use uniform distribution for zero-boundary windows
            reduction: 'mean', 'sum', or 'none'

        Example:
            >>> loss_fn = FocalDistributionCELoss(gamma=2.0)
            >>> loss = loss_fn(logits, labels)
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.uniform_on_zero = uniform_on_zero
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal distribution cross-entropy loss.

        Args:
            logits: Model logits of shape (batch, window_size)
            targets: Smoothed probability targets of shape (batch, window_size)

        Returns:
            Scalar loss value
        """
        # Normalize targets
        label_sum = targets.sum(dim=1, keepdim=True)

        if self.uniform_on_zero:
            mask_zero = (label_sum < self.epsilon).squeeze(1)
            label_sum = torch.clamp(label_sum, min=self.epsilon)
            target_dist = targets / label_sum

            if mask_zero.any():
                uniform = torch.ones_like(targets) / targets.size(1)
                target_dist = torch.where(
                    mask_zero.unsqueeze(1),
                    uniform,
                    target_dist
                )
        else:
            mask_nonzero = (label_sum >= self.epsilon).squeeze(1)
            if not mask_nonzero.any():
                return torch.tensor(0.0, device=logits.device)

            label_sum = label_sum[mask_nonzero]
            target_dist = targets[mask_nonzero] / label_sum
            logits = logits[mask_nonzero]

        # Get probabilities
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Compute agreement between prediction and target
        # pt represents how well prediction matches target distribution
        pt = (target_dist * probs).sum(dim=1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy with focal weighting
        ce_loss = -(target_dist * log_probs).sum(dim=1)
        loss = focal_weight * ce_loss

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

        Distribution-based (RECOMMENDED for window-based boundary prediction):
        - 'dist_ce': DistributionCrossEntropyLoss (RECOMMENDED)
        - 'kl_div': KLDivergenceLoss
        - 'js_div': JensenShannonLoss
        - 'focal_dist_ce': FocalDistributionCELoss

        Per-frame binary classification (legacy):
        - 'focal': FocalLoss
        - 'soft_focal': SoftFocalLoss
        - 'focal_bce': FocalLossBCE
        - 'bce': SoftBCELoss
        - 'weighted_bce': WeightedBCELoss
        - 'mse': MSEProbabilityLoss

    Example:
        >>> # Recommended for boundary prediction
        >>> loss_fn = get_loss_function('dist_ce')
        >>>
        >>> # Legacy BCE-based
        >>> loss_fn = get_loss_function('soft_focal', alpha=0.25, gamma=2.0)
    """
    loss_map = {
        # Distribution-based losses (recommended)
        'dist_ce': DistributionCrossEntropyLoss,
        'kl_div': KLDivergenceLoss,
        'js_div': JensenShannonLoss,
        'focal_dist_ce': FocalDistributionCELoss,

        # Per-frame losses (legacy)
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
