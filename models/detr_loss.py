"""Loss functions for DETR-style boundary detection.

This module implements the set-based loss with Hungarian matching for training
the boundary DETR model. The key components are:

1. HungarianMatcher: Finds optimal one-to-one matching between predictions and GT
2. SetCriterion: Computes losses for matched and unmatched predictions

The loss consists of:
- Position loss (L1): For matched predictions only
- Confidence loss (BCE): For all predictions (matched should be confident)

Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Performs Hungarian matching between predictions and ground truth boundaries.

    The Hungarian algorithm (Kuhn-Munkres algorithm) finds the optimal one-to-one
    assignment between predictions and ground truth that minimizes the total cost.

    Cost function:
        cost(pred_i, gt_j) = λ_pos * |pos_i - pos_j| + λ_conf * (1 - conf_i)

    Where:
        - Position cost encourages predictions close to ground truth
        - Confidence cost encourages high confidence for matched predictions
    """

    def __init__(self, cost_position: float = 5.0, cost_confidence: float = 2.0):
        """Initialize matcher.

        Args:
            cost_position: Weight for position cost in matching
            cost_confidence: Weight for confidence cost in matching
        """
        super().__init__()
        self.cost_position = cost_position
        self.cost_confidence = cost_confidence

    @torch.no_grad()
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching.

        Args:
            predictions: Dictionary with:
                - 'pred_positions': (batch, num_queries) normalized positions [0, 1]
                - 'pred_confidences': (batch, num_queries) confidence scores [0, 1]
            targets: List of dictionaries (length = batch_size), each with:
                - 'positions': (num_gt,) normalized GT positions [0, 1]
                - 'labels': (num_gt,) all ones (for compatibility)

        Returns:
            indices: List of (query_indices, target_indices) tuples, one per batch sample
                - query_indices: Indices of matched predictions
                - target_indices: Indices of matched ground truth
        """
        batch_size = predictions['pred_positions'].shape[0]
        num_queries = predictions['pred_positions'].shape[1]

        # Flatten batch for efficiency
        # We'll process each batch element separately but vectorize cost computation
        pred_positions = predictions['pred_positions']  # (batch, num_queries)
        pred_confidences = predictions['pred_confidences']  # (batch, num_queries)

        indices = []

        for b in range(batch_size):
            num_gt = len(targets[b]['positions'])

            if num_gt == 0:
                # No ground truth boundaries in this sample
                # No queries should be matched
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_positions.device),
                    torch.tensor([], dtype=torch.long, device=pred_positions.device)
                ))
                continue

            # Build cost matrix for this sample
            cost_matrix = self._build_cost_matrix(
                pred_positions[b],  # (num_queries,)
                pred_confidences[b],  # (num_queries,)
                targets[b]['positions']  # (num_gt,)
            )

            # Hungarian algorithm (runs on CPU)
            cost_matrix_np = cost_matrix.cpu().numpy()
            query_idx, target_idx = linear_sum_assignment(cost_matrix_np)

            # Convert to tensors
            query_idx = torch.tensor(query_idx, dtype=torch.long, device=pred_positions.device)
            target_idx = torch.tensor(target_idx, dtype=torch.long, device=pred_positions.device)

            indices.append((query_idx, target_idx))

        return indices

    def _build_cost_matrix(
        self,
        pred_pos: torch.Tensor,
        pred_conf: torch.Tensor,
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """Build cost matrix for a single sample.

        Args:
            pred_pos: (num_queries,) predicted positions
            pred_conf: (num_queries,) predicted confidences
            target_pos: (num_gt,) ground truth positions

        Returns:
            cost_matrix: (num_queries, num_gt)
        """
        num_queries = len(pred_pos)
        num_targets = len(target_pos)

        # Expand for broadcasting: (num_queries, 1) and (1, num_targets)
        pred_pos_expanded = pred_pos.unsqueeze(1).expand(num_queries, num_targets)
        target_pos_expanded = target_pos.unsqueeze(0).expand(num_queries, num_targets)

        # Position cost: L1 distance
        position_cost = torch.abs(pred_pos_expanded - target_pos_expanded)

        # Confidence cost: 1 - confidence
        # Intuition: High confidence predictions are "cheaper" to match
        confidence_cost = (1.0 - pred_conf).unsqueeze(1).expand(num_queries, num_targets)

        # Total cost
        cost_matrix = (
            self.cost_position * position_cost +
            self.cost_confidence * confidence_cost
        )

        return cost_matrix


class SetCriterion(nn.Module):
    """Computes the loss for DETR boundary detection.

    This loss consists of:
    1. Position loss (L1): For matched query-target pairs only
    2. Confidence loss (BCE): For all queries
       - Matched queries should have high confidence (target = 1)
       - Unmatched queries should have low confidence (target = 0)

    The loss uses Hungarian matching to find optimal assignment between
    predictions and ground truth before computing losses.
    """

    def __init__(
        self,
        num_queries: int,
        position_weight: float = 5.0,
        confidence_weight: float = 2.0,
        eos_coef: float = 0.1,
        matcher_cost_position: float = 5.0,
        matcher_cost_confidence: float = 2.0
    ):
        """Initialize loss criterion.

        Args:
            num_queries: Number of boundary queries in model
            position_weight: Weight for position loss in total loss
            confidence_weight: Weight for confidence loss in total loss
            eos_coef: Relative weight for "no boundary" class (lower = upweight positives)
            matcher_cost_position: Position cost weight for Hungarian matching
            matcher_cost_confidence: Confidence cost weight for Hungarian matching
        """
        super().__init__()

        self.num_queries = num_queries
        self.position_weight = position_weight
        self.confidence_weight = confidence_weight
        self.eos_coef = eos_coef

        # Create matcher
        self.matcher = HungarianMatcher(
            cost_position=matcher_cost_position,
            cost_confidence=matcher_cost_confidence
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with Hungarian matching.

        Args:
            predictions: Dictionary with:
                - 'pred_positions': (batch, num_queries) normalized positions [0, 1]
                - 'pred_confidences': (batch, num_queries) confidence scores [0, 1]
            targets: List of dictionaries (length = batch_size), each with:
                - 'positions': (num_gt,) normalized GT positions [0, 1]
                - 'labels': (num_gt,) all ones

        Returns:
            losses: Dictionary with:
                - 'loss_position': Position L1 loss
                - 'loss_confidence': Confidence BCE loss
                - 'loss_total': Weighted sum of losses
                - 'num_boundaries': Average number of boundaries per sample (for logging)
        """
        # Get matching indices using Hungarian algorithm
        indices = self.matcher(predictions, targets)

        # Compute individual losses
        loss_position = self._compute_position_loss(predictions, targets, indices)
        loss_confidence = self._compute_confidence_loss(predictions, targets, indices)

        # Weighted total loss
        loss_total = (
            self.position_weight * loss_position +
            self.confidence_weight * loss_confidence
        )

        # Additional statistics for logging
        num_boundaries = sum(len(t['positions']) for t in targets) / len(targets)

        return {
            'loss_position': loss_position,
            'loss_confidence': loss_confidence,
            'loss_total': loss_total,
            'num_boundaries': num_boundaries
        }

    def _compute_position_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute L1 loss for matched positions only.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            indices: Matched indices from Hungarian algorithm

        Returns:
            position_loss: Scalar loss value
        """
        pred_positions = predictions['pred_positions']
        batch_size = pred_positions.shape[0]

        # Collect all matched pairs
        matched_pred_positions = []
        matched_target_positions = []

        for b in range(batch_size):
            query_idx, target_idx = indices[b]

            if len(query_idx) == 0:
                # No matches in this sample
                continue

            # Get matched predictions and targets
            matched_pred = pred_positions[b, query_idx]
            matched_target = targets[b]['positions'][target_idx]

            matched_pred_positions.append(matched_pred)
            matched_target_positions.append(matched_target)

        if len(matched_pred_positions) == 0:
            # No boundaries in batch, return zero loss
            return torch.tensor(0.0, device=pred_positions.device, requires_grad=True)

        # Concatenate all matched pairs
        all_pred = torch.cat(matched_pred_positions)
        all_target = torch.cat(matched_target_positions)

        # L1 loss
        position_loss = F.l1_loss(all_pred, all_target)

        return position_loss

    def _compute_confidence_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss for confidence scores.

        Matched queries should predict high confidence (target = 1)
        Unmatched queries should predict low confidence (target = 0)

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            indices: Matched indices from Hungarian algorithm

        Returns:
            confidence_loss: Scalar loss value
        """
        pred_confidences = predictions['pred_confidences']
        batch_size, num_queries = pred_confidences.shape

        # Create target confidence labels
        # Initially all zeros (unmatched = no boundary)
        target_confidences = torch.zeros_like(pred_confidences)

        # Set matched queries to 1
        for b in range(batch_size):
            query_idx, target_idx = indices[b]
            if len(query_idx) > 0:
                target_confidences[b, query_idx] = 1.0

        # Compute BCE loss with class weighting
        # Since most queries are unmatched (negative), upweight the positive class
        # pos_weight = 1 / eos_coef means positive class is weighted more
        pos_weight = torch.tensor([1.0 / self.eos_coef], device=pred_confidences.device)

        # Compute BCE loss for each sample
        loss = F.binary_cross_entropy(
            pred_confidences,
            target_confidences,
            reduction='none'
        )

        # Apply position-dependent weighting
        # Matched queries (target=1) get higher weight
        weights = torch.where(
            target_confidences == 1.0,
            pos_weight.expand_as(loss),
            torch.ones_like(loss)
        )

        weighted_loss = (loss * weights).mean()

        return weighted_loss


class FocalSetCriterion(SetCriterion):
    """Set criterion with focal loss for confidence.

    Extends SetCriterion by replacing BCE with Focal Loss for confidence prediction.
    Focal loss helps handle extreme class imbalance by focusing on hard examples.

    Focal Loss: FL(pt) = -α(1-pt)^γ log(pt)
    where:
        - pt is the probability of the true class
        - α balances positive/negative examples
        - γ focuses on hard examples (typical: 2.0)
    """

    def __init__(
        self,
        num_queries: int,
        position_weight: float = 5.0,
        confidence_weight: float = 2.0,
        eos_coef: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        matcher_cost_position: float = 5.0,
        matcher_cost_confidence: float = 2.0
    ):
        """Initialize focal loss criterion.

        Args:
            num_queries: Number of boundary queries
            position_weight: Weight for position loss
            confidence_weight: Weight for confidence loss
            eos_coef: Relative weight for negative class
            focal_alpha: Focal loss alpha (class balance)
            focal_gamma: Focal loss gamma (focus on hard examples)
            matcher_cost_position: Position cost for matching
            matcher_cost_confidence: Confidence cost for matching
        """
        super().__init__(
            num_queries=num_queries,
            position_weight=position_weight,
            confidence_weight=confidence_weight,
            eos_coef=eos_coef,
            matcher_cost_position=matcher_cost_position,
            matcher_cost_confidence=matcher_cost_confidence
        )

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _compute_confidence_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute focal loss for confidence scores.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            indices: Matched indices

        Returns:
            confidence_loss: Scalar focal loss value
        """
        pred_confidences = predictions['pred_confidences']
        batch_size, num_queries = pred_confidences.shape

        # Create target confidence labels
        target_confidences = torch.zeros_like(pred_confidences)

        for b in range(batch_size):
            query_idx, target_idx = indices[b]
            if len(query_idx) > 0:
                target_confidences[b, query_idx] = 1.0

        # Compute focal loss
        # BCE component
        bce_loss = F.binary_cross_entropy(
            pred_confidences,
            target_confidences,
            reduction='none'
        )

        # Focal weight: (1 - pt)^gamma
        # pt is the probability of the true class
        pt = torch.where(
            target_confidences == 1.0,
            pred_confidences,
            1.0 - pred_confidences
        )
        focal_weight = (1.0 - pt) ** self.focal_gamma

        # Alpha weight: balance positive/negative
        alpha_weight = torch.where(
            target_confidences == 1.0,
            torch.tensor(self.focal_alpha, device=pred_confidences.device),
            torch.tensor(1.0 - self.focal_alpha, device=pred_confidences.device)
        )

        # Combine: Focal Loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Apply eos_coef weighting
        # Upweight positive examples (rare boundaries)
        final_weight = torch.where(
            target_confidences == 1.0,
            torch.tensor(1.0 / self.eos_coef, device=pred_confidences.device),
            torch.ones_like(target_confidences)
        )

        weighted_focal_loss = (focal_loss * final_weight).mean()

        return weighted_focal_loss


def create_detr_criterion(
    config,
    use_focal_loss: bool = False
) -> nn.Module:
    """Factory function to create DETR loss criterion.

    Args:
        config: Model/training configuration with loss parameters
        use_focal_loss: If True, use focal loss for confidence

    Returns:
        Initialized loss criterion
    """
    if use_focal_loss:
        return FocalSetCriterion(
            num_queries=config.num_boundary_queries,
            position_weight=config.loss_position_weight,
            confidence_weight=config.loss_confidence_weight,
            eos_coef=config.loss_eos_coef,
            focal_alpha=config.get('loss_focal_alpha', 0.25),
            focal_gamma=config.get('loss_focal_gamma', 2.0),
            matcher_cost_position=config.get('matcher_cost_position', 5.0),
            matcher_cost_confidence=config.get('matcher_cost_confidence', 2.0)
        )
    else:
        return SetCriterion(
            num_queries=config.num_boundary_queries,
            position_weight=config.loss_position_weight,
            confidence_weight=config.loss_confidence_weight,
            eos_coef=config.loss_eos_coef,
            matcher_cost_position=config.get('matcher_cost_position', 5.0),
            matcher_cost_confidence=config.get('matcher_cost_confidence', 2.0)
        )
