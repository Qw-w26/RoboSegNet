# DETR-Style Boundary Detection: Technical Design Document

**Author**: Claude Code
**Date**: 2025-11-05
**Version**: 1.0

---

## Executive Summary

This document outlines the design and implementation of a DETR-style (Detection Transformer) approach for robotic task boundary detection. Unlike the current dense prediction approach, this method directly predicts a **set of boundaries** using learned queries and Hungarian matching.

### Key Innovations
- **Set-based prediction**: Directly output N boundaries instead of per-frame probabilities
- **Learned queries**: Specialized embeddings that "ask" about boundaries
- **Hungarian matching**: Optimal assignment between predictions and ground truth
- **No post-processing**: Direct boundary extraction without NMS (though NMS still useful for window aggregation)

---

## 1. Problem Formulation

### Current Approach (Baseline)
```
Input: Window features (18 frames)
Output: Probability distribution over 360 frames (fixed vector)
Loss: MSE/BCE on dense probabilities
Post-processing: Peak detection + NMS
```

**Limitations:**
- Fixed output size (360) regardless of actual number of boundaries
- CLS token aggregation loses temporal information
- Requires careful post-processing tuning

### DETR Approach (Proposed)
```
Input: Window features (18 frames or variable length)
Output: Set of N boundary predictions {(position₁, confidence₁), ..., (positionₙ, confidenceₙ)}
Loss: Hungarian matching + position L1 + confidence BCE
Post-processing: Confidence thresholding (simple)
```

**Advantages:**
- Variable number of boundaries naturally handled
- Direct boundary positions (no peak detection needed)
- End-to-end differentiable
- Better for multiple boundaries per window

---

## 2. Architecture Components

### 2.1 Overview

```
┌─────────────────────────────────────────────┐
│  Feature Extraction (Reuse Existing)        │
│  - TinyFASTEncoder (proprioceptive)         │
│  - TemporalVisualEncoder (DINO)             │
│  - CrossModalAttention (fusion)             │
│  Output: (batch, seq_len, hidden_dim)       │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Trajectory Encoder (NEW)                   │
│  - Bidirectional Transformer (6 layers)     │
│  - Sinusoidal positional encoding           │
│  - Full self-attention (no causal mask)     │
│  Output: (batch, seq_len, d_model=512)      │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Boundary Query Decoder (NEW)               │
│  - Learned query embeddings (N=9 queries)   │
│  - Cross-attention to trajectory            │
│  - Self-attention between queries           │
│  - 6 decoder layers                         │
│  Output: (batch, num_queries, d_model)      │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Prediction Heads (NEW)                     │
│  - Position head: → (batch, N) ∈ [0, 1]    │
│  - Confidence head: → (batch, N) ∈ [0, 1]  │
└─────────────────────────────────────────────┘
```

### 2.2 Trajectory Encoder

**Purpose**: Encode temporal sequence with full bidirectional context

**Design Choices**:
- **Bidirectional attention**: No causal masking (we see full trajectory at inference)
- **6 layers**: Balance between capacity and speed
- **d_model=512**: Standard transformer size
- **Pre-norm**: Better training stability
- **Sinusoidal PE**: Fixed positional encoding (alternatives: learned, RoPE)

**Input/Output**:
```
Input:  (batch, seq_len, feature_dim) - e.g., (32, 6, 512) for 6 DINO frames
Output: (batch, seq_len, 512) - encoded trajectory
```

### 2.3 Boundary Query Decoder

**Purpose**: Learn specialized "boundary detector" queries that attend to trajectory

**Key Concepts**:
- **Query embeddings**: Learned vectors initialized randomly, trained to specialize
  - Query 0 might learn to detect first boundary
  - Query 1 might learn to detect second boundary
  - Etc.
- **Positional queries**: Additional positional encoding for queries (helps specialization)
- **Cross-attention**: Queries look at trajectory to find boundaries
- **Self-attention**: Queries communicate to avoid duplicates

**Number of Queries**:
- Set to N=9 (max boundaries expected in an 18-frame window)
- If fewer boundaries exist, unused queries learn to output low confidence
- If more boundaries exist (rare), we miss some (but can tune N)

**Input/Output**:
```
Input:  Encoded trajectory (batch, seq_len, 512)
Output: Decoded queries (batch, 9, 512)
```

### 2.4 Prediction Heads

**Position Head**:
- Predicts normalized position in [0, 1]
- Multiply by window_size to get frame index
- Uses sigmoid activation to ensure [0, 1]

**Confidence Head**:
- Predicts probability that this query detected a real boundary
- Low confidence (< 0.5) = "no boundary found by this query"
- High confidence (> 0.5) = "real boundary at predicted position"
- Similar to DETR's objectness score

**Input/Output**:
```
Input:  Decoded queries (batch, 9, 512)
Output:
  - positions: (batch, 9) ∈ [0, 1]
  - confidences: (batch, 9) ∈ [0, 1]
```

---

## 3. Loss Function: Hungarian Matching

### 3.1 Why Hungarian Matching?

**Problem**: How to match predictions to ground truth?
```
Ground truth: [4, 8, 14] (3 boundaries)
Predictions:  9 queries with positions [4.1, 7.9, 3.8, 14.2, 12.3, 2.1, 16.5, 10.1, 5.5]
                         confidences [0.9, 0.85, 0.05, 0.92, 0.08, 0.02, 0.04, 0.03, 0.06]

Which prediction should supervise which GT?
```

**Solution**: Hungarian algorithm finds optimal one-to-one matching that minimizes total cost

### 3.2 Cost Matrix

For each (prediction, GT) pair, compute cost:
```
cost(pred_i, gt_j) = λ_pos * |pos_i - pos_j| + λ_conf * (1 - conf_i)
```

**Intuition**:
- Low cost if prediction is close to GT (small position error)
- Low cost if prediction is confident (high conf_i)
- High cost if prediction is far from GT or not confident

**Hyperparameters**:
- λ_pos = 5.0 (position weight)
- λ_conf = 2.0 (confidence weight)

### 3.3 Loss Computation

After matching, compute:

1. **Position Loss** (L1): For matched pairs only
   ```
   L_pos = (1/N_matched) * Σ |pred_pos_i - gt_pos_j|
   ```

2. **Confidence Loss** (BCE): For all queries
   - Matched queries: target = 1.0 (should be confident)
   - Unmatched queries: target = 0.0 (should not be confident)
   ```
   L_conf = BCE(pred_conf, target_conf)
   ```
   - Apply class weighting: upweight positives (rare) vs negatives (common)

3. **Total Loss**:
   ```
   L_total = λ_pos * L_pos + λ_conf * L_conf
   ```

### 3.4 Training Dynamics

**Early training**:
- Queries output random positions
- Hungarian matching finds best assignment (even if poor)
- Gradients guide queries to improve

**Mid training**:
- Queries start to specialize
- Some queries consistently detect early boundaries
- Others detect late boundaries

**Late training**:
- Queries converge to stable specializations
- Unused queries (for windows with few boundaries) learn low confidence

---

## 4. Data Pipeline Modifications

### 4.1 Current Format
```python
{
    'tokenized_actions': (batch, seq_len),
    'dino_features': (batch, num_frames * 1024),
    'labels': (batch, 360)  # Dense smoothed labels
}
```

### 4.2 New Format (Set-Based)
```python
{
    'tokenized_actions': (batch, seq_len),
    'dino_features': (batch, num_frames * 1024),
    'targets': [
        {
            'positions': tensor([0.22, 0.44, 0.78]),  # Normalized [0, 1]
            'labels': tensor([1, 1, 1]),  # All ones (for compatibility)
            'num_boundaries': 3
        },
        # ... for each sample in batch
    ]
}
```

### 4.3 Label Preparation

```python
def prepare_detr_target(boundaries, window_start, window_size):
    """
    Extract boundaries within window and normalize.

    Args:
        boundaries: [25, 68, 142, ...] (full trajectory)
        window_start: 60
        window_size: 18

    Returns:
        {
            'positions': [0.444],  # (68-60)/18
            'num_boundaries': 1
        }
    """
    window_end = window_start + window_size
    boundaries_in_window = [b for b in boundaries if window_start <= b < window_end]
    normalized = [(b - window_start) / window_size for b in boundaries_in_window]
    return {
        'positions': torch.tensor(normalized),
        'labels': torch.ones(len(normalized)),
        'num_boundaries': len(normalized)
    }
```

---

## 5. Training Strategy

### 5.1 Hyperparameters

```yaml
model:
  architecture: boundary_detr
  num_boundary_queries: 9
  d_model: 512
  num_encoder_layers: 6
  num_decoder_layers: 6
  num_heads: 8
  dropout: 0.1

loss:
  position_weight: 5.0
  confidence_weight: 2.0
  eos_coef: 0.1  # Reweight for unmatched queries

training:
  learning_rate: 1e-4  # Lower for transformer
  weight_decay: 1e-4
  warmup_steps: 2000
  batch_size: 64
  gradient_clip: 0.1  # IMPORTANT for DETR stability
  num_epochs: 200
```

### 5.2 Critical Training Tips

1. **Gradient Clipping**: DETR is sensitive to gradient explosions
   - Use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)`

2. **Learning Rate Warmup**: Start with small LR, increase gradually
   - First 2000 steps: linear warmup from 1e-7 to 1e-4

3. **Loss Balancing**: Monitor position and confidence losses separately
   - If position loss dominates, decrease λ_pos
   - If confidence loss dominates, increase λ_pos

4. **Convergence**: DETR typically takes longer to converge than dense methods
   - Expect 100-200 epochs
   - Monitor F1 score on validation set

### 5.3 Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        predictions = model(batch['tokenized_actions'], None, batch['dino_features'])

        # Loss with Hungarian matching
        losses = criterion(predictions, batch['targets'])
        loss = losses['loss_total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
```

---

## 6. Inference Pipeline

### 6.1 Single Window Prediction

```python
predictions = model(window_features)
positions = predictions['pred_positions'][0]  # (9,)
confidences = predictions['pred_confidences'][0]  # (9,)

# Filter by confidence
valid_mask = confidences > 0.5
boundaries = (positions[valid_mask] * window_size).long()
# Example: [4, 8, 14] (frame indices in window)
```

### 6.2 Full Trajectory with Sliding Windows

**Challenge**: Overlapping windows produce duplicate predictions

**Solution**: NMS across windows
```python
all_predictions = []
for window in sliding_windows(trajectory, window_size=18, stride=5):
    boundaries = predict_window(window)
    # Convert to global frames
    global_boundaries = [start + b for b in boundaries]
    all_predictions.extend(global_boundaries)

# NMS: merge predictions within 3 frames
final_boundaries = non_maximum_suppression(all_predictions, threshold=3)
```

**NMS Strategy**:
1. Collect all predictions from all windows with their confidences
2. Sort by confidence (descending)
3. Greedily keep high-confidence predictions
4. Suppress nearby predictions (within 3 frames)

---

## 7. Evaluation Metrics

### 7.1 Set-Based Metrics

**Precision, Recall, F1** with tolerance:
```python
tolerance = 2  # ±2 frames acceptable

for pred in predictions:
    for gt in ground_truth:
        if abs(pred - gt) <= tolerance:
            matched = True
            break

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

### 7.2 Additional Metrics

- **Mean Absolute Error** (for matched boundaries): `mean(|pred - gt|)`
- **Detection Rate at Tolerance T**: Percentage of GT boundaries detected within T frames
- **Per-task analysis**: Some tasks may be easier than others

---

## 8. Implementation Plan

### Phase 1: Core Architecture (Week 1)
- [ ] Implement `TrajectoryEncoder`
- [ ] Implement `BoundaryQueryDecoder`
- [ ] Implement `BoundaryPredictionHeads`
- [ ] Implement `BoundaryDETR` (full model)
- [ ] Unit tests for forward pass

### Phase 2: Loss Function (Week 2)
- [ ] Implement `HungarianMatcher`
- [ ] Implement `SetCriterion`
- [ ] Test on synthetic data
- [ ] Verify gradient flow

### Phase 3: Data Pipeline (Week 3)
- [ ] Modify dataset to return set-based targets
- [ ] Update batch collation
- [ ] Test with existing dataloaders

### Phase 4: Training (Week 4)
- [ ] Create config files
- [ ] Initial training run (1-2 epochs)
- [ ] Debug convergence issues
- [ ] Hyperparameter sweep

### Phase 5: Inference (Week 5)
- [ ] Single window inference
- [ ] Sliding window with NMS
- [ ] Evaluation script
- [ ] Visualization tools

### Phase 6: Evaluation (Week 6)
- [ ] Compare with baseline
- [ ] Ablation studies
- [ ] Error analysis
- [ ] Documentation

---

## 9. Expected Challenges & Mitigations

### Challenge 1: Slow Convergence
**Symptom**: Loss plateaus, F1 stays low
**Solutions**:
- Increase warmup steps
- Decrease learning rate
- Check gradient norms
- Simplify model (fewer layers)

### Challenge 2: Overconfident Predictions
**Symptom**: All queries output confidence ~1.0
**Solutions**:
- Increase eos_coef (upweight negative class)
- Add label smoothing to confidence targets
- Check class balance in dataset

### Challenge 3: Duplicate Predictions
**Symptom**: Multiple queries predict same boundary
**Solutions**:
- Check self-attention in decoder (queries should communicate)
- Tune Hungarian matching costs
- Add diversity loss (auxiliary loss to encourage different positions)

### Challenge 4: Missing Boundaries
**Symptom**: Recall is low
**Solutions**:
- Increase num_queries
- Increase position weight in loss
- Check if ground truth is correct

### Challenge 5: Poor Window Aggregation
**Symptom**: Good per-window performance, poor full-trajectory
**Solutions**:
- Tune NMS threshold
- Use confidence-weighted aggregation
- Process longer windows (if memory allows)

---

## 10. Comparison with Baseline

| Aspect | Baseline (Dense) | DETR (Set-Based) |
|--------|-----------------|------------------|
| **Output** | Fixed 360-dim vector | Variable set of N boundaries |
| **Loss** | MSE on probabilities | Hungarian + L1 + BCE |
| **Post-processing** | Peak detection + NMS | Confidence threshold (simple) |
| **Multi-boundary** | Via multiple peaks | Native (multiple queries) |
| **Training** | Faster convergence | Slower but more principled |
| **Inference** | Denser predictions | Sparse direct output |
| **Interpretability** | Probability curve | Explicit boundaries |

**When DETR is Better**:
- Multiple boundaries per window (common)
- Need exact boundary positions
- Want end-to-end learning

**When Baseline is Better**:
- Need fast training iterations
- Simple single-boundary tasks
- Well-tuned post-processing available

---

## 11. Future Extensions

### Extension 1: Multi-Resolution Queries
Use queries at different temporal scales:
- Fine-grained queries: detect fast transitions
- Coarse queries: detect slow task changes

### Extension 2: Boundary Classification
Add classification head to predict boundary type:
```python
boundary_type_head = nn.Linear(512, num_boundary_types)
# Types: grasp, release, reach, transport, etc.
```

### Extension 3: Relation Modeling
Model relationships between boundaries (e.g., grasp → transport → release)

### Extension 4: Uncertainty Estimation
Predict uncertainty/variance for each boundary position

---

## 12. References

- **DETR**: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
- **MS-TCN**: Farha & Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network", CVPR 2019
- **ASFormer**: Yi et al., "ASFormer: Transformer for Action Segmentation", BMVC 2021
- **Hungarian Algorithm**: Kuhn, "The Hungarian Method for Assignment Problems", 1955

---

## Appendix A: Model Size Estimation

```
TrajectoryEncoder:
  - 6 layers × (8 heads × 512 d_model) ≈ 12M params

BoundaryQueryDecoder:
  - Query embeddings: 9 × 512 = 4.6K
  - 6 decoder layers ≈ 12M params

Prediction Heads:
  - Position: 512 → 256 → 128 → 1 ≈ 164K
  - Confidence: similar ≈ 164K

Total (DETR components): ~25M parameters
Total (with encoders): ~35M parameters
```

**Memory**: ~200MB for model, ~500MB for batch (64)
**Training**: Fits on single GPU (16GB+ recommended)

---

## Appendix B: Hyperparameter Tuning Guide

**Start with defaults**:
- num_queries = 9
- d_model = 512
- num_layers = 6 (encoder & decoder)
- lr = 1e-4
- position_weight = 5.0
- confidence_weight = 2.0

**If underfitting** (training loss high):
- Increase d_model (512 → 768)
- Increase num_layers (6 → 8)
- Decrease weight_decay

**If overfitting** (val loss >> train loss):
- Add dropout (0.1 → 0.2)
- Increase weight_decay
- Reduce model size

**If converging slowly**:
- Increase learning rate (1e-4 → 2e-4)
- Decrease batch size (better gradients)
- Increase warmup steps

**If missing boundaries**:
- Increase num_queries (9 → 12)
- Increase position_weight (more emphasis on localization)
- Check data: are labels correct?

**If duplicate predictions**:
- Check decoder self-attention
- Tune Hungarian matching costs
- Add auxiliary loss for diversity

---

**End of Design Document**
