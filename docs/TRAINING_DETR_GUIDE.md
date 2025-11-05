# Training DETR Model: Quick Start Guide

**Status**: ✅ Ready to Train
**Date**: 2025-11-05

---

## What Was Done

### 1. Core Implementation ✅
- ✅ BoundaryDETR model (51M parameters)
- ✅ HungarianMatcher for optimal assignment
- ✅ SetCriterion loss function
- ✅ All unit tests passing (7/7)

### 2. Data Pipeline ✅
- ✅ `dense_labels_to_set_targets()` - Converts dense labels to DETR format
- ✅ `collate_detr_batch()` - Custom collate function
- ✅ `compute_detr_metrics()` - Precision/recall/F1 metrics

### 3. Training Infrastructure ✅
- ✅ `train_detr()` - Custom training loop
- ✅ Updated `scripts/train.py` to support DETR
- ✅ Automatic detection of DETR architecture
- ✅ Gradient clipping (critical for stability)
- ✅ Callback support for W&B logging

### 4. Configuration ✅
- ✅ Base config: `configs/base/model_boundary_detr.yaml`
- ✅ Experiment config: `configs/experiments/multi_task_327_352_356_detr.yaml`

---

## How to Train

### Option 1: Single Task (Task 327 - Already configured!)

```bash
python3 scripts/train.py --config ./configs/experiments/multi_task_327_352_356_detr.yaml
```

The config is already set for task 327 only (lines 46-47):
```yaml
train_task_ids: [327]
eval_task_ids: [327]
```

### Option 2: Multi-Task (All tasks)

Edit config to uncomment lines 44-45:
```yaml
train_task_ids: [327, 352, 354, 356, 357, 358, 359, 360, 365, 366, 367, 368, 372, 374, 375, 376, 377, 378, 380, 384, 385, 389, 390]
eval_task_ids: [327, 352, 354, 356, 357, 358, 359, 360, 365, 366, 367, 368, 372, 374, 375, 376, 377, 378, 380, 384, 385, 389, 390]
```

Then run:
```bash
python3 scripts/train.py --config ./configs/experiments/multi_task_327_352_356_detr.yaml
```

### Option 3: Quick Test (Fewer epochs)

Override config from CLI:
```bash
python3 scripts/train.py \
  --config ./configs/experiments/multi_task_327_352_356_detr.yaml \
  training.num_epochs=5 \
  data.batch_size=32
```

---

## What the Training Script Does

### 1. Data Loading
```
✓ Loads trajectories from /shared/projects/agibot/full_dataset/
✓ Creates sliding windows (18 frames @ 2fps)
✓ Extracts DINO features (6 frames per window)
✓ Tokenizes proprioceptive data
```

### 2. Model Creation
```
✓ Creates BoundaryDETR with 51M parameters
✓ Uses AdamW optimizer (better for transformers)
✓ Sets up warmup + cosine annealing scheduler
```

### 3. Data Collation (DETR-specific)
```
✓ Converts dense smoothed labels → set of boundaries
✓ Example: [0, 0.1, 0.9, 1.0, 0.9, ...] → [(0.16, conf), (0.50, conf)]
✓ Handles variable # of boundaries per window
```

### 4. Training Loop
```
For each batch:
  1. Forward pass → predict (positions, confidences)
  2. Hungarian matching → find optimal assignment
  3. Compute losses:
     - Position loss (L1) for matched predictions
     - Confidence loss (BCE) for all queries
  4. Backward pass
  5. Gradient clipping (max_norm=0.1) ← CRITICAL
  6. Optimizer step
```

### 5. Evaluation
```
Every epoch:
  ✓ Compute validation loss
  ✓ Extract boundaries (confidence > 0.5)
  ✓ Match to GT (tolerance=±2 frames)
  ✓ Compute precision, recall, F1
  ✓ Save best model
```

---

## Expected Output

```
================================================================================
Loading configuration...
================================================================================
Experiment: multi_task_327_352_356_detr
Config file: ./configs/experiments/multi_task_327_352_356_detr.yaml

================================================================================
Loading data...
================================================================================
Meta directory: /shared/projects/agibot/full_dataset/
Training tasks: [327]
Evaluation tasks: [327]

✓ Created lazy dataset with XXXX samples from XX trajectories
Using DETR collate function (set-based targets)

================================================================================
Creating model...
================================================================================
Model parameters: 51,303,042

Optimizer: AdamW (lr=0.0001, wd=0.0001)
Scheduler: cosine
Loss: SetCriterion (Hungarian matching)
  Position weight: 5.0
  Confidence weight: 2.0
  EOS coefficient: 0.1

================================================================================
Initializing trainer...
================================================================================
Using custom DETR training loop

================================================================================
Starting training...
================================================================================

Epoch 1/200
================================================================================
Training: 100%|████████| 500/500 [02:30<00:00, 3.33it/s]
  Batch 10/500: Loss=6.3245 (pos=1.2341, conf=4.8904), LR=0.000005
  Batch 20/500: Loss=5.9123 (pos=1.1234, conf=4.4889), LR=0.000010
  ...

Train Loss: 5.2341 (pos=0.9876, conf=4.2465)
Train Metrics: P=0.234, R=0.312, F1=0.267

Validation: 100%|████████| 100/100 [00:20<00:00, 4.89it/s]
Val Loss: 4.8765 (pos=0.8234, conf=3.9531)
Val Metrics: P=0.456, R=0.423, F1=0.439
✓ Saved best model (val_loss=4.8765, val_f1=0.439)
```

---

## Training Parameters

### Model Architecture
```yaml
architecture: boundary_detr
num_boundary_queries: 9      # Max boundaries per 18-frame window
d_model: 512                 # Transformer hidden dim
num_encoder_layers: 6
num_decoder_layers: 6
num_attention_heads: 8
```

### Training Hyperparameters
```yaml
learning_rate: 1e-4          # Conservative for transformer
weight_decay: 1e-4
warmup_steps: 2000           # 2000 steps warmup
batch_size: 64               # Smaller than baseline (4096)
num_epochs: 200              # DETR needs more epochs
gradient_clip: 0.1           # CRITICAL for stability
```

### Loss Weights
```yaml
loss_position_weight: 5.0    # Weight for position L1 loss
loss_confidence_weight: 2.0  # Weight for confidence BCE
loss_eos_coef: 0.1          # Upweight positive class (boundaries are rare)
```

---

## Monitoring Training

### Key Metrics to Watch

1. **Loss Components**
   - `train_loss`: Total loss (should decrease steadily)
   - `train_loss_position`: Position L1 (should drop to ~0.1-0.5)
   - `train_loss_confidence`: Confidence BCE (should drop to ~0.3-0.8)

2. **Performance Metrics**
   - `val_f1`: F1 score (target: >0.7 after 50-100 epochs)
   - `val_precision`: Precision (target: >0.8)
   - `val_recall`: Recall (target: >0.7)

3. **Signs of Issues**
   - Loss = NaN → Gradient explosion (check gradient_clip)
   - Loss not decreasing → Learning rate too low (increase to 2e-4)
   - F1 stuck at 0 → Model not converging (check data/labels)

### W&B Logging (if enabled)

Go to https://wandb.ai/your-project/task-segmentation

Plots to check:
- Loss curves (train vs val)
- F1 score over time
- Precision/recall curves
- Learning rate schedule

---

## Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Reduce batch size
python3 scripts/train.py --config ... data.batch_size=32
```

### Issue 2: Training Too Slow
```bash
# Reduce num_workers or model size
python3 scripts/train.py --config ... \
  data.num_workers=8 \
  model.d_model=256 \
  model.num_encoder_layers=4
```

### Issue 3: Loss is NaN
```
Check:
1. Gradient clipping is enabled (gradient_clip_val=0.1)
2. Learning rate not too high (try 5e-5)
3. Data has valid boundaries (check dataset)
```

### Issue 4: No Boundaries Detected (F1=0)
```
Check:
1. confidence_threshold (try 0.3 instead of 0.5)
2. Data labels are correct (check dense_labels)
3. Model has enough capacity (increase d_model)
```

---

## Next Steps After Training

### 1. Evaluate on Test Set
```bash
python3 scripts/evaluate.py \
  --checkpoint ./checkpoints/multi_task_327_352_356_detr/best_model.pt \
  --split test \
  --output results.json
```

### 2. Compare with Baseline
```bash
# Train baseline transformer for comparison
python3 scripts/train.py --config ./configs/experiments/multi_task_327_352_356_transformer.yaml
```

### 3. Hyperparameter Tuning
Try different:
- `num_boundary_queries` (6, 9, 12)
- `learning_rate` (5e-5, 1e-4, 2e-4)
- `loss_position_weight` (3.0, 5.0, 10.0)

### 4. Ablation Studies
Test impact of:
- Focal loss vs standard BCE
- Different # of encoder/decoder layers
- Warmup steps

---

## Files Created/Modified

```
New Files:
  models/boundary_detr.py                     (1000 lines)
  models/detr_loss.py                         (500 lines)
  training/detr_utils.py                      (200 lines)
  training/detr_trainer.py                    (300 lines)
  tests/test_boundary_detr.py                 (500 lines)
  docs/DETR_BOUNDARY_DETECTION_DESIGN.md      (600 lines)
  docs/DETR_IMPLEMENTATION_SUMMARY.md         (400 lines)
  docs/TRAINING_DETR_GUIDE.md                 (this file)

Modified Files:
  models/__init__.py                          (added DETR exports)
  scripts/train.py                            (added DETR support)
  configs/experiments/multi_task_327_352_356_detr.yaml (new config)
```

---

## Success Criteria

After 100-200 epochs, expect:
- ✅ Validation loss < 2.0
- ✅ Validation F1 > 0.70
- ✅ Validation precision > 0.75
- ✅ Validation recall > 0.65

If achieved, model is ready for deployment!

---

## Questions?

Check documentation:
- Architecture: `docs/DETR_BOUNDARY_DETECTION_DESIGN.md`
- Implementation: `docs/DETR_IMPLEMENTATION_SUMMARY.md`
- Tests: `python tests/test_boundary_detr.py`

Good luck with training! 🚀
