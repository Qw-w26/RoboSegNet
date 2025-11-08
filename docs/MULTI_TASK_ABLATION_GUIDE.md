# Multi-Task Ablation Study Guide

## Overview

`scripts/run_multi_task_ablation.py` performs multi-task ablation studies where:
- **Each ablation trains ONE model on ALL training tasks** (multi-task learning)
- **Evaluates on BOTH in-distribution (training tasks) AND out-of-distribution tasks**
- **Uses 8 GPUs with torchrun for distributed training**

## Key Differences from Single-Task Ablation

| Aspect | Single-Task (`run_ablation_study.py`) | Multi-Task (`run_multi_task_ablation.py`) |
|--------|----------------------------------------|-------------------------------------------|
| Training | One model per task | One model for all tasks |
| Task IDs | Each trial trains on single task | Each trial trains on all training tasks |
| Training Command | `python scripts/train.py` | `torchrun --nproc_per_node=8 scripts/train.py` |
| Evaluation | On single task | On in-dist AND OOD tasks |
| Output | Per-task results | Aggregate + per-task results |

## Dataset Configuration

### Training Tasks (In-Distribution)
Default: 327, 352, 354, 356, 357, 358, 359, 360, 365, 366, 367, 368, 372, 374, 375, 376, 377, 378, 380, 384, 385, 389, 390 (23 tasks)

### OOD Tasks (Auto-detected >= 400)
Auto-detected: 410, 414, 421, 422, 424, 440 (6 tasks)

## Ablation Configurations

All ablation configurations are **IDENTICAL** to `run_ablation_study.py`:

### 1. Pure Proprioceptive
```yaml
model.use_proprio: True
model.use_dino: False
```

### 2. Pure Visual
```yaml
model.use_proprio: False
model.use_dino: True
```

### 3. MLP + MSE
```yaml
model.architecture: mlp
training.loss_type: mse_raw
training.optimizer: adam
data.batch_size: 128
```

### 4. DETR No Cross Attention
```yaml
model.architecture: boundary_detr
model.use_proprio: True
model.use_dino: True
model.fusion_mode: add
```

### 5. Cross Attention + MSE
```yaml
model.architecture: cross_modal_transformer
training.loss_type: mse_raw
training.optimizer: adamw
```

### 6. Multi-Task Baseline
Uses default settings from base config

## Usage

### Basic Usage (All Ablations)
```bash
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml
```

### Quick Test (5 epochs, 1000 samples)
```bash
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --quick-test
```

### Run Specific Ablations
```bash
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --ablations pure_proprio pure_visual multi_task_baseline
```

### Sequential Execution (1 at a time)
Since each trial uses 8 GPUs, run one at a time:
```bash
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --num-parallel 1
```

### Use 4 GPUs Instead of 8
```bash
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --gpus-per-trial 4
```

### Custom Task Sets
```bash
# Custom training tasks
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --train-task-ids 327 352 356 357

# Custom OOD tasks
python scripts/run_multi_task_ablation.py \
    --base-config configs/experiments/multi_task_327_352_356_detr.yaml \
    --ood-task-ids 410 414 421
```

## Output Files

The script generates comprehensive results in `multi_task_ablation_results/`:

### 1. Per-Task Results
- `multi_task_ablation_summary_indist.csv` - Detailed results for each in-distribution task
- `multi_task_ablation_summary_ood.csv` - Detailed results for each OOD task

### 2. Aggregate Results (Mean Across Tasks)
- `multi_task_ablation_aggregate_indist.csv` - **Average performance on in-dist tasks**
- `multi_task_ablation_aggregate_ood.csv` - **Average performance on OOD tasks**
- `multi_task_ablation_aggregate_comparison.csv` - **Side-by-side comparison of in-dist vs OOD**

### 3. Raw Results
- `multi_task_ablation_results.json` - Complete results with all metadata
- `evaluations/<ablation_name>/indist_evaluation_results.json` - Raw in-dist evaluation
- `evaluations/<ablation_name>/ood_evaluation_results.json` - Raw OOD evaluation

### 4. Configs
- `configs/<ablation_name>.yaml` - Generated config for each ablation

## Reported Metrics

### Boundary Detection Metrics
- **F1@1.0, F1@2.0, F1@3.0, F1@4.0, F1@5.0** - Boundary F1 at different frame tolerances
- **AP@1.0, AP@2.0, AP@3.0, AP@4.0, AP@5.0** - Average Precision at different tolerances
- **mAP** - mean Average Precision

### Action Segmentation Metrics (Standard)
- **Seg-F1@10%, Seg-F1@25%, Seg-F1@50%** - Segmental F1 at IoU thresholds
- **Edit-Dist** - Levenshtein distance for boundary sequences
- **Norm-Edit-Dist** - Normalized edit distance (0-1)
- **Acc** - Framewise accuracy

### Frame-Level Metrics
- **precision** - Frame-level precision
- **recall** - Frame-level recall
- **f1_score** - Frame-level F1
- **accuracy** - Frame-level accuracy

## Final Summary Display

The script displays:

1. **Per-Task Results** for both in-dist and OOD tasks
2. **Aggregate Performance** showing mean metrics across tasks for:
   - In-distribution tasks (23 tasks)
   - Out-of-distribution tasks (6 tasks)
3. **Comparison Table** with side-by-side in-dist vs OOD performance

## Resume Support

The script automatically:
- Saves results after each trial completes
- Skips already-completed trials when rerun
- Loads previous results for incremental updates

## Example Output Structure

```
multi_task_ablation_results/
├── configs/
│   ├── multi_task_ablation_baseline.yaml
│   ├── multi_task_ablation_pure_proprio.yaml
│   ├── multi_task_ablation_pure_visual.yaml
│   ├── multi_task_ablation_mlp_mse.yaml
│   ├── multi_task_ablation_detr_no_cross_attn.yaml
│   └── multi_task_ablation_cross_attn_mse.yaml
├── evaluations/
│   ├── multi_task_ablation_baseline/
│   │   ├── indist/evaluation_results.json
│   │   ├── ood/evaluation_results.json
│   │   ├── indist_evaluation_results.json
│   │   └── ood_evaluation_results.json
│   └── ...
├── multi_task_ablation_results.json                    # Raw results
├── multi_task_ablation_summary_indist.csv              # Per-task in-dist
├── multi_task_ablation_summary_ood.csv                 # Per-task OOD
├── multi_task_ablation_aggregate_indist.csv            # ⭐ Mean in-dist performance
├── multi_task_ablation_aggregate_ood.csv               # ⭐ Mean OOD performance
└── multi_task_ablation_aggregate_comparison.csv        # ⭐ In-dist vs OOD comparison
```

## Notes

- Each ablation training uses **8 GPUs** via `torchrun`
- Recommended to run **sequentially** (`--num-parallel 1`) to avoid resource conflicts
- Results are saved **incrementally** - safe to interrupt and resume
- All ablation configs match `run_ablation_study.py` exactly (only experiment names differ)
