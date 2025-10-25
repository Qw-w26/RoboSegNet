# Task Segmentation Codebase Merge Plan

**Date:** October 24, 2025
**Author:** System Analysis
**Version:** 1.0

---

## Executive Summary

This document outlines the comprehensive plan to merge two task segmentation codebases:
- **Current codebase** (`/home/syx/ucsf/task_seg`): Good modular structure, benchmarking framework
- **Andrew's codebase** (`/home/syx/keyframe_detection/andrew_code`): Superior neural architecture, excellent logging

**Goal:** Create a unified, production-ready codebase that combines the best features of both while adding critical missing components.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Architecture Design](#architecture-design)
3. [Implementation Phases](#implementation-phases)
4. [Migration Guide](#migration-guide)
5. [Testing Strategy](#testing-strategy)
6. [Timeline](#timeline)

---

## Current State Analysis

### Current Codebase Strengths
- ✅ Modular dataset abstraction (`BaseSegmentationDataset`)
- ✅ Multiple dataset backends (HuggingFace, local)
- ✅ Comprehensive benchmarking framework
- ✅ Multiple baseline methods (Spectral, ChangePoint, GMM)
- ✅ Various post-processing strategies (NMS, adaptive threshold)
- ✅ Good project organization

### Current Codebase Weaknesses
- ❌ Basic neural network architecture
- ❌ Weak configuration management (hardcoded parameters)
- ❌ Limited WandB integration
- ❌ No train/val/test splits
- ❌ Missing comprehensive evaluation metrics
- ❌ No unit tests

### Andrew's Codebase Strengths
- ✅ Advanced multi-modal neural architecture (DINO + Fast-DROID)
- ✅ Excellent WandB integration (config tracking, visualizations)
- ✅ Ray-parallelized DINO extraction (4 GPU workers)
- ✅ Structured configuration with dataclasses
- ✅ Comprehensive metrics (F1@IoU with Hungarian matching)
- ✅ Proper label smoothing implementations
- ✅ Train/val splitting

### Andrew's Codebase Weaknesses
- ❌ Less modular dataset abstraction
- ❌ No benchmarking framework
- ❌ No test set (only train/val)
- ❌ Hardcoded paths
- ❌ Minimal documentation
- ❌ No unit tests
- ❌ Code quality issues (breakpoints, placeholder metrics)

---

## Architecture Design

### New Directory Structure

```
task_seg/                           # Root directory
├── configs/                        # NEW: Configuration files
│   ├── config.yaml                 # Default configuration
│   ├── model/                      # Model configs
│   │   ├── boundary_mlp.yaml
│   │   └── boundary_mlp_large.yaml
│   ├── dataset/                    # Dataset configs
│   │   ├── agibot_327.yaml
│   │   └── agibot_352.yaml
│   ├── training/                   # Training configs
│   │   ├── default.yaml
│   │   └── fast_debug.yaml
│   └── experiment/                 # Experiment-specific configs
│       └── ablation_study.yaml
│
├── data/                           # MERGED: Dataset handling
│   ├── base_dataset.py             # Keep: Abstract interface
│   ├── agibot_dataset.py           # Keep: Local loading
│   ├── agibot_hf_direct.py         # Keep: HuggingFace integration
│   ├── training_dataset.py         # PORT: Andrew's TrainingDataset (refactored)
│   ├── dataloader.py               # NEW: Unified dataloader factory
│   └── splits.py                   # NEW: Train/val/test splitting
│
├── features/                       # RENAMED: from embedding/
│   ├── dino_extractor.py           # PORT: Andrew's Ray version
│   ├── droid_extractor.py          # PORT: Andrew's tokenizer
│   ├── fast_droid_embeddings.py    # Keep: Statistical features
│   └── feature_cache.py            # NEW: Unified caching layer
│
├── models/                         # MERGED: Neural networks
│   ├── boundary_prediction.py      # PORT: Andrew's BoundaryPredictionMLP
│   ├── encoders.py                 # PORT: TinyFASTEncoder, DinoFeatureEncoder
│   ├── losses.py                   # MERGE: Both focal losses
│   └── postprocessing.py           # Keep: NMS variants
│
├── training/                       # MAJOR REFACTOR
│   ├── config.py                   # NEW: Training configuration dataclasses
│   ├── trainer.py                  # MERGE: Best of both trainers
│   ├── callbacks.py                # NEW: Callback system
│   ├── label_smoothing.py          # MERGE: All label smoothers
│   └── data_augmentation.py        # NEW: Temporal augmentation
│
├── evaluation/                     # PORT: from andrew_code
│   ├── metrics.py                  # PORT: Comprehensive BoundaryMetrics
│   ├── f1_iou.py                   # PORT: Hungarian matching
│   ├── evaluator.py                # NEW: Unified evaluation pipeline
│   └── visualizer.py               # MERGE: Visualization utilities
│
├── benchmark/                      # Keep: Current framework
│   ├── benchmark_runner.py         # Keep with improvements
│   ├── methods/                    # Keep: Baseline methods
│   │   ├── base_method.py
│   │   └── clustering_methods.py
│   └── metrics/                    # INTEGRATE with evaluation/
│       └── segmentation_metrics.py
│
├── scripts/                        # NEW: Clean entry points
│   ├── prepare_data.py             # NEW: Dataset preparation
│   ├── train.py                    # PORT: Andrew's with Hydra
│   ├── evaluate.py                 # NEW: Comprehensive evaluation
│   ├── inference.py                # NEW: Production inference
│   └── benchmark.py                # REFACTOR: From examples/
│
├── tests/                          # NEW: Testing infrastructure
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
├── utils/                          # NEW: Shared utilities
│   ├── paths.py                    # Centralized path management
│   ├── reproducibility.py          # Seed management
│   ├── ray_utils.py                # Ray initialization helpers
│   └── logging.py                  # Logging utilities
│
├── docs/                           # EXPANDED: Documentation
│   ├── MERGE_PLAN.md               # This document
│   ├── architecture.md             # System architecture
│   ├── data_pipeline.md            # Dataset guide
│   ├── training_guide.md           # Training walkthrough
│   ├── evaluation_guide.md         # Evaluation metrics
│   └── api_reference/              # Auto-generated docs
│
├── outputs/                        # NEW: Training outputs
│   └── {experiment_name}/
│       ├── checkpoints/
│       ├── logs/
│       └── config.yaml
│
├── README.md                       # UPDATED: Comprehensive guide
├── requirements.txt                # UPDATED: Dependencies
├── setup.py                        # NEW: Package installation
└── pyproject.toml                  # NEW: Modern Python packaging
```

---

## Implementation Phases

### Phase 1: Foundation & Data Pipeline (Week 1)

#### 1.1 Directory Restructuring
- Create new directory structure
- Set up package structure with `__init__.py`
- Create placeholder files for new modules

#### 1.2 Data Splitting
**File:** `data/splits.py`

```python
class DatasetSplitter:
    """Stratified train/val/test splitting"""

    @staticmethod
    def split_trajectories(
        trajectories: List[TrajectoryData],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratify_by_task: bool = True
    ) -> Dict[str, List[int]]:
        """
        Split trajectories into train/val/test sets.

        Args:
            trajectories: List of trajectory data
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility
            stratify_by_task: Ensure each task in all splits

        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
```

**Features:**
- Stratified splitting by task_id
- Reproducible with fixed seeds
- Save split manifests to JSON
- Support for custom split ratios

#### 1.3 Feature Extraction Module
**Port from andrew_code:**
- `features/dino_extractor.py` - Ray-parallelized DINO extraction
- `features/droid_extractor.py` - Fast-DROID tokenization

**Improvements:**
- Remove hardcoded paths
- Add proper cache invalidation
- Better error handling
- Progress bars for extraction

#### 1.4 Data Preparation Script
**File:** `scripts/prepare_data.py`

```bash
# Example usage
python scripts/prepare_data.py \
  --task_ids 327 352 \
  --output_dir ./preprocessed_data \
  --extract_dino \
  --num_ray_workers 4 \
  --split_ratios 0.7 0.15 0.15 \
  --cache_features
```

**Workflow:**
1. Load trajectories from dataset
2. Create train/val/test splits
3. Extract DINO features (Ray parallelized)
4. Extract DROID tokens
5. Compute statistical features
6. Cache preprocessed data
7. Save split manifest

---

### Phase 2: Model & Training (Week 2)

#### 2.1 Port Model Architecture
**From andrew_code:**
- `models/boundary_prediction.py` - BoundaryPredictionMLP
- `models/encoders.py` - TinyFASTEncoder, DinoFeatureEncoder
- `models/losses.py` - Merge focal losses from both

**Improvements:**
- Remove hardcoded dimensions (use config)
- Add model summary/parameter counting
- Better documentation
- Type hints everywhere

#### 2.2 Configuration System
**Technology:** Hydra (flexible, YAML-based)

**Example config:** `configs/config.yaml`
```yaml
defaults:
  - model: boundary_mlp
  - dataset: agibot_327
  - training: default
  - _self_

# Global settings
seed: 42
output_dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
device: cuda

# Hydra settings
hydra:
  run:
    dir: ${output_dir}
```

**Example model config:** `configs/model/boundary_mlp.yaml`
```yaml
_target_: models.boundary_prediction.BoundaryPredictionMLP

# Architecture
vocab_size: 2048
token_embed_dim: 128
dino_feature_dim: 3072
prediction_window_size: 10
hidden_dims: [512, 256, 128]

# Regularization
dropout: 0.3
transformer_dropout: 0.3
embedding_dropout: 0.1
dino_encoder_dropout: 0.3

# Feature processing
feature_noise_std: 0.05
use_proprio: true
use_dino: true
```

#### 2.3 Training Module
**File:** `training/trainer.py`

```python
class Trainer:
    """Unified trainer with callback system"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: TrainingConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        self.callbacks = callbacks or self._default_callbacks()

    def train(self):
        """Main training loop with callbacks"""
        for epoch in range(self.config.num_epochs):
            # Callback: on_epoch_start
            self._trigger_callbacks('on_epoch_start', epoch)

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate_epoch()

            # Callback: on_epoch_end
            self._trigger_callbacks('on_epoch_end', epoch, train_metrics, val_metrics)
```

**Callbacks:** `training/callbacks.py`
```python
class WandBCallback(Callback):
    """Weights & Biases logging (from andrew)"""

class TensorBoardCallback(Callback):
    """TensorBoard logging (from current)"""

class CheckpointCallback(Callback):
    """Save best and periodic checkpoints"""

class EarlyStoppingCallback(Callback):
    """Stop training when metric plateaus"""

class LearningRateCallback(Callback):
    """Learning rate scheduling"""

class VisualizationCallback(Callback):
    """Log prediction visualizations"""
```

#### 2.4 Training Script
**File:** `scripts/train.py`

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Set seeds
    set_all_seeds(cfg.seed)

    # Load data
    train_dataset, val_dataset = create_datasets(cfg)

    # Create model
    model = hydra.utils.instantiate(cfg.model)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg.training,
        callbacks=[
            WandBCallback(cfg.wandb),
            TensorBoardCallback(cfg.tensorboard),
            CheckpointCallback(cfg.checkpoint),
            EarlyStoppingCallback(cfg.early_stopping)
        ]
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Default config
python scripts/train.py

# Override parameters
python scripts/train.py \
  model.hidden_dims=[1024,512,256] \
  training.learning_rate=1e-3 \
  training.batch_size=256

# Use different config
python scripts/train.py --config-name experiment/ablation_dino_only
```

---

### Phase 3: Evaluation & Testing (Week 3)

#### 3.1 Port Evaluation Metrics
**From andrew_code:**
- `evaluation/metrics.py` - BoundaryMetrics class
- `evaluation/f1_iou.py` - F1@IoU with Hungarian matching

**Features:**
- Frame-level metrics (accuracy, precision, recall, F1)
- Segment-level metrics with IoU matching
- Temporal metrics (boundary detection error)
- Confidence metrics (AUC-ROC, AUC-PR)
- Per-task and aggregated statistics

#### 3.2 Evaluation Pipeline
**File:** `evaluation/evaluator.py`

```python
class ModelEvaluator:
    """Comprehensive model evaluation"""

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        save_dir: Path,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with all metrics
        """

        results = {
            'frame_metrics': self._compute_frame_metrics(),
            'segment_metrics': self._compute_segment_metrics(),
            'f1_iou': self._compute_f1_iou(),
            'temporal_metrics': self._compute_temporal_metrics(),
            'confidence_metrics': self._compute_confidence_metrics(),
            'per_task_metrics': self._compute_per_task_metrics(),
        }

        # Save results
        self._save_results(results, save_dir)

        # Generate visualizations
        if visualize:
            self._generate_visualizations(save_dir)

        return results
```

#### 3.3 Evaluation Script
**File:** `scripts/evaluate.py`

```bash
# Example usage
python scripts/evaluate.py \
  --checkpoint ./outputs/2025-10-24/12-00-00/checkpoints/best_model.pt \
  --split test \
  --output_dir ./evaluation_results \
  --visualize \
  --num_visualizations 10
```

**Output structure:**
```
evaluation_results/
├── metrics.json                    # All metrics
├── metrics_summary.txt             # Human-readable summary
├── per_trajectory_metrics.csv      # Detailed per-trajectory results
├── confusion_matrix.png
├── pr_curve.png
├── roc_curve.png
├── f1_iou_plot.png
└── visualizations/
    ├── traj_0_predictions.mp4
    ├── traj_1_predictions.mp4
    └── ...
```

#### 3.4 Unit Tests
**Structure:**
```
tests/
├── test_data/
│   ├── test_base_dataset.py
│   ├── test_agibot_dataset.py
│   ├── test_splits.py
│   └── test_training_dataset.py
├── test_features/
│   ├── test_dino_extractor.py
│   ├── test_droid_extractor.py
│   └── test_feature_cache.py
├── test_models/
│   ├── test_boundary_prediction.py
│   ├── test_encoders.py
│   └── test_losses.py
├── test_training/
│   ├── test_trainer.py
│   ├── test_callbacks.py
│   └── test_label_smoothing.py
└── test_evaluation/
    ├── test_metrics.py
    ├── test_f1_iou.py
    └── test_evaluator.py
```

**Example test:**
```python
# tests/test_models/test_boundary_prediction.py
import pytest
import torch
from models.boundary_prediction import BoundaryPredictionMLP

def test_forward_pass():
    """Test model forward with dummy inputs"""
    model = BoundaryPredictionMLP(
        vocab_size=2048,
        token_embed_dim=128,
        dino_feature_dim=3072,
        prediction_window_size=10,
        hidden_dims=[512, 256, 128]
    )

    batch_size = 4
    max_token_length = 100

    tokenized_actions = torch.randint(0, 2048, (batch_size, max_token_length))
    dino_features = torch.randn(batch_size, 3072)

    output = model(tokenized_actions, None, dino_features)

    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()

def test_backward_pass():
    """Test gradient flow"""
    # ... implementation
```

---

### Phase 4: Missing Features & Polish (Week 4)

#### 4.1 Data Augmentation
**File:** `training/data_augmentation.py`

```python
class TemporalAugmentation:
    """Augmentation for temporal robotic data"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def time_jittering(self, window: Window, max_shift: int = 5):
        """Shift window boundaries by random amount"""

    def gaussian_noise(self, proprio: np.ndarray, std: float = 0.01):
        """Add Gaussian noise to proprioception"""

    def frame_dropout(self, dino_features: torch.Tensor, drop_prob: float = 0.1):
        """Randomly drop frames to simulate missing data"""

    def speed_variation(self, window: Window, speed_range: Tuple[float, float] = (0.8, 1.2)):
        """Interpolate/subsample to simulate speed variations"""
```

#### 4.2 Inference Pipeline
**File:** `scripts/inference.py`

```python
"""Production inference on new trajectories"""

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    # Load model
    model = load_checkpoint(cfg.checkpoint)

    # Load data
    trajectory = load_trajectory(cfg.input_video, cfg.input_proprio)

    # Extract features
    features = extract_features(trajectory, cfg)

    # Predict
    predictions = model.predict(features)

    # Post-process
    boundaries = postprocess_predictions(predictions, cfg.postprocessing)

    # Save results
    save_predictions(boundaries, cfg.output)

    # Visualize
    if cfg.visualize:
        create_visualization(trajectory, boundaries, cfg.output_video)
```

**Usage:**
```bash
python scripts/inference.py \
  checkpoint=./checkpoints/best_model.pt \
  input_video=./data/new_traj/video.mp4 \
  input_proprio=./data/new_traj/proprio.h5 \
  output=./predictions.json \
  visualize=true
```

#### 4.3 Reproducibility Utilities
**File:** `utils/reproducibility.py`

```python
def set_all_seeds(seed: int):
    """Set seeds for all random number generators"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_experiment_state(cfg: DictConfig, output_dir: Path):
    """Save complete experiment state for reproducibility"""
    state = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'git_commit': get_git_commit(),
        'git_diff': get_git_diff(),
        'packages': get_installed_packages(),
        'system_info': get_system_info(),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'experiment_state.json', 'w') as f:
        json.dump(state, f, indent=2)
```

#### 4.4 Documentation
**Create:**
1. `docs/architecture.md` - System architecture with diagrams
2. `docs/data_pipeline.md` - Dataset preparation walkthrough
3. `docs/training_guide.md` - Training guide with examples
4. `docs/evaluation_guide.md` - Evaluation metrics explained
5. `docs/api_reference/` - Auto-generated API docs (Sphinx)

**Update:**
1. `README.md` - Quick start guide
2. `CONTRIBUTING.md` - Development guidelines
3. `CHANGELOG.md` - Version history

---

## Migration Guide

### For Existing Users

#### Updating from Current Codebase

**Step 1: Backup**
```bash
# Create backup branch
git checkout -b backup-pre-merge
git push origin backup-pre-merge
```

**Step 2: Install new dependencies**
```bash
pip install -r requirements.txt
# New dependencies: hydra-core, scipy (for Hungarian algorithm)
```

**Step 3: Prepare data with new splits**
```bash
# This creates train/val/test splits
python scripts/prepare_data.py \
  --task_ids 327 \
  --output_dir ./preprocessed_data \
  --split_ratios 0.7 0.15 0.15
```

**Step 4: Convert old training scripts**
```bash
# Old way (hardcoded parameters)
python examples/train_boundary_model.py

# New way (Hydra config)
python scripts/train.py dataset=agibot_327
```

**Step 5: Use new evaluation**
```bash
# Old way (manual inference in notebook)
# ... manual code ...

# New way (comprehensive evaluation)
python scripts/evaluate.py \
  checkpoint=./outputs/.../checkpoints/best_model.pt \
  split=test
```

---

## Testing Strategy

### Unit Tests
- **Coverage target:** >80%
- **Framework:** pytest
- **Run:** `pytest tests/ -v --cov=.`

### Integration Tests
- End-to-end data preparation
- Full training run (1 epoch)
- Complete evaluation pipeline
- Benchmark runner

### Regression Tests
- Compare metrics with known baselines
- Ensure deterministic results with fixed seeds
- Track training curves over versions

### Performance Tests
- DINO extraction speed
- Training throughput (samples/sec)
- Inference latency

---

## Timeline

### Week 1: Foundation
- [x] Create documentation (this file)
- [ ] Create new directory structure
- [ ] Implement data splitting module
- [ ] Port feature extraction with improvements
- [ ] Create data preparation script
- [ ] Write tests for data module

### Week 2: Model & Training
- [ ] Port model architecture
- [ ] Set up Hydra configuration
- [ ] Implement trainer with callbacks
- [ ] Port WandB integration
- [ ] Create training script
- [ ] Write tests for models and training

### Week 3: Evaluation
- [ ] Port evaluation metrics
- [ ] Create evaluation pipeline
- [ ] Integrate with benchmarking
- [ ] Create evaluation script
- [ ] Generate comprehensive visualizations
- [ ] Write tests for evaluation

### Week 4: Polish
- [ ] Add data augmentation
- [ ] Create inference pipeline
- [ ] Fix code quality issues
- [ ] Write comprehensive documentation
- [ ] Create API reference
- [ ] Update README
- [ ] Final testing and validation

---

## Success Criteria

### Functionality
- ✅ Train/val/test splits working
- ✅ Model training converges
- ✅ Evaluation metrics computed correctly
- ✅ Benchmarking framework integrated
- ✅ Inference pipeline works on new data

### Quality
- ✅ >80% test coverage
- ✅ All tests passing
- ✅ Code linted (flake8, black)
- ✅ Type hints everywhere (mypy)
- ✅ Comprehensive documentation

### Performance
- ✅ DINO extraction <1 min/trajectory (4 GPUs)
- ✅ Training <30 min/100 epochs (1 A100)
- ✅ Inference <100ms/trajectory

### Reproducibility
- ✅ Fixed seeds produce identical results
- ✅ Experiment configs saved automatically
- ✅ WandB logs complete

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing code | High | High | Feature branch, comprehensive tests |
| DINO extraction too slow | Medium | Medium | Keep Ray parallelization, better caching |
| Config migration complex | Medium | Low | Conversion script, examples |
| WandB logs lost | Low | Medium | Offline mode, cloud backup |
| Test split contamination | Low | High | Save split indices, version control |
| Dependencies conflicts | Medium | Medium | Pin versions, use virtual env |

---

## Appendix

### Key Design Decisions

1. **Hydra over Tyro**: More flexible, YAML-based, better for non-coders
2. **Keep both dataset backends**: Unified under BaseSegmentationDataset
3. **Separate benchmarking**: benchmark/ for baselines, evaluation/ for neural models
4. **Centralized Ray init**: utils/ray_utils.py with proper checks
5. **70/15/15 split**: More training data, adequate test set

### References
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [WandB Best Practices](https://docs.wandb.ai/guides/track/config)
- [PyTorch Lightning Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Document Version:** 1.0
**Last Updated:** October 24, 2025
**Status:** Ready for Implementation
