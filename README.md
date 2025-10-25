# Task Segmentation for Robot Manipulation

**A unified framework for multi-modal task segmentation combining visual (DINOv3) and proprioceptive (Fast-DROID) features.**

This codebase merges the best components from two task segmentation projects, providing:
- Advanced multi-modal neural architecture
- Ray-parallelized feature extraction
- Comprehensive evaluation metrics (F1@IoU, temporal, confidence)
- Proper train/val/test splitting
- Production-ready code with extensive documentation

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Citation](#citation)

---

## ✨ Features

### Multi-Modal Learning
- **Visual Features**: DINOv3-ViT-L16 vision transformer (1024-dim per frame)
- **Proprioceptive Features**: Fast-DROID tokenization of joint trajectories
- **Fusion**: Transformer-based proprioception encoder + per-frame DINO encoder

### Efficient Processing
- **Ray Parallelization**: Multi-GPU feature extraction with 4 workers
- **Feature Caching**: MD5-based caching for preprocessed features
- **Batch Processing**: Efficient batch DINO extraction (256 frames/batch)

### Robust Training
- **Callback System**: WandB logging, checkpointing, early stopping, TensorBoard
- **Loss Functions**: Focal Loss, Soft Focal Loss, BCE variants, Weighted BCE
- **Label Smoothing**: Gaussian, Triangular, Uniform, Exponential smoothers
- **Trainer Classes**: Modular `Trainer` and `BoundaryTrainer` with extensible hooks
- **Feature Caching**: Automatic caching of DINO features for instant re-runs
- **Scripts**: Complete CLI scripts for data prep, training, evaluation, inference

### Comprehensive Evaluation
- **Frame-level**: Accuracy, precision, recall, F1
- **Segment-level**: F1@IoU with Hungarian algorithm matching
- **Temporal**: Boundary detection error, tolerance-based metrics
- **Confidence**: AUC-ROC, AUC-PR, optimal threshold finding

### Production-Ready
- **Type Hints**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings with examples
- **Testing**: Unit tests for all modules (target: >80% coverage)
- **Reproducibility**: Seed management, experiment state saving
- **Modular**: Clean separation of concerns, easy to extend

---

## 📦 Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- 32GB+ RAM recommended
- 4+ GPUs recommended for parallel feature extraction

### Install from Source

```bash
# Clone repository
git clone https://github.com/your-org/task_seg.git
cd task_seg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies:
```
torch>=2.0.0
transformers>=4.30.0
ray>=2.5.0
av  # PyAV for video processing
numpy
Pillow
scikit-learn
scipy  # For Hungarian algorithm
h5py  # For proprioception data
tqdm
```

Optional dependencies:
```
wandb  # For experiment tracking
tensorboard  # For logging
pytest  # For testing
hydra-core  # For configuration (coming soon)
```

---

## 🚀 Quick Start

### Complete Workflow (Using Scripts)

The easiest way to get started is using our command-line scripts:

```bash
# 1. Prepare data splits
python scripts/prepare_data.py \
    --task-ids 327 \
    --output-dir ./splits \
    --extract-dino \
    --cache-dir ./cache

# 2. Train model
python scripts/train.py \
    --train-split ./splits/task_327_train.json \
    --val-split ./splits/task_327_val.json \
    --wandb-project boundary-prediction \
    --num-epochs 100 \
    --early-stopping

# 3. Evaluate on test set
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --test-split ./splits/task_327_test.json \
    --output-dir ./evaluation_results

# 4. Run inference on new data
python scripts/inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --video-path ./data/new_episode.mp4 \
    --proprio-path ./data/new_episode_proprio.npy \
    --output results.json \
    --visualize
```

### Programmatic Usage

For more control, use the Python API:

#### 1. Data Preparation

```python
from data.splits import DatasetSplitter, SplitConfig

# Create splits
config = SplitConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    stratify_by_task=True
)

splitter = DatasetSplitter(config)
train_split, val_split, test_split = splitter.split_trajectories(trajectories)

# Save splits
splitter.save_split(train_split, './splits/train.json')
splitter.save_split(val_split, './splits/val.json')
splitter.save_split(test_split, './splits/test.json')
```

#### 2. Training

```python
from training import (
    BoundaryDataset, TrainingDatasetConfig,
    BoundaryTrainer, WandBCallback, CheckpointCallback
)
from models import create_model, ModelConfig
import torch.optim as optim

# Create dataset with caching
config = TrainingDatasetConfig(
    window_size=180,
    stride=10,
    vocab_size=2048,
    cache_path='./cache'
)
train_dataset = BoundaryDataset(train_trajectories, config)

# Create model
model = create_model(ModelConfig(vocab_size=2048))
optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# Train with callbacks
callbacks = [
    WandBCallback(project='boundary-prediction'),
    CheckpointCallback(save_dir='./checkpoints')
]

trainer = BoundaryTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=callbacks
)

history = trainer.fit(train_loader, val_loader, num_epochs=100)
```

#### 3. Evaluation

```python
from evaluation import ModelEvaluator, EvaluationConfig

# Configure evaluation
eval_config = EvaluationConfig(
    iou_thresholds=[0.3, 0.5, 0.7],
    save_visualizations=True
)

# Run evaluation
evaluator = ModelEvaluator(model, eval_config)
results = evaluator.evaluate(
    test_loader,
    save_dir='./evaluation_results',
    visualize=True
)

print(f"F1@IoU(0.5): {results['f1_iou_0.5']['f1']:.3f}")
print(f"AUC-ROC: {results['auc_roc']:.3f}")
```

---

## 🏗️ Architecture

### Model Architecture

```
┌─────────────────────┐      ┌─────────────────────┐
│  Tokenized Actions  │      │   DINO Features     │
│   (batch, seq_len)  │      │   (batch, 3×1024)   │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐
│  TinyFASTEncoder    │      │ DinoFeatureEncoder  │
│  - Embedding        │      │ - L2 Normalize      │
│  - Transformer      │      │ - Noise (train)     │
│  - Mean Pooling     │      │ - Per-frame Linear  │
│  → (batch, 128)     │      │ → (batch, 96)       │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │   Concatenate   │
              │  (batch, 224)   │
              └────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │   MLP Stack     │
              │  512 → 256 → 128│
              │  + BatchNorm    │
              │  + ReLU         │
              │  + Dropout      │
              └────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │   Output Layer  │
              │  → (batch, 10)  │
              └─────────────────┘
                   Predictions
```

### Data Pipeline

```
1. Load Trajectories
   ├─ HuggingFace Dataset
   └─ Local Files (H5 + MP4)
         │
         ▼
2. Create Splits
   ├─ Train (70%)
   ├─ Val (15%)
   └─ Test (15%)
         │
         ▼
3. Feature Extraction
   ├─ DINO (Ray Parallel)
   │  └─ 4 GPU workers
   └─ DROID Tokenization
         │
         ▼
4. Create Windows
   ├─ Sliding Window (size=180, stride=10)
   ├─ Label Smoothing (Gaussian)
   └─ Cache Features
         │
         ▼
5. Training
   ├─ DataLoader (batch=512)
   ├─ Model Forward
   ├─ Loss Computation
   └─ Optimization
         │
         ▼
6. Evaluation
   ├─ Frame Metrics
   ├─ F1@IoU
   ├─ Temporal Metrics
   └─ Visualizations
```

---

## 📚 Usage

### Training with Custom Configuration

```python
from models import ModelConfig, create_model
from training import Trainer, TrainingConfig
from training.callbacks import WandBCallback, CheckpointCallback
from data import create_dataloaders

# Training configuration
train_config = TrainingConfig(
    num_epochs=100,
    batch_size=512,
    learning_rate=3e-3,
    weight_decay=1e-4,
    scheduler='cosine',
    early_stopping_patience=10
)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    dataset_path='./preprocessed_data/task_327',
    batch_size=train_config.batch_size,
    num_workers=4
)

# Create model
model_config = ModelConfig()
model = create_model(model_config)

# Callbacks
callbacks = [
    WandBCallback(project='task-segmentation', name='experiment-1'),
    CheckpointCallback(save_dir='./checkpoints', monitor='val_f1'),
]

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=train_config,
    callbacks=callbacks
)

trainer.train()
```

### Inference on New Data

```python
from models import BoundaryPredictionMLP
from features import DinoFeatureExtractor, DroidTokenEmbedder
from model.postprocessing import non_maximum_suppression

# Load trained model
model = torch.load('checkpoints/best_model.pt')
model.eval()

# Extract features
dino_features = dino_extractor.extract_batch(video_path, frame_indices)
droid_tokens = droid_embedder.embed_frames(proprio, frames, window_size)

# Predict
with torch.no_grad():
    logits = model(tokens, None, dino)
    probs = torch.sigmoid(logits)

# Post-process
boundaries = non_maximum_suppression(
    probs.cpu().numpy(),
    threshold=0.5,
    min_distance=10
)

print(f"Detected {len(boundaries)} boundaries at frames: {boundaries}")
```

---

## ⚙️ Configuration

### Model Configuration

```python
from models import ModelConfig

config = ModelConfig(
    # Vocabulary
    vocab_size=2048,              # Tokenizer vocabulary size
    token_embed_dim=128,          # Token embedding dimension

    # Features
    dino_feature_dim=3072,        # 3 frames × 1024

    # Architecture
    hidden_dims=[512, 256, 128],  # MLP layers
    prediction_window_size=10,    # Output window size

    # Dropout
    transformer_dropout=0.3,      # Transformer dropout
    embedding_dropout=0.1,        # Embedding dropout
    dino_encoder_dropout=0.3,     # DINO encoder dropout
    mlp_dropout=0.3,              # MLP dropout

    # Augmentation
    feature_noise_std=0.05,       # Std for noise injection

    # Modalities
    use_proprio=True,             # Use proprioception
    use_dino=True,                # Use visual features
)
```

### Split Configuration

```python
from data.splits import SplitConfig

config = SplitConfig(
    train_ratio=0.7,              # 70% training
    val_ratio=0.15,               # 15% validation
    test_ratio=0.15,              # 15% testing
    seed=42,                      # Random seed
    stratify_by_task=True,        # Stratify by task ID
    min_trajectories_per_task=3  # Minimum per split
)
```

---

## 📖 Documentation

### Full Documentation

- **[Merge Plan](docs/MERGE_PLAN.md)**: Comprehensive plan for codebase merge
- **[Architecture Guide](docs/architecture.md)**: System architecture (coming soon)
- **[Data Pipeline](docs/data_pipeline.md)**: Dataset preparation guide (coming soon)
- **[Training Guide](docs/training_guide.md)**: Training walkthrough (coming soon)
- **[Evaluation Guide](docs/evaluation_guide.md)**: Metrics explanation (coming soon)
- **[API Reference](docs/api_reference/)**: Auto-generated docs (coming soon)

### Module Documentation

All modules have comprehensive docstrings with examples:

```python
from models import BoundaryPredictionMLP
help(BoundaryPredictionMLP)  # Full documentation
```

---

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_models/test_boundary_prediction.py -v
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black .
flake8 .
mypy .
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{task_segmentation_2025,
  title={Task Segmentation for Robot Manipulation},
  author={Your Team},
  year={2025},
  url={https://github.com/your-org/task_seg}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **DINOv3**: Meta AI's vision transformer
- **Fast-DROID**: Physical Intelligence's proprioceptive tokenizer
- **AgiBotWorld**: Dataset and annotations
- **Ray**: Distributed computing framework

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your-email@example.com

---

**Status**: 🚧 Active Development

**Latest Version**: 0.1.0 (October 2025)
