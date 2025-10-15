# Segment Boundary Prediction Training

This module provides a complete training pipeline for learning to predict segment boundaries using multi-modal features (DINO visual features + Fast-DROID tokenized proprioception).

## Overview

The training pipeline consists of:

1. **Dataset Generator** (`dataset_generator.py`): Generates training samples from trajectories
2. **Model** (`models.py`): MLP architecture for boundary prediction
3. **Trainer** (`train.py`): Training loop with logging and checkpointing

## Architecture

### Input Features (per window)

- **DINO Visual Features** (3072 dims): Frozen DINOv3 features from begin, middle, and end frames
  - Begin frame: First frame of the window
  - Middle frame: Center frame of the window
  - End frame: Last frame of the window
  - Each frame → 1024 dims (DINOv3 ViT-L)

- **Fast-DROID Tokenized Proprio** (600 tokens): Tokenized action sequences
  - Actions computed as differences between consecutive proprio states
  - Tokenized using Fast-DROID tokenizer
  - Embedded via learned embedding layer (vocab_size=32000, embed_dim=128)
  - Encoded via 1D CNN → 128 dims

- **Statistical Proprio Features** (84 dims): Statistical features from the window
  - Mean, std, min, max, 25th percentile, 75th percentile
  - Computed for all 14 proprio dimensions (left + right end effector poses)

### Model Architecture

```
Input:
  - tokenized_actions: (batch, 600) token IDs
  - proprio_stats: (batch, 84) float features
  - dino_features: (batch, 3072) frozen features

Processing:
  1. Token Embedding: (batch, 600) → (batch, 600, 128)
  2. 1D Conv + Pooling: (batch, 600, 128) → (batch, 128)
  3. Concatenate: [token_features, proprio_stats, dino_features]
                  = (batch, 128 + 84 + 3072) = (batch, 3284)
  4. MLP: (batch, 3284) → (batch, 512) → (batch, 256) → (batch, 128) → (batch, 2)

Output:
  - Binary logits: (batch, 2) - [non-boundary, boundary]
```

### Labels

- **Binary Classification**: Is the center frame of the window a segment boundary?
  - 0: Non-boundary frame
  - 1: Boundary frame (within `boundary_tolerance` of a ground truth boundary)

## Usage

### Quick Start

```python
from pathlib import Path
from benchmark.data import AgiBotWorldDataset
from training import SegmentBoundaryDataset, create_dataloaders, create_model, Trainer, FocalLoss

# Load dataset
dataset = AgiBotWorldDataset(
    data_root=Path("./agibot_full_data/extracted"),
    task_ids=[327],
    max_trajectories=50
)

# Generate training data
train_dataset = SegmentBoundaryDataset(
    dataset=dataset,
    window_size=15,
    stride=1,
    max_token_length=600,
    boundary_tolerance=2
)

# Create data loader
train_loader, _ = create_dataloaders(train_dataset, batch_size=32)

# Create model
model = create_model(device='cuda')

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    criterion=FocalLoss(alpha=0.25, gamma=2.0)
)

# Train
trainer.train(num_epochs=30)
```

### Command Line Training

```bash
python training/train.py \
  --data_root ./agibot_full_data/extracted \
  --task_ids 327 \
  --max_trajectories 50 \
  --window_size 15 \
  --stride 1 \
  --boundary_tolerance 2 \
  --batch_size 32 \
  --num_epochs 30 \
  --lr 1e-3 \
  --use_focal_loss \
  --output_dir ./training_output
```

### Example Script

```bash
python examples/train_boundary_model.py
```

## Dataset Generation

The `SegmentBoundaryDataset` class handles:

1. **Window Creation**: Sliding windows over trajectories with configurable size and stride
2. **Feature Extraction**:
   - Loads proprioception from H5 files
   - Extracts DINO features from video frames
   - Tokenizes actions using Fast-DROID
   - Computes statistical features
3. **Label Generation**: Marks frames near boundaries as positive samples
4. **Caching**: Can save/load preprocessed datasets for faster iteration

### Parameters

- `window_size` (default=15): Number of frames in each window
- `stride` (default=1): Stride for sliding window
- `max_token_length` (default=600): Fixed token sequence length
- `boundary_tolerance` (default=2): Frames within ±N of a boundary are labeled as positive

## Model Training

### Loss Functions

Two loss functions are provided to handle class imbalance:

1. **Focal Loss** (recommended): Focuses on hard examples
   ```python
   criterion = FocalLoss(alpha=0.25, gamma=2.0)
   ```

2. **Weighted BCE Loss**: Weights positive class
   ```python
   criterion = WeightedBCELoss(pos_weight=5.0)
   ```

### Optimization

- Optimizer: AdamW with weight decay
- Scheduler: ReduceLROnPlateau (reduces LR when F1 plateaus)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### Monitoring

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir ./training_output/logs
```

Metrics tracked:
- Training: loss, accuracy, precision, recall, F1 (per epoch)
- Validation: loss, accuracy, precision, recall, F1, ROC-AUC (per epoch)
- Learning rate

## Output

Training produces:

1. **Checkpoints** (`./training_output/checkpoints/`):
   - `best_model.pt`: Best model by validation F1
   - `final_model.pt`: Final model after all epochs
   - `checkpoint_epoch_N.pt`: Periodic checkpoints

2. **Logs** (`./training_output/logs/`):
   - TensorBoard event files

3. **Config** (`./training_output/config.json`):
   - All hyperparameters and settings

## File Structure

```
training/
├── __init__.py              # Module exports
├── dataset_generator.py     # Dataset and data loading
├── models.py                # Model architectures and losses
├── train.py                 # Training script and trainer
└── README.md                # This file

examples/
└── train_boundary_model.py  # Example training script
```

## Requirements

- PyTorch >= 1.10
- transformers (for Fast-DROID tokenizer)
- opencv-python (for video loading)
- h5py (for proprioception data)
- tensorboard (for logging)
- scikit-learn (for metrics)
- tqdm (for progress bars)

## Notes

- **DINO features are frozen**: The DINOv3 model is used for feature extraction only, not fine-tuned
- **Class imbalance**: Boundary frames are rare (~5-10% of frames), use Focal Loss to handle this
- **GPU memory**: DINO feature extraction requires ~8GB GPU memory. Reduce batch size if OOM.
- **Data loading**: Set `num_workers=0` if you encounter multiprocessing issues

## Future Improvements

1. Add validation split
2. Implement temporal smoothing for predictions
3. Add data augmentation (temporal jittering, noise)
4. Experiment with transformer architectures
5. Multi-task learning (predict segment labels + boundaries)
