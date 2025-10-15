# Task Segmentation Framework

A comprehensive framework for task segmentation in robotics trajectories, featuring benchmarking tools and trainable models.

## Project Structure

```
task_seg/
├── data/                     # Data loading and dataset management
│   ├── base_dataset.py      # Abstract dataset interface
│   ├── agibot_dataset.py    # AgiBotWorld dataset loader
│   └── agibot_hf_direct.py  # HuggingFace direct loading
│
├── embedding/                # Feature extraction and embeddings
│   └── fast_droid_embeddings.py  # Fast-DROID + DINOv3 embeddings
│
├── benchmark/                # Benchmarking framework
│   ├── methods/             # Segmentation methods
│   │   ├── base_method.py   # Base method interface
│   │   └── clustering_methods.py  # Clustering-based methods
│   ├── metrics/             # Evaluation metrics
│   │   └── segmentation_metrics.py
│   └── benchmark_runner.py  # Main benchmark runner
│
├── model/                    # Model architectures
│   └── boundary_prediction.py  # MLP for boundary prediction
│
├── training/                 # Training utilities
│   ├── dataset_generator.py # Training dataset generation
│   ├── train.py            # Training script
│   └── README.md           # Training documentation
│
└── examples/                 # Example scripts
    ├── train_boundary_model.py
    └── run_with_real_agibot.py
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd task_seg

# Install dependencies
pip install -r requirements.txt

# For DINO visual features
pip install transformers opencv-python

# For Fast-DROID tokenization
# (Fast-DROID tokenizer will be loaded from HuggingFace)
```

## Quick Start

### 1. Loading Data

```python
from data import AgiBotWorldDataset
from pathlib import Path

# Load AgiBotWorld dataset
dataset = AgiBotWorldDataset(
    data_root=Path("./agibot_full_data/extracted"),
    task_ids=[327],  # Optional: filter by task
    max_trajectories=50
)

# Print statistics
dataset.print_statistics()

# Access trajectories
for traj in dataset.trajectories:
    print(f"Trajectory {traj.trajectory_id}: {traj.num_segments} segments")
```

### 2. Generating Embeddings

```python
from embedding import create_embedding_function

# Create embedding function
embedding_fn = create_embedding_function(
    window_size=15,
    stride=1,
    use_dino=True,
    max_token_length=600
)

# Generate embeddings for a trajectory
trajectory = dataset[0]
embeddings = embedding_fn(trajectory)  # Shape: (n_windows, embedding_dim)
```

### 3. Running Benchmarks

```python
from benchmark import BenchmarkRunner
from benchmark.methods import (
    SpectralClusteringMethod,
    ChangePointDetectionMethod
)

# Define methods to benchmark
methods = [
    SpectralClusteringMethod(n_clusters=5, temporal_weight=0.3),
    ChangePointDetectionMethod(penalty=3.0, max_segments=5)
]

# Create benchmark runner
runner = BenchmarkRunner(
    dataset=dataset,
    methods=methods,
    embedding_fn=embedding_fn,
    output_dir=Path("./benchmark_results")
)

# Run benchmark
results = runner.run_all_methods()

# Get best method
best_method = runner.get_best_method(metric='macro_f1')
print(f"Best method: {best_method}")
```

### 4. Training Boundary Prediction Model

```python
from data import AgiBotWorldDataset
from training import SegmentBoundaryDataset, create_dataloaders, Trainer
from model import create_model, FocalLoss
import torch

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
    boundary_tolerance=2
)

# Create data loader
train_loader, _ = create_dataloaders(train_dataset, batch_size=32)

# Create model
model = create_model(device='cuda')

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    criterion=FocalLoss(alpha=0.25, gamma=2.0)
)
trainer.train(num_epochs=30)
```

Or use the example script:

```bash
python examples/train_boundary_model.py
```

## Command Line Interface

### Training Script

```bash
python training/train.py \
  --data_root ./agibot_full_data/extracted \
  --task_ids 327 \
  --max_trajectories 50 \
  --batch_size 32 \
  --num_epochs 30 \
  --use_focal_loss \
  --output_dir ./training_output
```

View training progress:
```bash
tensorboard --logdir ./training_output/logs
```

## Modules

### Data (`data/`)
- **AgiBotWorldDataset**: Loads and manages AgiBotWorld trajectories with segmentation annotations
- **TrajectoryData**: Represents a single trajectory with segments
- **SegmentAnnotation**: Represents a segment with start/end frames and action text

### Embedding (`embedding/`)
- **FastDROIDEmbedder**: Combines Fast-DROID tokenization and DINOv3 visual features
- Supports sliding windows over trajectories
- Handles proprioception from H5 files and videos

### Benchmark (`benchmark/`)
- **BenchmarkRunner**: Unified interface for evaluating segmentation methods
- **Methods**: Various segmentation approaches (clustering, change point detection, etc.)
- **Metrics**: Comprehensive evaluation (accuracy, F1, boundary detection, etc.)

### Model (`model/`)
- **BoundaryPredictionMLP**: Neural network for predicting segment boundaries
- Combines frozen DINO features, tokenized actions, and statistical features
- **FocalLoss**: Loss function for handling class imbalance

### Training (`training/`)
- **SegmentBoundaryDataset**: PyTorch dataset for boundary prediction
- **Trainer**: Training loop with checkpointing and logging
- See `training/README.md` for detailed documentation

## Features

- **Multi-modal Features**: Combines visual (DINO), proprioceptive (Fast-DROID), and statistical features
- **Flexible Benchmarking**: Easy to add new segmentation methods
- **Comprehensive Metrics**: Frame-level, boundary detection, and clustering metrics
- **Training Framework**: Complete pipeline for training boundary prediction models
- **Visualization**: TensorBoard integration for training monitoring

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- transformers (for Fast-DROID and DINO)
- opencv-python (for video processing)
- h5py (for proprioception data)
- scikit-learn (for metrics and clustering)
- ruptures (for change point detection)
- tqdm (for progress bars)
- tensorboard (for training visualization)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{task_segmentation_framework,
  title = {Task Segmentation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/task_seg}
}
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
