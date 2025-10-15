# Task Segmentation Benchmark

A modular framework for benchmarking task segmentation methods on robotics datasets with ground truth action segmentations.

## Overview

This benchmark provides:
- **Modular data loaders** for different datasets (currently supports AgiBotWorld)
- **Standard evaluation metrics** for task segmentation quality
- **Unified interface** for segmentation methods
- **Benchmark runner** for systematic comparison of methods
- **Pre-implemented methods**: Spectral Clustering, Change Point Detection, GMM, Hierarchical Clustering, K-Means

## Architecture

```
benchmark/
├── data/                    # Dataset loaders
│   ├── base_dataset.py     # Abstract base classes
│   ├── agibot_dataset.py   # AgiBotWorld dataset loader
│   └── __init__.py
├── methods/                 # Segmentation methods
│   ├── base_method.py      # Abstract base classes
│   ├── clustering_methods.py  # Implemented methods
│   └── __init__.py
├── metrics/                 # Evaluation metrics
│   ├── segmentation_metrics.py
│   └── __init__.py
├── benchmark_runner.py     # Main benchmark runner
└── __init__.py
```

## Installation

```bash
# Install required packages
pip install numpy scipy scikit-learn ruptures h5py tqdm

# For AgiBotWorld dataset, install git-lfs
sudo apt-get install git-lfs
git lfs install
```

## Quick Start

**Fastest way to get started:**

```bash
# Install required packages
pip install transformers torch numpy scipy scikit-learn ruptures h5py tqdm datasets

# Run the main example (uses real Fast-DROID embeddings)
python examples/run_with_real_agibot.py
```

This will:
- Download 5 real trajectories with ground truth from AgiBotWorld (~KB, no videos/H5)
- Initialize Fast-DROID tokenizer + DINOv3 (if available)
- Run 3 segmentation methods: Spectral, ChangePoint, GMM
- Evaluate against ground truth labels
- Display comprehensive metrics and comparison
- Save results to `./fastdroid_results/`

### What You Get

The example demonstrates:
- ✅ **Real AgiBotWorld data** with 6 segments per trajectory
- ✅ **Fast-DROID embeddings** - tokenized action sequences
- ✅ **Forward kinematics** - Cartesian space conversion (if URDF available)
- ✅ **DINOv3 features** - visual embeddings (if videos available)
- ✅ **Multiple methods** - Spectral, ChangePoint, GMM
- ✅ **Complete metrics** - Frame accuracy, boundary detection, temporal coherence
- ✅ **Ground truth evaluation** - Real segment labels for benchmarking

### Example Output

```
Dataset Statistics:
  Number of trajectories: 5
  Total segments: 30
  Avg segments per trajectory: 6.00 ± 0.00
  Total frames: 5852

Method Comparison:
  Method                    Acc      F1     ARI    B-F1
  --------------------------------------------------------
  Gaussian Mixture Model    0.15   0.14   0.93   1.00
  Spectral Clustering       0.13   0.13   0.99   1.00
  Change Point Detection    0.20   0.10   0.93   1.00

Best Method: Gaussian Mixture Model
```

## Usage Examples

### Complete Example with Fast-DROID

```python
from benchmark import (
    AgiBotHFDirectDataset,
    SpectralClusteringMethod,
    ChangePointDetectionMethod,
    GMMMethod,
    BenchmarkRunner,
    create_embedding_function
)

# Load dataset
dataset = AgiBotHFDirectDataset(
    task_ids=[327],
    max_trajectories=5
)

# Create Fast-DROID + DINOv3 embedding function
embedding_fn = create_embedding_function(
    window_size=15,
    stride=1,
    use_cartesian=True,      # Use forward kinematics
    use_dino=True,           # Use visual features (if available)
    urdf_path="/path/to/robot.urdf"
)

# Define segmentation methods
methods = [
    SpectralClusteringMethod(n_clusters=6, temporal_weight=0.3),
    ChangePointDetectionMethod(max_segments=10),
    GMMMethod(n_components=6)
]

# Run benchmark
runner = BenchmarkRunner(
    dataset=dataset,
    methods=methods,
    embedding_fn=embedding_fn,  # Use Fast-DROID embeddings!
    output_dir="./results"
)
results = runner.run_all_methods()

# Get best method
best = runner.get_best_method(metric='macro_f1')
print(f"Best method: {best}")
```

### Adding Custom Datasets

Create a new dataset loader by inheriting from `BaseSegmentationDataset`:

```python
from benchmark.data import BaseSegmentationDataset, TrajectoryData, SegmentAnnotation

class MyCustomDataset(BaseSegmentationDataset):
    def _load_trajectories(self) -> List[TrajectoryData]:
        # Implement your data loading logic
        trajectories = []

        # For each trajectory in your dataset:
        segments = [
            SegmentAnnotation(
                start_frame=0,
                end_frame=100,
                action_text="Pick object",
                skill="pick",
                segment_id=0
            ),
            # ... more segments
        ]

        traj = TrajectoryData(
            trajectory_id="traj_001",
            task_name="Task name",
            segments=segments,
            proprioception_path=Path("path/to/data.h5"),
            video_paths={'camera1': Path("path/to/video.mp4")}
        )
        trajectories.append(traj)

        return trajectories
```

### Adding Custom Methods

Implement your segmentation method by inheriting from `BaseSegmentationMethod`:

```python
from benchmark.methods import WindowBasedSegmentationMethod
import numpy as np

class MySegmentationMethod(WindowBasedSegmentationMethod):
    def __init__(self, my_param: float = 1.0, **kwargs):
        super().__init__(
            name="My Method",
            window_size=15,
            stride=1,
            my_param=my_param,
            **kwargs
        )
        self.my_param = my_param

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        # Optional: train your method
        pass

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        # Implement your segmentation logic
        # Return frame-level segment labels
        num_frames = trajectory_data.total_frames

        # Your segmentation algorithm here...
        frame_labels = np.zeros(num_frames, dtype=int)

        return frame_labels
```

## Evaluation Metrics

The benchmark computes comprehensive metrics:

### Frame-Level Metrics
- **Accuracy**: Overall frame classification accuracy
- **Precision/Recall/F1**: Macro and weighted averages
- **Adjusted Rand Index (ARI)**: Clustering quality
- **Normalized Mutual Information (NMI)**: Information overlap

### Boundary Detection Metrics
- **Boundary Precision/Recall/F1**: How well segment boundaries are detected
- **Mean Boundary Error**: Average distance to nearest ground truth boundary (frames)

### Temporal Metrics
- **Over-segmentation Score**: How much predictions split ground truth segments
- **Under-segmentation Score**: How much predictions merge ground truth segments

## Dataset Format

### AgiBotWorld Dataset Structure

```
AgiBotWorld-Alpha/
├── task_info/
│   └── task_[id].json          # Contains segmentation annotations
├── observations/
│   └── [task_id]/
│       └── [episode_id]/
│           └── *.mp4           # Video observations
├── proprio_stats/
│   └── [task_id]/
│       └── [episode_id].h5     # Proprioceptive data
└── parameters/
    └── camera_params.json
```

### JSON Annotation Format

```json
[
  {
    "episode_id": 649078,
    "task_id": 327,
    "task_name": "Picking items in Supermarket",
    "init_scene_text": "The robot is in front of the fruit shelf...",
    "lable_info": {
      "action_config": [
        {
          "start_frame": 0,
          "end_frame": 435,
          "action_text": "Pick up onion from the shelf.",
          "skill": "Pick"
        },
        {
          "start_frame": 435,
          "end_frame": 619,
          "action_text": "Place onion into the plastic bag.",
          "skill": "Place"
        }
      ]
    }
  }
]
```

## Implemented Methods

### 1. Spectral Clustering
- Uses spectral clustering with temporal constraints
- Balances spatial (embedding) and temporal similarity
- Parameters: `n_clusters`, `temporal_weight`, `gamma`

### 2. Change Point Detection
- Detects change points using kernel-based detection (ruptures library)
- Automatically determines segment boundaries
- Parameters: `penalty`, `min_size`, `max_segments`

### 3. Gaussian Mixture Model (GMM)
- Soft clustering using Gaussian mixture
- Parameters: `n_components`, `covariance_type`

### 4. Hierarchical Clustering
- Agglomerative hierarchical clustering
- Parameters: `n_clusters`, `linkage_method`

### 5. K-Means
- Standard K-Means clustering on embeddings
- Parameters: `n_clusters`

## Embeddings

Methods require embeddings computed from trajectory data. The framework supports custom embedding functions:

```python
def my_embedding_fn(trajectory_data):
    """Compute embeddings from trajectory data."""
    # Load proprioceptive data
    proprio = trajectory_data.load_proprioception()

    # Your embedding logic (e.g., Fast-DROID + DINOv3)
    embeddings = compute_embeddings(proprio)

    return embeddings

runner = BenchmarkRunner(
    dataset=dataset,
    methods=methods,
    embedding_fn=my_embedding_fn
)
```

## Training Models with the Benchmark

The modular design supports training segmentation models:

1. **Data Loading**: Use dataset loaders to get trajectories with ground truth
2. **Feature Extraction**: Extract embeddings/features for training
3. **Model Training**: Train your segmentation model
4. **Evaluation**: Use benchmark metrics to evaluate

Example workflow:

```python
# Load training data
train_dataset = AgiBotWorldDataset(data_root="...", split='train')

# Extract features for training
train_features = []
train_labels = []
for traj in train_dataset:
    features = extract_features(traj)
    labels = traj.get_segment_labels()
    train_features.append(features)
    train_labels.append(labels)

# Train your model
model = MySegmentationModel()
model.fit(train_features, train_labels)

# Evaluate
test_dataset = AgiBotWorldDataset(data_root="...", split='test')
runner = BenchmarkRunner(dataset=test_dataset, methods=[model])
results = runner.run_all_methods()
```

## Future Extensions

Potential additions:
- More datasets (DROID, RoboNet, etc.)
- Deep learning methods (Temporal CNNs, Transformers)
- Online/streaming segmentation
- Multi-modal fusion methods
- Hierarchical segmentation evaluation

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{task_segmentation_benchmark,
  title={Task Segmentation Benchmark for Robotics},
  year={2025},
  url={https://github.com/...}
}
```

## License

MIT License
