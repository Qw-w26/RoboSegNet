# Examples

## Main Example

### `run_with_real_agibot.py` ⭐

**This is the primary example showing how to use the benchmark framework.**

Features:
- Loads real AgiBotWorld trajectories with ground truth segmentation
- Uses Fast-DROID tokenizer for action embeddings
- Optional DINOv3 visual features (if videos available)
- Optional forward kinematics (if URDF available)
- Benchmarks multiple segmentation methods
- Comprehensive evaluation metrics

**Usage:**
```bash
python examples/run_with_real_agibot.py
```

**What it does:**
1. Downloads 5 real trajectories from AgiBotWorld (annotations only, ~KB)
2. Initializes Fast-DROID + DINOv3 embedder
3. Runs 3 segmentation methods (Spectral, ChangePoint, GMM)
4. Evaluates against ground truth labels
5. Saves results to `./fastdroid_results/`

**Output:**
- Frame-level accuracy, precision, recall, F1
- Boundary detection metrics
- Temporal coherence metrics
- Method comparison table
- Per-trajectory predictions

## Customization

### Embedding Configuration

```python
from benchmark import create_embedding_function

# Full pipeline with all features
embedding_fn = create_embedding_function(
    window_size=15,           # Sliding window size
    stride=1,                 # Window stride
    use_cartesian=True,       # Use FK to convert to Cartesian
    use_dino=True,            # Extract visual features from videos
    urdf_path="/path/to/robot.urdf"  # Robot model for FK
)
```

### Adding Custom Methods

```python
from benchmark.methods import WindowBasedSegmentationMethod

class MyMethod(WindowBasedSegmentationMethod):
    def fit(self, trajectories, embeddings=None):
        # Train your method (optional)
        pass

    def predict(self, trajectory_data, embeddings=None):
        # Return frame-level labels
        window_labels = self.your_segmentation(embeddings)
        return self.window_labels_to_frame_labels(
            window_labels,
            trajectory_data.total_frames
        )
```

### Using Local Datasets

For datasets with proprioceptive and visual data:

```python
from benchmark.data import BaseSegmentationDataset

class MyDataset(BaseSegmentationDataset):
    def _load_trajectories(self):
        # Load your trajectories with segments
        return trajectories
```

## Requirements

```bash
# Core dependencies
pip install numpy scipy scikit-learn ruptures h5py tqdm

# For Fast-DROID embeddings
pip install transformers torch

# For DINOv3 visual features
pip install opencv-python Pillow

# For forward kinematics (optional)
pip install yourdfpy

# For HuggingFace dataset streaming
pip install datasets huggingface_hub
```

## Other Files

- `run_agibot_benchmark.py` - Advanced CLI interface (deprecated, use for reference)

## Tips

1. **Start simple**: Run the main example first to understand the pipeline
2. **Add real data**: Use local datasets with H5 files and videos for better embeddings
3. **Customize**: Modify embedding parameters, segmentation methods, or evaluation metrics
4. **Train models**: Use the ground truth labels to train your own segmentation models
5. **Scale up**: Process more trajectories, add more methods, run hyperparameter sweeps

## Troubleshooting

**"No proprioceptive data":**
- AgiBotWorld streaming only includes annotations
- Download full dataset or use local robot data
- Embeddings will fall back gracefully

**"CUDA not available":**
- DINOv3 works on CPU but is slower
- Set `use_dino=False` to disable visual features

**"URDF not found":**
- Forward kinematics requires a robot model
- Set `use_cartesian=False` to use joint space directly
- Provide correct path with `urdf_path` parameter

**"HuggingFace errors":**
- Check internet connection
- Some datasets may require authentication
- Use `hf_token` parameter if needed
