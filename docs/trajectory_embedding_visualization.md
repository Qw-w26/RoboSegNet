# Trajectory Embedding Visualization

This document describes the trajectory embedding visualization feature that logs embedding plots to W&B during training.

## Overview

This visualization shows how trajectory embeddings evolve **temporally** within individual trajectories. Unlike per-sample visualizations, this plots **windows** from complete trajectories, colored by their temporal position (timestep/window index).

Based on `plot_fast_droid_embeddings.py` - adapted for integration with the training pipeline.

## What It Shows

The visualization:
1. Takes several complete trajectories from the dataset
2. Extracts embeddings for all windows within each trajectory
3. Reduces to 2D using PCA (or t-SNE)
4. Colors points by **window index** (temporal position in trajectory)
5. Shows trajectory progression through embedding space

**Key insight**: You can see how the embedding space is traversed as a trajectory unfolds over time.

## Embedding Structure

The visualized embeddings concatenate:
```python
embedding = [
    proprio_tokens (window_size,),  # Fast-DROID token IDs
    dino_features (3072,)            # 3 frames × 1024 DINO features
]
```

These are **pre-model embeddings** (inputs to the model, not learned representations).

## Usage

### Automatic (Default)

Embedding plots are automatically logged to W&B when training with WandB enabled:

```bash
python scripts/train.py --config configs/experiments/task_357.yaml wandb.enabled=true
```

Plots are generated:
- **On first epoch** - baseline visualization
- **When best model is saved** - on validation loss improvement
- **At end of training** - final state

### Configuration

Edit `scripts/train.py` to customize:

```python
embedding_vis_callback = EmbeddingVisualizationCallback(
    method='pca',                  # 'pca' or 'tsne'
    num_trajectories=3,            # Number of trajectories per split
    max_windows_per_traj=None,     # Limit windows (None = all)
    log_on_save=True,              # Log when best model saved
    log_on_first=True,             # Log on first epoch
    log_on_last=True               # Log at end of training
)
```

### Disable

To disable, remove or comment out the callback in `scripts/train.py`:

```python
# embedding_vis_callback = EmbeddingVisualizationCallback(...)
# callbacks.append(embedding_vis_callback)
```

## Visualization Details

### Plot Structure

- **X/Y axes**: PCA/t-SNE components
- **Colors**: Window index (timestep in trajectory) - colormap shows temporal progression
- **Points**: Individual windows from trajectories
- **Labels**: Sample annotations showing `T{task_id}-E{episode_id}-W{window_index}`

### Color Scheme

The colorbar shows **Window/Segment Index (Timestep)**:
- **Dark purple** (low values): Early windows in trajectory
- **Yellow** (high values): Late windows in trajectory

This lets you see temporal flow through the embedding space.

### Interpretation

**What to look for:**

1. **Smooth trajectories**: Sequential windows should form continuous paths
2. **Temporal clustering**: Early/late windows may cluster separately
3. **Task separation**: Different trajectories should occupy different regions (if tasks differ)
4. **Loops and returns**: Repetitive behaviors show as loops in embedding space

**Example patterns:**

- **Linear progression**: Trajectory moves linearly through space
- **Circular/loop**: Repetitive cyclic behavior
- **Branching**: Different execution paths in similar tasks
- **Convergence**: Different trajectories end in similar regions

## W&B Integration

Plots are logged to W&B as:

```
embeddings/
├── train_trajectory    # Training set trajectory embeddings
└── val_trajectory      # Validation set trajectory embeddings
```

Each plot includes:
- Multiple trajectories (default: 3 per split)
- All windows from each trajectory (unless limited)
- Temporal coloring by window index
- Variance explained (PCA only)

## Performance

| Method | Time (3 trajectories) | When to Use |
|--------|----------------------|-------------|
| PCA    | ~2-5 seconds         | Default - fast, shows variance |
| t-SNE  | ~30-60 seconds       | Detailed analysis, preserves local structure |

**Recommendation**: Use PCA (default) for routine logging during training.

## Implementation Details

### How It Works

1. **Callback initialization**: Stores references to trajectory data and datasets
2. **On first epoch**: Generates baseline plot
3. **On best model save**: Generates updated plot when validation improves
4. **Extract trajectories**: Finds all windows belonging to selected trajectories
5. **Extract embeddings**: Concatenates tokens + DINO features for each window
6. **Reduce dimensions**: Applies PCA/t-SNE
7. **Plot**: Creates scatter plot colored by window index
8. **Log to W&B**: Uploads figure

### Data Flow

```
Trajectory Data (ParquetTrajectory)
    ↓
BoundaryDataset (windows with features)
    ↓
Extract embeddings (tokens + DINO) for each window
    ↓
Stack embeddings, track window indices
    ↓
Normalize → PCA/t-SNE → 2D points
    ↓
Plot with temporal coloring
    ↓
Log to W&B
```

### Memory & Performance

- Processes 3 trajectories by default (~100-300 windows)
- All computation on CPU (no GPU memory used)
- Plots are closed immediately after logging
- Minimal training overhead (~5 seconds per plot)

## Troubleshooting

### "No trajectory data available"
- Check that `dataset.trajectories` attribute exists
- Verify BoundaryDataset was created with trajectory data
- Ensure trajectories were passed to dataset constructor

### "No embeddings extracted"
- Check that trajectories have data in the dataset
- Verify task_id and episode_id matching works
- Try increasing `num_trajectories`

### Slow t-SNE
- Use PCA instead (default)
- Limit windows with `max_windows_per_traj=50`
- Reduce `num_trajectories=1`

### Plots not logging
- Verify `wandb.enabled=true` in config
- Check that WandBCallback is added to callbacks
- Look for warnings in training output

## Differences from Original Script

| Original `plot_fast_droid_embeddings.py` | This Implementation |
|-------------------------------------------|---------------------|
| Standalone script | Integrated callback |
| Uses URDF forward kinematics | Uses end effector poses from H5 |
| Processes full dataset offline | Samples few trajectories |
| Ray parallelization for DINO | Uses pre-extracted features from dataset |
| Saves plots to files | Logs to W&B |
| Manual execution | Automatic during training |

## Files

- `utils/trajectory_embedding_visualizer.py` - Core visualization logic
- `training/callbacks.py::EmbeddingVisualizationCallback` - Training integration
- `scripts/train.py` - Callback instantiation

## Future Enhancements

Potential improvements:
1. **Learned embeddings**: Visualize encoder outputs instead of raw inputs
2. **Animation**: Show trajectory progression as animated paths
3. **Interactive plots**: Use Plotly for zoom/pan
4. **Segment boundaries**: Mark boundaries on trajectories
5. **Comparison plots**: Show before/after training

---

**Created**: October 25, 2025
**Author**: Claude Code
**Version**: 2.0 (trajectory-based)
