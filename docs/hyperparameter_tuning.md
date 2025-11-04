# Hyperparameter Tuning with Ray Tune

This guide explains how to use `scripts/tune.py` for automated hyperparameter search.

## Overview

The tuning script uses Ray Tune to perform parallel hyperparameter optimization across multiple configurations. It's based on the standard `train.py` script but adapted for distributed search.

## Quick Start

### Basic Usage

```bash
# Run hyperparameter search with default settings
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml

# Specify number of trials
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
    --num-trials 20

# Use multiple GPUs efficiently (fractional allocation)
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
    --num-trials 20 \
    --gpus-per-trial 0.25 \
    --max-concurrent-trials 4
```

### Advanced Options

```bash
# Use different search algorithm and scheduler
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
    --search-alg optuna \
    --scheduler pbt \
    --num-trials 30

# Optimize for different metric
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
    --metric val_loss \
    --mode min

# Custom experiment name and results directory
python scripts/tune.py --config configs/experiments/multi_task_327_352_356.yaml \
    --name my_tuning_run \
    --results-dir ./my_ray_results
```

## Search Space

By default, the following hyperparameters are tuned:

### Core Hyperparameters
- **learning_rate**: Log-uniform in [1e-5, 1e-3]
- **classification_threshold**: Uniform in [0.3, 0.6]
- **weight_decay**: Log-uniform in [1e-6, 1e-3]
- **batch_size**: Choice of [512, 1024, 2048, 4096]

### Regularization
- **transformer_dropout**: Uniform in [0.1, 0.5]
- **mlp_dropout**: Uniform in [0.1, 0.5]
- **feature_noise_std**: Uniform in [0.0, 0.1]

### Loss-specific Parameters (if applicable)
- **loss_pos_weight**: Uniform in [5.0, 20.0]
- **loss_alpha**: Uniform in [0.1, 0.5] (for focal losses)
- **loss_gamma**: Uniform in [1.0, 3.0] (for focal losses)

## Search Algorithms

### HyperOpt (Default)
Uses Tree-structured Parzen Estimator (TPE) for Bayesian optimization.

```bash
--search-alg hyperopt
```

### Optuna
Alternative Bayesian optimization algorithm.

```bash
--search-alg optuna
```

### Random Search
Simple random sampling (baseline).

```bash
--search-alg random
```

## Schedulers

### ASHA (Default)
Asynchronous Successive Halving Algorithm - efficiently terminates poor trials early.

```bash
--scheduler asha
```

**Parameters:**
- `grace_period=10`: Minimum epochs before early stopping
- `reduction_factor=3`: Reduction factor for successive halving
- `max_t=100`: Maximum training epochs

### Population Based Training (PBT)
Evolves hyperparameters during training.

```bash
--scheduler pbt
```

**Note:** PBT works best with longer training runs and modifies hyperparameters online.

### No Scheduler
Run all trials to completion.

```bash
--scheduler none
```

## Resource Allocation

### GPU Sharing
You can run multiple trials on a single GPU by using fractional allocation:

```bash
# Run 4 trials on 1 GPU simultaneously
--gpus-per-trial 0.25 --max-concurrent-trials 4

# Run 2 trials on 1 GPU
--gpus-per-trial 0.5 --max-concurrent-trials 2
```

### CPU Allocation
```bash
--cpus-per-trial 4  # Allocate 4 CPUs per trial
```

## Monitoring

### Real-time Progress
The script displays a live table showing:
- Trial ID and status
- Current metric values (val_f1, val_loss, train_loss)
- Epoch progress

### TensorBoard Integration
Ray Tune automatically logs to TensorBoard:

```bash
tensorboard --logdir ./ray_results
```

## Results

### Best Trial
After completion, the script prints:
- Best trial ID
- Best metric value
- Best hyperparameters
- Path to best checkpoint

### Results Directory Structure
```
ray_results/
└── tune_multi_task_327_352_356/
    ├── trial_00000/
    │   ├── checkpoint_000000/
    │   ├── events.out.tfevents...
    │   └── result.json
    ├── trial_00001/
    │   └── ...
    └── best_config.txt
```

### Loading Best Model
```python
import torch
from pathlib import Path

# Load best checkpoint
checkpoint_path = Path("ray_results/tune_multi_task_327_352_356/trial_00005/checkpoint_000050/checkpoint.pt")
checkpoint = torch.load(checkpoint_path)

# Load into model
model.load_state_dict(checkpoint['model_state_dict'])
```

## Example Workflows

### Quick Development Tuning
Fast search with few trials for rapid iteration:

```bash
python scripts/tune.py \
    --config configs/experiments/multi_task_327_352_356.yaml \
    --num-trials 5 \
    --scheduler none \
    --search-alg random
```

### Production Tuning
Comprehensive search with early stopping:

```bash
python scripts/tune.py \
    --config configs/experiments/multi_task_327_352_356.yaml \
    --num-trials 50 \
    --scheduler asha \
    --search-alg hyperopt \
    --gpus-per-trial 0.5 \
    --max-concurrent-trials 4
```

### Multi-GPU Tuning
Efficient use of multiple GPUs:

```bash
# On machine with 4 GPUs, run 8 concurrent trials
python scripts/tune.py \
    --config configs/experiments/multi_task_327_352_356.yaml \
    --num-trials 100 \
    --gpus-per-trial 0.5 \
    --max-concurrent-trials 8
```

## Customizing Search Space

To modify the search space, edit the `get_search_space()` function in `scripts/tune.py`:

```python
def get_search_space(base_config_path: str) -> Dict[str, Any]:
    search_space = {
        # Modify ranges
        'learning_rate': tune.loguniform(1e-6, 1e-2),  # Wider range

        # Add new parameters
        'label_smoother_sigma': tune.uniform(0.5, 2.0),

        # Remove parameters
        # 'batch_size': tune.choice([512, 1024, 2048, 4096]),  # Commented out
    }
    return search_space
```

## Tips and Best Practices

1. **Start Small**: Begin with a few trials to validate the setup
2. **Use ASHA**: The ASHA scheduler is usually the best default choice
3. **GPU Sharing**: Use fractional GPUs to maximize throughput
4. **Metric Selection**: Optimize for the metric that matters most to your application (F1, precision, recall, etc.)
5. **Checkpoint Management**: Ray Tune automatically keeps only the best checkpoint to save disk space
6. **Monitor Resources**: Use `nvidia-smi` to check GPU utilization
7. **Resume Support**: Ray Tune supports resuming interrupted experiments automatically

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in search space
- Reduce `--gpus-per-trial` to give each trial more memory
- Reduce `--max-concurrent-trials`

### Slow Progress
- Increase `--max-concurrent-trials` if you have GPU capacity
- Use ASHA scheduler to terminate poor trials early
- Reduce `num_workers` in base config

### All Trials Failing
- Check base config works with `train.py` first
- Verify GPU availability
- Check data loading paths are correct

## Command Reference

```bash
python scripts/tune.py \
    --config CONFIG_PATH \           # Required: base YAML config
    --num-trials N \                 # Number of trials (default: 10)
    --gpus-per-trial G \            # GPUs per trial (default: 1.0)
    --cpus-per-trial C \            # CPUs per trial (default: 4)
    --max-concurrent-trials M \     # Max concurrent trials (default: 4)
    --search-alg {random,hyperopt,optuna} \  # Search algorithm (default: hyperopt)
    --scheduler {asha,pbt,none} \   # Trial scheduler (default: asha)
    --metric METRIC \               # Metric to optimize (default: val_f1)
    --mode {min,max} \              # Optimization mode (default: max)
    --results-dir DIR \             # Results directory (default: ./ray_results)
    --name NAME                     # Experiment name
```
