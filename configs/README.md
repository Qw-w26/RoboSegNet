# Configuration System

This directory contains YAML configuration files for boundary prediction experiments.

## Structure

```
configs/
├── base/               # Base configurations with defaults
│   ├── model.yaml      # Model architecture defaults
│   ├── data.yaml       # Data loading defaults
│   ├── training.yaml   # Training loop defaults
│   └── wandb.yaml      # WandB logging defaults
├── experiments/        # Experiment-specific configs
│   ├── task_357.yaml   # Full config for task 357
│   └── debug.yaml      # Fast debug configuration
└── README.md           # This file
```

## Usage

### Basic Training

```bash
# Train with experiment config
python scripts/train.py --config configs/experiments/task_357.yaml
```

### With Overrides

Override any config value from command line:

```bash
# Override learning rate
python scripts/train.py --config configs/experiments/task_357.yaml \
    training.learning_rate=0.001

# Override multiple values
python scripts/train.py --config configs/experiments/task_357.yaml \
    training.learning_rate=0.001 \
    data.batch_size=256 \
    model.vocab_size=4096
```

### Quick Debug

```bash
# Use debug config for fast iteration
python scripts/train.py --config configs/experiments/debug.yaml
```

## Creating New Experiments

1. Copy an existing experiment config:
   ```bash
   cp configs/experiments/task_357.yaml configs/experiments/my_experiment.yaml
   ```

2. Edit the config file to change parameters

3. Run training:
   ```bash
   python scripts/train.py --config configs/experiments/my_experiment.yaml
   ```

## Configuration Schema

See `config/config_schema.py` for the complete configuration schema and documentation.

### Key Sections

- **model**: Neural network architecture
  - vocab_size, hidden_dims, dropout rates, etc.

- **data**: Data loading and preprocessing
  - batch_size, window_size, label smoothing, etc.

- **training**: Training loop parameters
  - num_epochs, learning_rate, loss function, etc.

- **wandb**: Weights & Biases logging
  - project, name, tags, etc.

## Tips

1. **Start with base configs**: Copy from `base/` and override only what you need

2. **Use debug config**: Test changes with `debug.yaml` before full training

3. **Version control your configs**: Configs are small and should be committed to git

4. **Descriptive names**: Use clear experiment names that describe what you're testing

5. **Document your experiments**: Add notes in the `wandb.notes` field
