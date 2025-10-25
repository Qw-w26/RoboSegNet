"""Configuration module for boundary prediction experiments."""

from .config_schema import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    WandBConfig,
)
from .config_loader import (
    load_config,
    load_yaml,
    merge_configs,
    apply_cli_overrides,
    parse_cli_overrides,
    validate_config,
)

__all__ = [
    # Schema
    'ExperimentConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'WandBConfig',

    # Loader
    'load_config',
    'load_yaml',
    'merge_configs',
    'apply_cli_overrides',
    'parse_cli_overrides',
    'validate_config',
]
