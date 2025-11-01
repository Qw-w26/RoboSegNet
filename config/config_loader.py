"""Configuration loading utilities with YAML and CLI override support.

This module provides functions to load experiment configurations from YAML files
with support for:
- Base config + experiment overrides
- Command-line overrides
- Type validation
- Nested config access

Example:
    >>> from config import load_config
    >>>
    >>> # Load from YAML
    >>> config = load_config('configs/experiments/task_357.yaml')
    >>>
    >>> # With CLI overrides
    >>> overrides = {'training.learning_rate': 1e-3, 'data.batch_size': 256}
    >>> config = load_config('configs/experiments/task_357.yaml', overrides)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from omegaconf import OmegaConf, DictConfig

from .config_schema import ExperimentConfig


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load YAML file into dictionary.

    Args:
        yaml_path: Path to YAML file

    Returns:
        Dictionary with config data

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return config_dict or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def apply_cli_overrides(
    config_dict: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply CLI overrides to config dictionary.

    Supports dot notation for nested keys:
        'training.learning_rate': 1e-3  →  config['training']['learning_rate'] = 1e-3

    Args:
        config_dict: Base configuration dictionary
        overrides: Dictionary of override keys and values

    Returns:
        Updated configuration dictionary
    """
    config = OmegaConf.create(config_dict)

    for key, value in overrides.items():
        # Use OmegaConf for clean nested key setting
        OmegaConf.update(config, key, value, merge=False)

    return OmegaConf.to_container(config, resolve=True)


def parse_override_string(override_str: str) -> tuple[str, Any]:
    """Parse override string into key and value.

    Supports formats:
        'key=value'
        'nested.key=value'

    Args:
        override_str: Override string (e.g., 'training.lr=1e-3')

    Returns:
        Tuple of (key, value)

    Example:
        >>> parse_override_string('training.learning_rate=0.001')
        ('training.learning_rate', 0.001)
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value'")

    key, value_str = override_str.split('=', 1)
    key = key.strip()

    # Try to infer type
    value_str = value_str.strip()

    # Boolean
    if value_str.lower() in ('true', 'false'):
        value = value_str.lower() == 'true'
    # None/null
    elif value_str.lower() in ('none', 'null'):
        value = None
    # Number
    elif value_str.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
        value = float(value_str) if '.' in value_str or 'e' in value_str else int(value_str)
    # String
    else:
        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            value = value_str[1:-1]
        else:
            value = value_str

    return key, value


def parse_cli_overrides(override_strings: List[str]) -> Dict[str, Any]:
    """Parse list of override strings into dictionary.

    Args:
        override_strings: List of override strings

    Returns:
        Dictionary of overrides

    Example:
        >>> overrides = parse_cli_overrides([
        ...     'training.learning_rate=0.001',
        ...     'model.vocab_size=4096'
        ... ])
    """
    overrides = {}
    for override_str in override_strings:
        key, value = parse_override_string(override_str)
        overrides[key] = value

    return overrides


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    base_config_path: Optional[str] = None
) -> ExperimentConfig:
    """Load experiment configuration from YAML with optional overrides.

    Args:
        config_path: Path to experiment YAML config file
        overrides: Optional dictionary of overrides (supports dot notation)
        base_config_path: Optional path to base config to merge with

    Returns:
        ExperimentConfig instance

    Example:
        >>> # Simple loading
        >>> config = load_config('configs/experiments/task_357.yaml')
        >>>
        >>> # With overrides
        >>> config = load_config(
        ...     'configs/experiments/task_357.yaml',
        ...     overrides={'training.learning_rate': 1e-3}
        ... )
        >>>
        >>> # With base config
        >>> config = load_config(
        ...     'configs/experiments/task_357.yaml',
        ...     base_config_path='configs/base/default.yaml'
        ... )
    """
    # Load experiment config
    config_dict = load_yaml(config_path)

    # Load and merge base config if specified
    if base_config_path:
        base_dict = load_yaml(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)

    # Apply overrides
    if overrides:
        config_dict = apply_cli_overrides(config_dict, overrides)

    # Create ExperimentConfig instance
    try:
        config = ExperimentConfig.from_dict(config_dict)
    except Exception as e:
        raise ValueError(f"Failed to create config from {config_path}: {e}")

    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate configuration values.

    Args:
        config: Experiment configuration to validate

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required directories don't exist
    """
    # Multi-task data validation
    meta_path = Path(config.data.meta_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_dir does not exist: {config.data.meta_dir}")

    if not config.data.train_task_ids:
        raise ValueError("train_task_ids cannot be empty")

    # Validate all task directories exist
    all_task_ids = set(config.data.train_task_ids) | set(config.data.eval_task_ids)

    for task_id in all_task_ids:
        task_dir = meta_path / f"task_{task_id}_processed"
        if not task_dir.exists():
            raise FileNotFoundError(
                f"Task directory not found: {task_dir}\n"
                f"Expected directory: task_{task_id}_processed in {config.data.meta_dir}"
            )

    # Model validation
    if config.model.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if config.model.prediction_window_size <= 0:
        raise ValueError("prediction_window_size must be positive")

    # Data validation
    if config.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.data.window_size <= 0:
        raise ValueError("window_size must be positive")

    if config.data.stride <= 0:
        raise ValueError("stride must be positive")

    # Training validation
    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    # Consistency checks
    if config.model.vocab_size != config.data.vocab_size:
        raise ValueError(
            f"Model vocab_size ({config.model.vocab_size}) must match "
            f"data vocab_size ({config.data.vocab_size})"
        )


__all__ = [
    'load_config',
    'load_yaml',
    'merge_configs',
    'apply_cli_overrides',
    'parse_cli_overrides',
    'parse_override_string',
    'validate_config',
]
