"""Reproducibility utilities for ensuring deterministic experiments."""

import os
import random
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import torch


def set_all_seeds(seed: int):
    """
    Set seeds for all random number generators to ensure reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variables for Python hashing

    Additionally, it configures PyTorch for deterministic operation.

    Args:
        seed: Random seed value

    Example:
        >>> set_all_seeds(42)
        >>> # Now all random operations will be deterministic
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for Python hashing
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Note: Some operations in PyTorch are non-deterministic by default
    # even with these settings (e.g., some CUDA operations)
    # Set the following for maximum determinism (may impact performance):
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Commit hash string or None if not in a git repository
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_diff() -> Optional[str]:
    """
    Get git diff of uncommitted changes.

    Returns:
        Diff string or None if not in a git repository
    """
    try:
        diff = subprocess.check_output(
            ['git', 'diff', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8')
        return diff if diff else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_installed_packages() -> Dict[str, str]:
    """
    Get list of installed Python packages and their versions.

    Returns:
        Dictionary mapping package names to versions
    """
    try:
        import pkg_resources
        packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        }
        return packages
    except Exception:
        return {}


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility tracking.

    Returns:
        Dictionary with system information
    """
    import platform
    import sys

    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_implementation': platform.python_implementation(),
        'cpu_count': os.cpu_count(),
    }

    # Add PyTorch info
    info['torch_version'] = torch.__version__
    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]

    return info


def save_experiment_state(
    config: Dict[str, Any],
    output_dir: Path,
    include_packages: bool = True,
    include_git: bool = True
) -> Path:
    """
    Save complete experiment state for reproducibility.

    This saves:
    - Configuration used
    - Git commit and diff (if in git repo)
    - Installed packages and versions
    - System information
    - Timestamp

    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save state file
        include_packages: Whether to include package list (can be large)
        include_git: Whether to include git information

    Returns:
        Path to saved state file

    Example:
        >>> config = {'learning_rate': 0.001, 'batch_size': 32}
        >>> save_experiment_state(config, Path('./outputs/exp1'))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
    }

    if include_git:
        state['git_commit'] = get_git_commit()
        state['git_diff'] = get_git_diff()

    if include_packages:
        state['packages'] = get_installed_packages()

    # Save to JSON
    state_path = output_dir / 'experiment_state.json'
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return state_path


def load_experiment_state(state_path: Path) -> Dict[str, Any]:
    """
    Load saved experiment state.

    Args:
        state_path: Path to experiment state JSON file

    Returns:
        Dictionary with experiment state
    """
    with open(state_path, 'r') as f:
        state = json.load(f)
    return state


def verify_reproducibility(
    state_path: Path,
    warn_on_diff: bool = True
) -> bool:
    """
    Verify that current environment matches saved experiment state.

    Args:
        state_path: Path to saved experiment state
        warn_on_diff: Whether to print warnings on differences

    Returns:
        True if environment matches, False otherwise
    """
    saved_state = load_experiment_state(state_path)
    current_info = get_system_info()

    differences = []

    # Check PyTorch version
    if saved_state['system_info']['torch_version'] != current_info['torch_version']:
        differences.append(
            f"PyTorch version mismatch: "
            f"{saved_state['system_info']['torch_version']} vs {current_info['torch_version']}"
        )

    # Check CUDA availability
    if saved_state['system_info']['cuda_available'] != current_info['cuda_available']:
        differences.append(
            f"CUDA availability mismatch: "
            f"{saved_state['system_info']['cuda_available']} vs {current_info['cuda_available']}"
        )

    # Check git commit if available
    if 'git_commit' in saved_state and saved_state['git_commit']:
        current_commit = get_git_commit()
        if current_commit != saved_state['git_commit']:
            differences.append(
                f"Git commit mismatch: "
                f"{saved_state['git_commit'][:8]} vs {current_commit[:8] if current_commit else 'None'}"
            )

    if differences and warn_on_diff:
        print("WARNING: Environment differences detected:")
        for diff in differences:
            print(f"  - {diff}")
        print("\nResults may not be perfectly reproducible.")

    return len(differences) == 0
