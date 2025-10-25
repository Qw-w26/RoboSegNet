"""Centralized path management for datasets and outputs."""

import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PathManager:
    """
    Centralized path management for the task segmentation project.

    This class provides a single source of truth for all paths used in the project,
    eliminating hardcoded paths scattered throughout the codebase.

    Attributes:
        root_dir: Root directory of the project
        data_dir: Base data directory
        output_dir: Base output directory
        cache_dir: Cache directory for preprocessed data
        checkpoint_dir: Checkpoint directory
        log_dir: Log directory
    """

    root_dir: Path
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    log_dir: Optional[Path] = None

    def __post_init__(self):
        """Set up default paths if not provided."""
        self.root_dir = Path(self.root_dir).resolve()

        if self.data_dir is None:
            # Check environment variable first
            env_data_dir = os.environ.get('TASK_SEG_DATA_DIR')
            if env_data_dir:
                self.data_dir = Path(env_data_dir)
            else:
                self.data_dir = self.root_dir / 'data'

        if self.output_dir is None:
            self.output_dir = self.root_dir / 'outputs'

        if self.cache_dir is None:
            self.cache_dir = self.root_dir / 'cache'

        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / 'checkpoints'

        if self.log_dir is None:
            self.log_dir = self.output_dir / 'logs'

        # Convert all to Path objects
        self.root_dir = Path(self.root_dir)
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

    def create_dirs(self, exist_ok: bool = True):
        """Create all directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.checkpoint_dir,
            self.log_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=exist_ok)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for a specific dataset."""
        return self.data_dir / dataset_name

    def get_cache_path(self, cache_name: str) -> Path:
        """Get path for a specific cache file."""
        return self.cache_dir / cache_name

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get path for a specific checkpoint."""
        return self.checkpoint_dir / checkpoint_name

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Get directory for a specific experiment."""
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_split_manifest_path(self, task_id: int) -> Path:
        """Get path for split manifest file."""
        splits_dir = self.data_dir / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        return splits_dir / f'task_{task_id}_split.json'

    def get_preprocessed_dir(self, task_id: int, split: str) -> Path:
        """
        Get directory for preprocessed data.

        Args:
            task_id: Task ID
            split: One of 'train', 'val', 'test'

        Returns:
            Path to preprocessed data directory
        """
        preprocessed_dir = self.cache_dir / 'preprocessed' / f'task_{task_id}' / split
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        return preprocessed_dir

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with string paths."""
        return {
            'root_dir': str(self.root_dir),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir),
        }

    @classmethod
    def from_env(cls) -> 'PathManager':
        """
        Create PathManager from environment variables.

        Environment variables:
            TASK_SEG_ROOT: Root directory (required)
            TASK_SEG_DATA_DIR: Data directory (optional)
            TASK_SEG_OUTPUT_DIR: Output directory (optional)
            TASK_SEG_CACHE_DIR: Cache directory (optional)

        Returns:
            PathManager instance
        """
        root = os.environ.get('TASK_SEG_ROOT')
        if not root:
            # Default to current working directory
            root = os.getcwd()

        return cls(
            root_dir=Path(root),
            data_dir=Path(os.environ['TASK_SEG_DATA_DIR']) if 'TASK_SEG_DATA_DIR' in os.environ else None,
            output_dir=Path(os.environ['TASK_SEG_OUTPUT_DIR']) if 'TASK_SEG_OUTPUT_DIR' in os.environ else None,
            cache_dir=Path(os.environ['TASK_SEG_CACHE_DIR']) if 'TASK_SEG_CACHE_DIR' in os.environ else None,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PathManager(\n"
            f"  root_dir={self.root_dir},\n"
            f"  data_dir={self.data_dir},\n"
            f"  output_dir={self.output_dir},\n"
            f"  cache_dir={self.cache_dir},\n"
            f"  checkpoint_dir={self.checkpoint_dir},\n"
            f"  log_dir={self.log_dir}\n"
            f")"
        )


# Global instance that can be imported
DEFAULT_PATH_MANAGER = PathManager(root_dir=Path(__file__).parent.parent)
