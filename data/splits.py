"""Dataset splitting utilities for train/val/test splits.

This module provides functionality for creating stratified splits of trajectory data,
ensuring proper distribution of tasks across all splits and reproducibility.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .base_dataset import TrajectoryData


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by_task: bool = True
    min_trajectories_per_task: int = 3  # Minimum required for train/val/test

    def __post_init__(self):
        """Validate split ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative, train must be positive")


@dataclass
class SplitManifest:
    """Manifest storing split information for reproducibility."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    config: Dict[str, Any]
    task_distribution: Dict[str, Dict[str, int]]  # task_id -> {train: N, val: N, test: N}
    total_trajectories: int
    timestamp: str

    def save(self, path: Path):
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SplitManifest':
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class DatasetSplitter:
    """Stratified train/val/test splitting for trajectory datasets."""

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize splitter.

        Args:
            config: Split configuration. Uses defaults if None.
        """
        self.config = config or SplitConfig()

    def split_trajectories(
        self,
        trajectories: List[TrajectoryData],
        task_ids: Optional[List[int]] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split trajectories into train/val/test sets with stratification.

        Args:
            trajectories: List of trajectory data objects
            task_ids: Optional list of task IDs for each trajectory.
                     If None, extracted from trajectories.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)

        Raises:
            ValueError: If insufficient trajectories for stratified split
        """
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Get task IDs
        if task_ids is None:
            task_ids = [traj.task_id if hasattr(traj, 'task_id') else 0
                       for traj in trajectories]

        if len(task_ids) != len(trajectories):
            raise ValueError(f"task_ids length ({len(task_ids)}) must match "
                           f"trajectories length ({len(trajectories)})")

        # Group trajectories by task
        task_to_indices = defaultdict(list)
        for idx, task_id in enumerate(task_ids):
            task_to_indices[task_id].append(idx)

        # Validate sufficient data for stratification
        if self.config.stratify_by_task:
            self._validate_stratification(task_to_indices)

        # Perform stratified split
        if self.config.stratify_by_task and len(task_to_indices) > 1:
            return self._stratified_split(task_to_indices)
        else:
            # Simple random split if only one task or stratification disabled
            return self._random_split(len(trajectories))

    def _validate_stratification(self, task_to_indices: Dict[int, List[int]]):
        """Validate that each task has enough trajectories for stratified split."""
        for task_id, indices in task_to_indices.items():
            n_trajectories = len(indices)
            min_required = self.config.min_trajectories_per_task

            if n_trajectories < min_required:
                raise ValueError(
                    f"Task {task_id} has {n_trajectories} trajectories, "
                    f"but stratified split requires at least {min_required}. "
                    f"Either add more data or disable stratification."
                )

    def _stratified_split(
        self,
        task_to_indices: Dict[int, List[int]]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Perform stratified split ensuring each task is represented in all splits.

        Args:
            task_to_indices: Mapping from task_id to trajectory indices

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        train_indices = []
        val_indices = []
        test_indices = []

        for task_id, indices in task_to_indices.items():
            # Shuffle indices for this task
            task_indices = indices.copy()
            random.shuffle(task_indices)

            # Calculate split points
            n = len(task_indices)
            n_train = max(1, int(n * self.config.train_ratio))
            n_val = max(1, int(n * self.config.val_ratio)) if self.config.val_ratio > 0 else 0
            # Test gets remainder to ensure we use all data

            # Split
            train_indices.extend(task_indices[:n_train])
            val_indices.extend(task_indices[n_train:n_train + n_val])
            test_indices.extend(task_indices[n_train + n_val:])

        return train_indices, val_indices, test_indices

    def _random_split(self, n_trajectories: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Perform simple random split without stratification.

        Args:
            n_trajectories: Total number of trajectories

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Create and shuffle indices
        indices = list(range(n_trajectories))
        random.shuffle(indices)

        # Calculate split points
        n_train = int(n_trajectories * self.config.train_ratio)
        n_val = int(n_trajectories * self.config.val_ratio)

        # Split
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return train_indices, val_indices, test_indices

    def create_manifest(
        self,
        trajectories: List[TrajectoryData],
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        task_ids: Optional[List[int]] = None
    ) -> SplitManifest:
        """
        Create a split manifest for reproducibility.

        Args:
            trajectories: List of all trajectories
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            task_ids: Optional task IDs for distribution tracking

        Returns:
            SplitManifest object
        """
        from datetime import datetime

        # Get task IDs if not provided
        if task_ids is None:
            task_ids = [traj.task_id if hasattr(traj, 'task_id') else 0
                       for traj in trajectories]

        # Calculate task distribution
        task_distribution = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

        for idx in train_indices:
            task_distribution[str(task_ids[idx])]['train'] += 1
        for idx in val_indices:
            task_distribution[str(task_ids[idx])]['val'] += 1
        for idx in test_indices:
            task_distribution[str(task_ids[idx])]['test'] += 1

        return SplitManifest(
            train_indices=sorted(train_indices),
            val_indices=sorted(val_indices),
            test_indices=sorted(test_indices),
            config=asdict(self.config),
            task_distribution=dict(task_distribution),
            total_trajectories=len(trajectories),
            timestamp=datetime.now().isoformat()
        )


def load_split_indices(manifest_path: Path) -> Tuple[List[int], List[int], List[int]]:
    """
    Load train/val/test indices from a saved manifest.

    Args:
        manifest_path: Path to split manifest JSON file

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    manifest = SplitManifest.load(manifest_path)
    return manifest.train_indices, manifest.val_indices, manifest.test_indices


def print_split_summary(manifest: SplitManifest):
    """Print a summary of the dataset split."""
    print("\n" + "="*80)
    print("DATASET SPLIT SUMMARY")
    print("="*80)
    print(f"\nTotal trajectories: {manifest.total_trajectories}")
    print(f"Created: {manifest.timestamp}")
    print(f"\nSplit configuration:")
    print(f"  Train ratio: {manifest.config['train_ratio']:.2f}")
    print(f"  Val ratio:   {manifest.config['val_ratio']:.2f}")
    print(f"  Test ratio:  {manifest.config['test_ratio']:.2f}")
    print(f"  Seed:        {manifest.config['seed']}")
    print(f"  Stratified:  {manifest.config['stratify_by_task']}")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(manifest.train_indices)} ({len(manifest.train_indices)/manifest.total_trajectories*100:.1f}%)")
    print(f"  Val:   {len(manifest.val_indices)} ({len(manifest.val_indices)/manifest.total_trajectories*100:.1f}%)")
    print(f"  Test:  {len(manifest.test_indices)} ({len(manifest.test_indices)/manifest.total_trajectories*100:.1f}%)")

    print(f"\nTask distribution:")
    print(f"{'Task ID':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 50)

    for task_id in sorted(manifest.task_distribution.keys(), key=int):
        dist = manifest.task_distribution[task_id]
        total = dist['train'] + dist['val'] + dist['test']
        print(f"{task_id:<10} {dist['train']:<10} {dist['val']:<10} {dist['test']:<10} {total:<10}")

    print("="*80 + "\n")
