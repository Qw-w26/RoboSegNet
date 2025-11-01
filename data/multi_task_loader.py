"""Multi-task data loader for combined training across tasks.

This module provides functionality to load and combine trajectory data from
multiple task directories for multi-task training and cross-task evaluation.

Example:
    >>> from data import load_multi_task_split
    >>>
    >>> # Load training data from multiple tasks
    >>> train_trajectories = load_multi_task_split(
    ...     'train',
    ...     task_ids=[327, 352, 356],
    ...     meta_dir='./full_dataset'
    ... )
    >>>
    >>> # Load evaluation data from different tasks
    >>> val_trajectories = load_multi_task_split(
    ...     'val',
    ...     task_ids=[357],
    ...     meta_dir='./full_dataset'
    ... )
"""

from pathlib import Path
from typing import List, Dict, Any
from .parquet_loader import load_parquet_split, ParquetTrajectory


def load_multi_task_split(
    split_name: str,
    task_ids: List[int],
    meta_dir: str
) -> List[ParquetTrajectory]:
    """
    Load and combine trajectories from multiple tasks.

    This function loads data from multiple task_*_processed directories and
    combines them into a single list of trajectories. This enables:
    - Multi-task training (train on multiple tasks simultaneously)
    - Cross-task evaluation (train on some tasks, evaluate on others)

    Args:
        split_name: Name of split to load ('train', 'val', or 'test')
        task_ids: List of task IDs to load (e.g., [327, 352, 356])
        meta_dir: Path to directory containing task_*_processed folders

    Returns:
        Combined list of ParquetTrajectory objects from all specified tasks

    Raises:
        FileNotFoundError: If meta_dir or any task directory doesn't exist
        ValueError: If split_name is invalid or task_ids is empty

    Example:
        >>> # Single task
        >>> trajectories = load_multi_task_split('train', [327], './full_dataset')
        >>> print(f"Loaded {len(trajectories)} trajectories from task 327")
        >>>
        >>> # Multi-task
        >>> trajectories = load_multi_task_split(
        ...     'train',
        ...     [327, 352, 356],
        ...     './full_dataset'
        ... )
        >>> print(f"Loaded {len(trajectories)} trajectories from 3 tasks")
    """
    # Validate inputs
    if not task_ids:
        raise ValueError("task_ids cannot be empty")

    valid_splits = {'train', 'val', 'test'}
    if split_name not in valid_splits:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be one of {valid_splits}")

    meta_path = Path(meta_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

    # Load trajectories from each task
    all_trajectories = []
    task_stats = []

    print(f"\nLoading {split_name} split from {len(task_ids)} task(s):")

    for task_id in task_ids:
        task_dir = meta_path / f"task_{task_id}_processed"

        if not task_dir.exists():
            raise FileNotFoundError(
                f"Task directory not found: {task_dir}\n"
                f"Expected directory: task_{task_id}_processed in {meta_dir}"
            )

        # Load trajectories for this task using existing loader
        trajectories = load_parquet_split(split_name, splits_dir=str(task_dir))

        # Collect statistics
        num_traj = len(trajectories)
        all_trajectories.extend(trajectories)
        task_stats.append((task_id, num_traj))

        print(f"  Task {task_id}: {num_traj} trajectories")

    # Print summary
    total_trajectories = len(all_trajectories)
    print(f"✓ Total {split_name} trajectories: {total_trajectories}")

    if len(task_ids) > 1:
        # Print distribution for multi-task
        print(f"  Task distribution:")
        for task_id, num_traj in task_stats:
            percentage = (num_traj / total_trajectories) * 100
            print(f"    Task {task_id}: {num_traj} ({percentage:.1f}%)")

    print()  # Empty line for readability

    return all_trajectories


def get_multi_task_statistics(
    task_ids: List[int],
    meta_dir: str,
    split_name: str = 'train'
) -> Dict[str, Any]:
    """
    Get statistics about multi-task dataset without loading all data.

    Useful for understanding dataset composition before training.

    Args:
        task_ids: List of task IDs
        meta_dir: Path to meta directory
        split_name: Split to analyze ('train', 'val', or 'test')

    Returns:
        Dictionary with statistics:
        {
            'num_tasks': int,
            'total_trajectories': int,
            'task_breakdown': {task_id: num_trajectories, ...},
            'task_names': {task_id: task_name, ...}
        }

    Example:
        >>> stats = get_multi_task_statistics([327, 352, 356], './full_dataset')
        >>> print(f"Total trajectories: {stats['total_trajectories']}")
        >>> print(f"Tasks: {stats['task_breakdown']}")
    """
    meta_path = Path(meta_dir)

    task_breakdown = {}
    task_names = {}
    total_trajectories = 0

    for task_id in task_ids:
        task_dir = meta_path / f"task_{task_id}_processed"
        if not task_dir.exists():
            continue

        # Load just to get count and task name
        trajectories = load_parquet_split(split_name, splits_dir=str(task_dir))

        num_traj = len(trajectories)
        task_breakdown[task_id] = num_traj
        total_trajectories += num_traj

        # Get task name from first trajectory
        if trajectories:
            task_names[task_id] = trajectories[0].task_name

    return {
        'num_tasks': len(task_ids),
        'total_trajectories': total_trajectories,
        'task_breakdown': task_breakdown,
        'task_names': task_names,
    }


__all__ = [
    'load_multi_task_split',
    'get_multi_task_statistics',
]
