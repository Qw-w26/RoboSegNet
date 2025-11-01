"""Data loading modules for task segmentation benchmarks."""

# Modern parquet-based data loading (preferred)
from .parquet_loader import (
    Segment,
    ProprioData,
    ParquetTrajectory,
    load_parquet_split,
    get_split_statistics,
    print_split_statistics,
)

# Multi-task data loading
from .multi_task_loader import (
    load_multi_task_split,
    get_multi_task_statistics,
)

# Legacy dataset classes (used by prepare_data.py)
from .base_dataset import (
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation
)
from .agibot_dataset import AgiBotWorldDataset, AgiBotTrajectoryData, download_agibot_dataset

__all__ = [
    # Modern parquet loading (use this for training)
    'Segment',
    'ProprioData',
    'ParquetTrajectory',
    'load_parquet_split',
    'get_split_statistics',
    'print_split_statistics',

    # Multi-task loading
    'load_multi_task_split',
    'get_multi_task_statistics',

    # Legacy classes (for prepare_data.py)
    'BaseSegmentationDataset',
    'TrajectoryData',
    'SegmentAnnotation',
    'AgiBotWorldDataset',
    'AgiBotTrajectoryData',
    'download_agibot_dataset',
]
