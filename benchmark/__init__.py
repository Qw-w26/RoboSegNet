"""
Task Segmentation Benchmark Framework

A modular framework for benchmarking task segmentation methods on robotics datasets.
"""

from .data import (
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation,
    AgiBotWorldDataset,
    AgiBotHFDirectDataset
)
from .methods import (
    BaseSegmentationMethod,
    WindowBasedSegmentationMethod,
    SpectralClusteringMethod,
    ChangePointDetectionMethod,
    GMMMethod,
    HierarchicalClusteringMethod,
    KMeansMethod
)
from .metrics import (
    SegmentationMetrics,
    evaluate_segmentation,
    aggregate_metrics
)
from .benchmark_runner import BenchmarkRunner
from .embeddings import (
    FastDROIDEmbedder,
    load_fast_droid_tokenizer,
    create_embedding_function
)

__version__ = "0.1.0"

__all__ = [
    # Data
    'BaseSegmentationDataset',
    'TrajectoryData',
    'SegmentAnnotation',
    'AgiBotWorldDataset',
    'AgiBotHFDirectDataset',
    # Methods
    'BaseSegmentationMethod',
    'WindowBasedSegmentationMethod',
    'SpectralClusteringMethod',
    'ChangePointDetectionMethod',
    'GMMMethod',
    'HierarchicalClusteringMethod',
    'KMeansMethod',
    # Metrics
    'SegmentationMetrics',
    'evaluate_segmentation',
    'aggregate_metrics',
    # Runner
    'BenchmarkRunner',
    # Embeddings
    'FastDROIDEmbedder',
    'load_fast_droid_tokenizer',
    'create_embedding_function',
]
