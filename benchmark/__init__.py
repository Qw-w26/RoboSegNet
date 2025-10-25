"""
Task Segmentation Benchmark Framework

A modular framework for benchmarking task segmentation methods on robotics datasets.
"""

from data import (
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation,
    AgiBotWorldDataset,
)
from benchmark.methods import (
    BaseSegmentationMethod,
    WindowBasedSegmentationMethod,
    SpectralClusteringMethod,
    ChangePointDetectionMethod,
    GMMMethod,
    HierarchicalClusteringMethod,
    KMeansMethod
)
from benchmark.metrics import (
    SegmentationMetrics,
    evaluate_segmentation,
    aggregate_metrics
)
from benchmark.benchmark_runner import BenchmarkRunner
from embedding import (
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
