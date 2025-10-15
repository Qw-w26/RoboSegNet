"""
Benchmark runner for task segmentation methods.

This module provides a unified interface for running and evaluating
different segmentation methods on benchmark datasets.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from dataclasses import asdict
import time

from .data import BaseSegmentationDataset, TrajectoryData
from .methods import BaseSegmentationMethod
from .metrics import SegmentationMetrics, evaluate_segmentation, aggregate_metrics


class BenchmarkRunner:
    """
    Runner for task segmentation benchmarks.

    Handles:
    - Running multiple methods on a dataset
    - Computing and aggregating metrics
    - Saving results
    - Visualization
    """

    def __init__(
        self,
        dataset: BaseSegmentationDataset,
        methods: List[BaseSegmentationMethod],
        embedding_fn: Optional[Callable] = None,
        output_dir: Path = Path("./benchmark_results"),
        boundary_tolerance: int = 5
    ):
        """
        Initialize benchmark runner.

        Args:
            dataset: Dataset to benchmark on
            methods: List of segmentation methods to evaluate
            embedding_fn: Function to compute embeddings from TrajectoryData
            output_dir: Directory to save results
            boundary_tolerance: Tolerance for boundary detection (frames)
        """
        self.dataset = dataset
        self.methods = methods
        self.embedding_fn = embedding_fn
        self.output_dir = Path(output_dir)
        self.boundary_tolerance = boundary_tolerance

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.results = {}

    def run_method(
        self,
        method: BaseSegmentationMethod,
        trajectories: Optional[List[TrajectoryData]] = None,
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single method on the dataset.

        Args:
            method: Segmentation method to evaluate
            trajectories: Optional subset of trajectories to evaluate on
            save_predictions: Whether to save per-trajectory predictions

        Returns:
            Dictionary with results
        """
        if trajectories is None:
            trajectories = self.dataset.trajectories

        print(f"\n{'='*70}")
        print(f"Evaluating: {method.name}")
        print(f"{'='*70}")

        # Fit method on full dataset if needed
        print("Fitting method...")
        all_embeddings = []
        if self.embedding_fn:
            for traj in tqdm(self.dataset.trajectories, desc="Computing embeddings for fitting"):
                embeddings = self.embedding_fn(traj)
                if embeddings is not None:
                    all_embeddings.append(embeddings)

        if all_embeddings:
            # Pad embeddings to same dimension
            max_dim = max(emb.shape[1] for emb in all_embeddings)
            padded_embeddings = []
            for emb in all_embeddings:
                if emb.shape[1] < max_dim:
                    padding = np.zeros((emb.shape[0], max_dim - emb.shape[1]))
                    emb = np.hstack([emb, padding])
                padded_embeddings.append(emb)

            all_embeddings = np.vstack(padded_embeddings)
            method.fit(self.dataset.trajectories, all_embeddings)
        else:
            method.fit(self.dataset.trajectories)

        # Evaluate on trajectories
        per_trajectory_metrics = []
        per_trajectory_predictions = {}
        total_time = 0

        for traj in tqdm(trajectories, desc=f"Evaluating {method.name}"):
            # Skip if no ground truth
            if traj.num_segments == 0:
                continue

            # Compute embeddings for this trajectory
            embeddings = None
            if self.embedding_fn:
                embeddings = self.embedding_fn(traj)

            # Predict
            start_time = time.time()
            try:
                y_pred = method.predict(traj, embeddings)
            except Exception as e:
                print(f"Warning: Failed to predict for trajectory {traj.trajectory_id}: {e}")
                continue
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # Get ground truth
            y_true = traj.get_segment_labels()

            # Handle length mismatch
            if len(y_pred) != len(y_true):
                min_len = min(len(y_pred), len(y_true))
                y_pred = y_pred[:min_len]
                y_true = y_true[:min_len]

            # Evaluate
            try:
                metrics = evaluate_segmentation(
                    y_true,
                    y_pred,
                    boundary_tolerance=self.boundary_tolerance
                )
                per_trajectory_metrics.append(metrics)

                if save_predictions:
                    per_trajectory_predictions[traj.trajectory_id] = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'metrics': metrics.to_dict()
                    }
            except Exception as e:
                print(f"Warning: Failed to compute metrics for trajectory {traj.trajectory_id}: {e}")
                continue

        # Aggregate metrics
        if per_trajectory_metrics:
            aggregated_metrics = aggregate_metrics(per_trajectory_metrics)
        else:
            # Return empty metrics if no trajectories were processed
            aggregated_metrics = SegmentationMetrics(
                frame_accuracy=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                macro_f1=0.0,
                weighted_precision=0.0,
                weighted_recall=0.0,
                weighted_f1=0.0,
                adjusted_rand_index=0.0,
                normalized_mutual_info=0.0
            )

        # Print results
        aggregated_metrics.print_summary(method.name)

        # Store results
        results = {
            'method_name': method.name,
            'method_config': method.get_config(),
            'aggregated_metrics': aggregated_metrics.to_dict(),
            'per_trajectory_metrics': [m.to_dict() for m in per_trajectory_metrics],
            'num_trajectories': len(per_trajectory_metrics),
            'total_time': total_time,
            'avg_time_per_trajectory': total_time / len(per_trajectory_metrics) if per_trajectory_metrics else 0
        }

        if save_predictions:
            results['predictions'] = per_trajectory_predictions

        return results

    def run_all_methods(self, save_predictions: bool = False) -> Dict[str, Any]:
        """
        Run all methods on the dataset.

        Args:
            save_predictions: Whether to save per-trajectory predictions

        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*70}")
        print(f"Running Benchmark on {self.dataset.__class__.__name__}")
        print(f"Dataset: {len(self.dataset)} trajectories")
        print(f"Methods: {len(self.methods)}")
        print(f"{'='*70}\n")

        all_results = {}

        for method in self.methods:
            results = self.run_method(method, save_predictions=save_predictions)
            all_results[method.name] = results
            self.results[method.name] = results

        # Save results
        self.save_results(all_results)

        # Print comparison
        self.print_comparison(all_results)

        return all_results

    def save_results(self, results: Dict[str, Any]):
        """Save results to disk."""
        # Save as JSON
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            # Remove numpy arrays and non-serializable objects
            serializable_results = {}
            for method_name, method_results in results.items():
                serializable_results[method_name] = {
                    'method_name': method_results['method_name'],
                    'method_config': method_results['method_config'],
                    'aggregated_metrics': method_results['aggregated_metrics'],
                    'num_trajectories': method_results['num_trajectories'],
                    'total_time': method_results['total_time'],
                    'avg_time_per_trajectory': method_results['avg_time_per_trajectory']
                }
            json.dump(serializable_results, f, indent=2)

        print(f"\nSaved results to {json_path}")

        # Save full results with predictions as pickle
        pickle_path = self.output_dir / 'results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Saved full results to {pickle_path}")

    def print_comparison(self, results: Dict[str, Any]):
        """Print comparison table of all methods."""
        print(f"\n{'='*70}")
        print("Method Comparison")
        print(f"{'='*70}\n")

        # Header
        print(f"{'Method':<30} {'Acc':>8} {'F1':>8} {'ARI':>8} {'B-F1':>8} {'Time':>8}")
        print("-" * 70)

        # Sort by F1 score
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1]['aggregated_metrics']['macro_f1'],
            reverse=True
        )

        # Print each method
        for method_name, method_results in sorted_methods:
            metrics = method_results['aggregated_metrics']
            avg_time = method_results['avg_time_per_trajectory']

            acc = metrics['frame_accuracy']
            f1 = metrics['macro_f1']
            ari = metrics['adjusted_rand_index']
            b_f1 = metrics.get('boundary_f1', 0.0) or 0.0
            time_str = f"{avg_time:.2f}s"

            print(f"{method_name:<30} {acc:>8.4f} {f1:>8.4f} {ari:>8.4f} {b_f1:>8.4f} {time_str:>8}")

        print(f"\n{'='*70}\n")

    def get_best_method(self, metric: str = 'macro_f1') -> str:
        """
        Get the best performing method based on a metric.

        Args:
            metric: Metric to use for ranking

        Returns:
            Name of the best method
        """
        if not self.results:
            raise ValueError("No results available. Run benchmark first.")

        best_method = max(
            self.results.items(),
            key=lambda x: x[1]['aggregated_metrics'].get(metric, 0)
        )

        return best_method[0]


if __name__ == "__main__":
    # Example usage
    from .data import AgiBotWorldDataset
    from .methods import (
        SpectralClusteringMethod,
        ChangePointDetectionMethod,
        GMMMethod,
        HierarchicalClusteringMethod
    )

    # Load dataset
    dataset = AgiBotWorldDataset(
        data_root=Path("./data/AgiBotWorld-Alpha"),
        max_trajectories=10
    )

    # Define methods to benchmark
    methods = [
        SpectralClusteringMethod(n_clusters=5, temporal_weight=0.3),
        ChangePointDetectionMethod(penalty=3.0, max_segments=5),
        GMMMethod(n_components=5),
        HierarchicalClusteringMethod(n_clusters=5)
    ]

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        methods=methods,
        output_dir=Path("./benchmark_results")
    )

    # Run benchmark
    results = runner.run_all_methods(save_predictions=True)

    # Get best method
    best_method = runner.get_best_method(metric='macro_f1')
    print(f"\nBest method (by F1): {best_method}")
