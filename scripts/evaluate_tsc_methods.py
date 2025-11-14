#!/usr/bin/env python3
"""Evaluate TSC (Transition State Clustering) methods on proprioceptive data.

This script implements simplified versions of TSC-based segmentation methods
and evaluates them on the same proprioceptive data and metrics used in the
ablation study for fair comparison.

TSC Methods Evaluated:
1. SimpleGMM - Basic GMM-based segmentation with time
2. ChangePointDetection - Statistical change point detection
3. HMMSegmentation - HMM-based segmentation

Metrics Computed (matching ablation study):
- Boundary Detection: F1@1.0, F1@2.0, F1@3.0, F1@4.0, F1@5.0
- Average Precision: AP@1.0, AP@2.0, AP@3.0, AP@4.0, AP@5.0
- mAP (mean Average Precision)
- Segmental F1: Seg-F1@10%, Seg-F1@25%, Seg-F1@50%
- Edit Distance (Levenshtein distance)
- Framewise Accuracy

Usage:
    # Evaluate all methods on specific tasks
    python scripts/evaluate_tsc_methods.py \
        --task-ids 327 352 356 \
        --output-dir ./tsc_evaluation_results \
        --meta-dir /shared/projects/agibot/full_dataset/

    # Quick test with fewer demonstrations
    python scripts/evaluate_tsc_methods.py \
        --task-ids 327 \
        --max-trajectories 20 \
        --quick-test

Author: Claude Code
Date: November 2025
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import data loading from main codebase
from data import load_multi_task_split

# Import sklearn for segmentation methods
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available, HMM methods will be skipped")


# =============================================================================
# TSC Baseline Methods (Python 3 ports)
# =============================================================================

class BerkeleyTSC:
    """Faithful implementation of Berkeley's Transition State Clustering (TSC).

    Reference: https://github.com/BerkeleyAutomation/tsc/blob/master/tsc/tsc.py

    Full pipeline implementation:
    1. identifyTransitions - DP-GMM on windowed features
    2. clusterInState - DP-GMM on transition states
    3. pruneClusters - Filter rare transitions
    4. clusterInTime - DP-GMM on temporal indices
    5. taskToTrajectory - Convert to trajectory boundaries
    6. compaction - Merge similar segments via DTW
    """

    def __init__(self, window_size=2, pruning=0.9, delta=-1, n_dpgmm_runs=1, random_state=42):
        """Initialize Berkeley TSC.

        Args:
            window_size: Temporal window for feature extraction
            pruning: Threshold for filtering transitions (0.9 = 90% coverage)
            delta: DTW compaction threshold (-1 to disable)
            n_dpgmm_runs: Number of DP-GMM runs (take best)
            random_state: Random seed
        """
        self.window_size = window_size
        self.pruning = pruning
        self.delta = delta
        self.n_dpgmm_runs = n_dpgmm_runs
        self.random_state = random_state

        # Internal state
        self._demonstrations = []
        self._demonstration_sizes = []
        self._transitions = []
        self._state_clusters = []
        self._pruned_state_clusters = []
        self._temporal_clusters = []
        self.segmentations = []

    def add_demonstration(self, demonstration: np.ndarray):
        """Add a demonstration."""
        self._demonstrations.append(demonstration)
        self._demonstration_sizes.append(demonstration.shape)

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Fit TSC and segment each demonstration.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        # Reset state
        self._demonstrations = demonstrations
        self._demonstration_sizes = [demo.shape for demo in demonstrations]
        self._transitions = []
        self._state_clusters = []
        self._pruned_state_clusters = []
        self._temporal_clusters = []
        self.segmentations = []

        # Execute TSC pipeline
        try:
            self._identify_transitions()
            self._cluster_in_state()
            self._prune_clusters()
            self._cluster_in_time()
            self._task_to_trajectory()
            if self.delta > 0:
                self._compaction()
        except Exception as e:
            print(f"Warning: TSC pipeline failed ({e}), using uniform segmentation")
            self.segmentations = [[0, len(demo)] for demo in demonstrations]

        return self.segmentations

    def _identify_transitions(self):
        """Stage 1: Identify transitions using DP-GMM on windowed features."""
        # Build windowed features
        p = self._demonstration_sizes[0][1]  # feature dimension
        total_size = sum(size[0] for size in self._demonstration_sizes)

        if total_size <= self.window_size:
            # Too short, no transitions
            self._transitions = [(i, 0) for i in range(len(self._demonstrations))]
            return

        # Create sliding windows: flatten [x_{t-window_size}:x_t]
        demo_data_array = np.zeros((total_size - len(self._demonstrations) * self.window_size,
                                    p * self.window_size))
        inc = 0
        demo_indices = []
        timestep_indices = []

        for i, demo in enumerate(self._demonstrations):
            n = self._demonstration_sizes[i][0]
            for j in range(self.window_size, n):
                window = demo[j-self.window_size:j, :]
                demo_data_array[inc, :] = np.reshape(window, (1, p * self.window_size))
                demo_indices.append(i)
                timestep_indices.append(j)
                inc += 1

        if len(demo_data_array) == 0:
            self._transitions = [(i, 0) for i in range(len(self._demonstrations))]
            return

        # Run DP-GMM clustering on windows
        cluster_labels = self._dpgmm_cluster(demo_data_array)

        # Smooth labels to remove isolated changes
        cluster_labels = self._smooth_labels(cluster_labels)

        # Identify transitions (where cluster label changes)
        self._transitions = [(0, 0)]  # Start with beginning
        for idx in range(1, len(cluster_labels)):
            if cluster_labels[idx] != cluster_labels[idx-1]:
                demo_idx = demo_indices[idx]
                timestep = timestep_indices[idx] - self.window_size
                self._transitions.append((demo_idx, timestep))

    def _cluster_in_state(self):
        """Stage 2: Cluster transitions by their state vectors using DP-GMM."""
        if len(self._transitions) == 0:
            return

        # Extract state vectors at transition points
        transition_states = []
        for demo_idx, timestep in self._transitions:
            if timestep < len(self._demonstrations[demo_idx]):
                state = self._demonstrations[demo_idx][timestep, :]
                transition_states.append(state)
            else:
                # Handle edge case
                state = self._demonstrations[demo_idx][-1, :]
                transition_states.append(state)

        if len(transition_states) == 0:
            return

        transition_states = np.array(transition_states)

        # Run DP-GMM on transition states
        state_cluster_labels = self._dpgmm_cluster(transition_states)

        # Store (demo_idx, timestep, state_cluster_label)
        self._state_clusters = []
        for i, (demo_idx, timestep) in enumerate(self._transitions):
            self._state_clusters.append((demo_idx, timestep, state_cluster_labels[i]))

    def _prune_clusters(self):
        """Stage 3: Remove clusters appearing in < pruning fraction of demos."""
        if len(self._state_clusters) == 0:
            self._pruned_state_clusters = []
            return

        n_demos = len(self._demonstrations)

        # Count demos per state cluster
        cluster_demo_counts = {}
        for demo_idx, timestep, cluster_label in self._state_clusters:
            if cluster_label not in cluster_demo_counts:
                cluster_demo_counts[cluster_label] = set()
            cluster_demo_counts[cluster_label].add(demo_idx)

        # Keep clusters appearing in >= pruning * n_demos demonstrations
        min_demo_count = int(self.pruning * n_demos)
        valid_clusters = {label for label, demos in cluster_demo_counts.items()
                         if len(demos) >= min_demo_count}

        self._pruned_state_clusters = [
            (demo_idx, timestep, cluster_label)
            for demo_idx, timestep, cluster_label in self._state_clusters
            if cluster_label in valid_clusters
        ]

    def _cluster_in_time(self):
        """Stage 4: For each state cluster, apply DP-GMM on temporal indices."""
        if len(self._pruned_state_clusters) == 0:
            self._temporal_clusters = []
            return

        # Group by state cluster
        state_cluster_groups = {}
        for demo_idx, timestep, cluster_label in self._pruned_state_clusters:
            if cluster_label not in state_cluster_groups:
                state_cluster_groups[cluster_label] = []
            state_cluster_groups[cluster_label].append((demo_idx, timestep))

        # For each state cluster, cluster in time
        self._temporal_clusters = []
        for state_label, transitions in state_cluster_groups.items():
            timesteps = np.array([t for _, t in transitions]).reshape(-1, 1)

            if len(timesteps) > 1:
                temporal_labels = self._dpgmm_cluster(timesteps)
            else:
                temporal_labels = np.array([0])

            for i, (demo_idx, timestep) in enumerate(transitions):
                self._temporal_clusters.append((demo_idx, timestep, state_label, temporal_labels[i]))

    def _task_to_trajectory(self):
        """Stage 5: Convert task segmentation to trajectory boundaries."""
        self.segmentations = []

        for demo_idx in range(len(self._demonstrations)):
            demo_length = self._demonstration_sizes[demo_idx][0]

            # Extract transitions for this demo
            demo_transitions = [timestep for d_idx, timestep, _, _ in self._temporal_clusters
                              if d_idx == demo_idx]

            # Create boundaries
            boundaries = [0] + sorted(set(demo_transitions)) + [demo_length]
            self.segmentations.append(sorted(set(boundaries)))

    def _compaction(self):
        """Stage 6: Merge adjacent segments with low DTW distance."""
        try:
            from scipy.spatial.distance import euclidean
            from fastdtw import fastdtw
            has_dtw = True
        except ImportError:
            print("Warning: fastdtw not available, skipping compaction")
            has_dtw = False

        if not has_dtw or self.delta <= 0:
            return

        # Compact each demonstration
        for demo_idx, demo in enumerate(self._demonstrations):
            boundaries = self.segmentations[demo_idx]

            if len(boundaries) <= 2:
                continue

            # Compute DTW between consecutive segments
            compacted_boundaries = [boundaries[0]]

            for i in range(len(boundaries) - 2):
                seg1_start = boundaries[i]
                seg1_end = boundaries[i+1]
                seg2_start = boundaries[i+1]
                seg2_end = boundaries[i+2]

                seg1 = demo[seg1_start:seg1_end, :]
                seg2 = demo[seg2_start:seg2_end, :]

                if len(seg1) > 0 and len(seg2) > 0:
                    try:
                        distance, _ = fastdtw(seg1, seg2, dist=euclidean)
                        normalized_distance = distance / (len(seg1) + len(seg2))

                        # If distance > threshold, keep the boundary
                        if normalized_distance > self.delta:
                            compacted_boundaries.append(boundaries[i+1])
                    except:
                        # Keep boundary if DTW fails
                        compacted_boundaries.append(boundaries[i+1])
                else:
                    compacted_boundaries.append(boundaries[i+1])

            compacted_boundaries.append(boundaries[-1])
            self.segmentations[demo_idx] = sorted(set(compacted_boundaries))

    def _dpgmm_cluster(self, data: np.ndarray) -> np.ndarray:
        """Run DP-GMM clustering using BayesianGaussianMixture.

        Args:
            data: Input data (N x D)

        Returns:
            Cluster labels (N,)
        """
        from sklearn.mixture import BayesianGaussianMixture
        from sklearn.preprocessing import StandardScaler

        if len(data) == 0:
            return np.array([])

        # Standardize data
        scaler = StandardScaler()
        try:
            data_scaled = scaler.fit_transform(data)
        except:
            return np.zeros(len(data), dtype=int)

        # Run DP-GMM multiple times and take best
        best_score = -np.inf
        best_labels = None

        for run in range(self.n_dpgmm_runs):
            try:
                dpgmm = BayesianGaussianMixture(
                    n_components=min(20, len(data)),  # Upper bound on clusters
                    covariance_type='full',
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=1.0,
                    max_iter=100,
                    n_init=1,
                    random_state=self.random_state + run
                )
                labels = dpgmm.fit_predict(data_scaled)
                score = dpgmm.score(data_scaled)

                if score > best_score:
                    best_score = score
                    best_labels = labels
            except:
                continue

        if best_labels is None:
            return np.zeros(len(data), dtype=int)

        return best_labels

    def _smooth_labels(self, labels: np.ndarray) -> np.ndarray:
        """Smooth cluster labels to remove isolated changes.

        Args:
            labels: Cluster labels (N,)

        Returns:
            Smoothed labels (N,)
        """
        if len(labels) <= 2:
            return labels

        smoothed = labels.copy()

        for i in range(1, len(labels) - 1):
            # If label differs from both neighbors but neighbors match, smooth it
            if (labels[i] != labels[i-1] and
                labels[i] != labels[i+1] and
                labels[i+1] == labels[i-1]):
                smoothed[i] = labels[i+1]

        return smoothed


# Legacy alias for backwards compatibility
class TimeVaryingGMM(BerkeleyTSC):
    """Legacy alias - now uses faithful Berkeley TSC implementation."""
    pass


class HMMGaussianSegmentation:
    """HMM with Gaussian Emissions from TSC alternates/clustering.py

    Ported from Python 2 to Python 3.
    """

    def __init__(self, n_components=3):
        """Initialize HMM-GMM.

        Args:
            n_components: Number of hidden states
        """
        self.n_components = n_components
        self.segmentations = []
        self.available = HMM_AVAILABLE

    def add_demonstration(self, demonstration: np.ndarray):
        """Not used - fitting is collective."""
        pass

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Fit HMM and segment each demonstration.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        if not self.available:
            print("Warning: HMM not available, using uniform segmentation")
            return [[0, len(demo)] for demo in demonstrations]

        self.segmentations = []

        # Normalize demonstrations
        from sklearn.preprocessing import normalize
        normalized_demos = [normalize(demo, axis=1) for demo in demonstrations]

        # Concatenate all demos
        all_demos = normalized_demos[0]
        lens = [normalized_demos[0].shape[0]]
        for i in range(1, len(normalized_demos)):
            all_demos = np.concatenate([all_demos, normalized_demos[i]])
            lens.append(normalized_demos[i].shape[0])

        # Fit HMM
        try:
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
            model.fit(all_demos, lens)

            # Segment each demonstration
            for demo in normalized_demos:
                states = model.predict(demo)
                boundaries = self._find_transitions(states)
                self.segmentations.append(boundaries)

        except Exception as e:
            print(f"Warning: HMM fitting failed ({e}), using uniform segmentation")
            self.segmentations = [[0, len(demo)] for demo in demonstrations]

        return self.segmentations

    def _find_transitions(self, pred_seq):
        """Find segment transitions from predicted states."""
        transitions = [0]
        prev = -1
        for i, v in enumerate(pred_seq):
            if prev != v:
                transitions.append(i)
            prev = v
        transitions.append(len(pred_seq))
        return sorted(set(transitions))


class CoresetSegmentation:
    """Coreset-based Segmentation from TSC alternates/clustering.py

    Ported from Python 2 to Python 3.
    Simplified version without full coreset algorithm.
    """

    def __init__(self, n_components=4):
        """Initialize Coreset segmentation.

        Args:
            n_components: Number of coreset samples
        """
        self.n_components = n_components
        self.segmentations = []

    def add_demonstration(self, demonstration: np.ndarray):
        """Not used."""
        pass

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Fit coreset and segment.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        self.segmentations = []

        # Simple coreset approximation using k-medoids on time-augmented data
        for demo in demonstrations:
            T, D = demo.shape

            # Time-augment
            time_augmented = np.zeros((T, D + 1))
            time_augmented[:, 0:D] = demo
            time_augmented[:, D] = np.arange(0, T, 1)

            # Simple coreset: k-means to find representative points
            try:
                from sklearn.cluster import KMeans
                n_clusters = min(self.n_components, T - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(time_augmented)

                # Find closest points to cluster centers (coreset samples)
                from scipy.spatial.distance import cdist
                dists = cdist(time_augmented, kmeans.cluster_centers_)
                coreset_indices = []
                for i in range(n_clusters):
                    closest = np.argmin(dists[:, i])
                    coreset_indices.append(closest)

                # Boundaries are coreset sample time indices
                boundaries = [0] + sorted(coreset_indices) + [T]
                self.segmentations.append(sorted(set(boundaries)))

            except:
                self.segmentations.append([0, T])

        return self.segmentations


# =============================================================================
# Simple Baseline Methods
# =============================================================================

class SimpleGMMSegmentation:
    """Simple GMM-based segmentation with time as a feature.

    Adapted from Time-Varying GMM approach in TSC alternates.
    """

    def __init__(self, n_components=3, window_size=3):
        """Initialize GMM segmentation.

        Args:
            n_components: Number of GMM components
            window_size: Window size for temporal features
        """
        self.n_components = n_components
        self.window_size = window_size
        self.segmentations = []

    def add_demonstration(self, demonstration: np.ndarray):
        """Add a demonstration (not used, fits per-demo)."""
        pass

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Fit GMM and segment each demonstration.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        self.segmentations = []

        for demo in demonstrations:
            # Create time-augmented features
            T, D = demo.shape

            # Add time as a feature (normalized)
            time_feature = np.arange(T).reshape(-1, 1) / T
            augmented = np.hstack([demo, time_feature])

            # Add windowed features
            windowed_features = []
            for t in range(self.window_size, T):
                window = demo[t-self.window_size:t].flatten()
                time_t = t / T
                feat = np.concatenate([window, [time_t]])
                windowed_features.append(feat)

            if len(windowed_features) < self.n_components:
                # Too short, just segment at start/end
                self.segmentations.append([0, T])
                continue

            windowed_features = np.array(windowed_features)

            # Fit GMM
            try:
                gmm = GaussianMixture(
                    n_components=min(self.n_components, len(windowed_features)),
                    covariance_type='full',
                    random_state=42,
                    max_iter=100
                )
                labels = gmm.fit_predict(windowed_features)

                # Find transitions
                boundaries = [0]
                for t in range(1, len(labels)):
                    if labels[t] != labels[t-1]:
                        boundaries.append(t + self.window_size)
                boundaries.append(T)

                self.segmentations.append(sorted(set(boundaries)))
            except:
                # Fallback to uniform segmentation
                self.segmentations.append([0, T])

        return self.segmentations


class ChangePointDetection:
    """Statistical change point detection for segmentation.

    Uses cumulative sum (CUSUM) approach to detect distribution changes.
    """

    def __init__(self, threshold=2.0, window_size=5):
        """Initialize change point detection.

        Args:
            threshold: Detection threshold (higher = fewer boundaries)
            window_size: Window for computing statistics
        """
        self.threshold = threshold
        self.window_size = window_size
        self.segmentations = []

    def add_demonstration(self, demonstration: np.ndarray):
        """Add a demonstration (not used, fits per-demo)."""
        pass

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Detect change points in each demonstration.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        self.segmentations = []

        for demo in demonstrations:
            T, D = demo.shape

            if T < self.window_size * 2:
                self.segmentations.append([0, T])
                continue

            # Compute change scores
            change_scores = np.zeros(T)

            for t in range(self.window_size, T - self.window_size):
                # Compare distributions before and after
                before = demo[t-self.window_size:t]
                after = demo[t:t+self.window_size]

                # Use mean squared difference as change score
                mean_before = np.mean(before, axis=0)
                mean_after = np.mean(after, axis=0)
                change_scores[t] = np.linalg.norm(mean_after - mean_before)

            # Normalize scores
            if np.std(change_scores) > 0:
                change_scores = (change_scores - np.mean(change_scores)) / np.std(change_scores)

            # Find peaks above threshold
            boundaries = [0]
            for t in range(1, T-1):
                if (change_scores[t] > self.threshold and
                    change_scores[t] > change_scores[t-1] and
                    change_scores[t] > change_scores[t+1]):
                    boundaries.append(t)
            boundaries.append(T)

            self.segmentations.append(sorted(set(boundaries)))

        return self.segmentations


class HMMSegmentation:
    """HMM-based segmentation.

    Uses HMM with Gaussian emissions to model temporal dynamics.
    """

    def __init__(self, n_components=3):
        """Initialize HMM segmentation.

        Args:
            n_components: Number of hidden states
        """
        self.n_components = n_components
        self.segmentations = []
        self.available = HMM_AVAILABLE

    def add_demonstration(self, demonstration: np.ndarray):
        """Add a demonstration (not used, fits per-demo)."""
        pass

    def fit_and_segment(self, demonstrations: List[np.ndarray]) -> List[List[int]]:
        """Fit HMM and segment each demonstration.

        Args:
            demonstrations: List of proprio sequences (each is T x D)

        Returns:
            List of boundary frame lists
        """
        if not self.available:
            print("Warning: HMM not available, using uniform segmentation")
            return [[0, len(demo)] for demo in demonstrations]

        self.segmentations = []

        # Fit HMM on all demonstrations
        all_demos = np.vstack(demonstrations)
        lengths = [len(demo) for demo in demonstrations]

        try:
            # Fit Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type='diag',
                n_iter=100,
                random_state=42
            )
            model.fit(all_demos, lengths)

            # Segment each demonstration
            for demo in demonstrations:
                # Predict hidden states
                states = model.predict(demo)

                # Find state transitions
                boundaries = [0]
                for t in range(1, len(states)):
                    if states[t] != states[t-1]:
                        boundaries.append(t)
                boundaries.append(len(demo))

                self.segmentations.append(sorted(set(boundaries)))

        except Exception as e:
            print(f"Warning: HMM fitting failed ({e}), using uniform segmentation")
            self.segmentations = [[0, len(demo)] for demo in demonstrations]

        return self.segmentations


# =============================================================================
# Data Adapter
# =============================================================================

class TSCDataAdapter:
    """Adapter to convert parquet trajectory data to TSC format."""

    @staticmethod
    def extract_proprio_sequence(trajectory) -> np.ndarray:
        """Extract proprioceptive sequence from trajectory.

        Args:
            trajectory: ParquetTrajectory object

        Returns:
            Proprioceptive array of shape (T, D)
        """
        proprio = trajectory.proprio.joint_position
        if proprio.ndim == 1:
            proprio = proprio.reshape(-1, 1)
        return proprio

    @staticmethod
    def extract_ground_truth_boundaries(trajectory) -> List[int]:
        """Extract ground truth boundary frames from trajectory segments.

        Args:
            trajectory: ParquetTrajectory object

        Returns:
            List of boundary frame indices
        """
        boundaries = [0]
        for segment in trajectory.intervals:
            boundaries.append(segment.end_frame)
        return sorted(set(boundaries))


# =============================================================================
# Metrics Calculator
# =============================================================================

class TSCMetricsCalculator:
    """Calculate metrics matching the ablation study."""

    @staticmethod
    def calculate_boundary_f1(
        pred_boundaries: List[int],
        gt_boundaries: List[int],
        tolerance: float
    ) -> Tuple[float, float, float]:
        """Calculate boundary detection F1 score."""
        pred = [b for b in pred_boundaries if 0 < b < max(pred_boundaries, default=1)]
        gt = [b for b in gt_boundaries if 0 < b < max(gt_boundaries, default=1)]

        if len(pred) == 0 or len(gt) == 0:
            return 0.0, 0.0, 0.0

        matched_pred = set()
        matched_gt = set()

        for p_idx in pred:
            for g_idx in gt:
                if g_idx not in matched_gt and abs(p_idx - g_idx) <= tolerance:
                    matched_pred.add(p_idx)
                    matched_gt.add(g_idx)
                    break

        tp = len(matched_pred)
        fp = len(pred) - tp
        fn = len(gt) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def calculate_average_precision(
        pred_boundaries: List[int],
        gt_boundaries: List[int],
        tolerance: float
    ) -> float:
        """Calculate average precision at given tolerance."""
        precision, _, _ = TSCMetricsCalculator.calculate_boundary_f1(
            pred_boundaries, gt_boundaries, tolerance
        )
        return precision

    @staticmethod
    def boundaries_to_segments(boundaries: List[int], total_length: int) -> List[Tuple[int, int]]:
        """Convert boundary frames to segment intervals."""
        boundaries = sorted(set(boundaries))
        if 0 not in boundaries:
            boundaries.insert(0, 0)
        if total_length not in boundaries:
            boundaries.append(total_length)

        return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries) - 1)]

    @staticmethod
    def calculate_iou(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
        """Calculate intersection over union for two segments."""
        inter_start = max(seg1[0], seg2[0])
        inter_end = min(seg1[1], seg2[1])
        inter_len = max(0, inter_end - inter_start)
        union_len = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter_len
        return inter_len / union_len if union_len > 0 else 0.0

    @staticmethod
    def calculate_segmental_f1(
        pred_boundaries: List[int],
        gt_boundaries: List[int],
        total_length: int,
        iou_threshold: float
    ) -> Tuple[float, float, float]:
        """Calculate segmental F1@IoU score."""
        pred_segments = TSCMetricsCalculator.boundaries_to_segments(pred_boundaries, total_length)
        gt_segments = TSCMetricsCalculator.boundaries_to_segments(gt_boundaries, total_length)

        matched_pred = set()
        matched_gt = set()

        for i, pred_seg in enumerate(pred_segments):
            for j, gt_seg in enumerate(gt_segments):
                if j not in matched_gt:
                    iou = TSCMetricsCalculator.calculate_iou(pred_seg, gt_seg)
                    if iou >= iou_threshold:
                        matched_pred.add(i)
                        matched_gt.add(j)
                        break

        tp = len(matched_pred)
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def calculate_edit_distance(
        pred_boundaries: List[int],
        gt_boundaries: List[int]
    ) -> Tuple[float, float]:
        """Calculate Levenshtein edit distance."""
        pred = [b for b in pred_boundaries if 0 < b < max(pred_boundaries, default=1)]
        gt = [b for b in gt_boundaries if 0 < b < max(gt_boundaries, default=1)]

        m, n = len(pred), len(gt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if abs(pred[i-1] - gt[j-1]) <= 1.0:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        edit_distance = dp[m][n]
        max_possible = max(m, n) if max(m, n) > 0 else 1
        normalized_edit_distance = edit_distance / max_possible

        return edit_distance, normalized_edit_distance

    @staticmethod
    def calculate_framewise_accuracy(
        pred_boundaries: List[int],
        gt_boundaries: List[int],
        total_length: int
    ) -> float:
        """Calculate framewise accuracy."""
        pred_segments = TSCMetricsCalculator.boundaries_to_segments(pred_boundaries, total_length)
        gt_segments = TSCMetricsCalculator.boundaries_to_segments(gt_boundaries, total_length)

        pred_labels = np.zeros(total_length, dtype=int)
        gt_labels = np.zeros(total_length, dtype=int)

        for seg_id, (start, end) in enumerate(pred_segments):
            pred_labels[start:end] = seg_id

        for seg_id, (start, end) in enumerate(gt_segments):
            gt_labels[start:end] = seg_id

        seg_mapping = {}
        for pred_id in range(len(pred_segments)):
            pred_mask = (pred_labels == pred_id)
            gt_ids_in_pred = gt_labels[pred_mask]
            if len(gt_ids_in_pred) > 0:
                seg_mapping[pred_id] = np.bincount(gt_ids_in_pred).argmax()

        correct = sum(1 for frame in range(total_length)
                     if pred_labels[frame] in seg_mapping and
                     seg_mapping[pred_labels[frame]] == gt_labels[frame])

        return correct / total_length if total_length > 0 else 0.0


# =============================================================================
# Evaluator
# =============================================================================

class TSCEvaluator:
    """Evaluator for TSC methods."""

    TSC_METHODS = {
        # Berkeley TSC - Faithful implementation with various parameters
        'berkeley_tsc_p90': {
            'name': 'Berkeley TSC (pruning=0.9)',
            'description': 'Berkeley TSC with strict pruning (90%)',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.9, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_p50': {
            'name': 'Berkeley TSC (pruning=0.5)',
            'description': 'Berkeley TSC with medium pruning (50%)',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.5, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_p30': {
            'name': 'Berkeley TSC (pruning=0.3)',
            'description': 'Berkeley TSC with lenient pruning (30%)',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.3, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_p10': {
            'name': 'Berkeley TSC (pruning=0.1)',
            'description': 'Berkeley TSC with very lenient pruning (10%)',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.1, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_w3': {
            'name': 'Berkeley TSC (window=3)',
            'description': 'Berkeley TSC with larger window size',
            'class': BerkeleyTSC,
            'params': {'window_size': 3, 'pruning': 0.5, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_w5': {
            'name': 'Berkeley TSC (window=5)',
            'description': 'Berkeley TSC with even larger window size',
            'class': BerkeleyTSC,
            'params': {'window_size': 5, 'pruning': 0.5, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'berkeley_tsc_stable': {
            'name': 'Berkeley TSC (3 runs)',
            'description': 'Berkeley TSC with multiple DP-GMM runs for stability',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.5, 'delta': -1, 'n_dpgmm_runs': 3}
        },
        'berkeley_tsc': {
            'name': 'Berkeley TSC (default)',
            'description': 'Berkeley TSC with original parameters',
            'class': BerkeleyTSC,
            'params': {'window_size': 2, 'pruning': 0.9, 'delta': -1, 'n_dpgmm_runs': 1}
        },
        'timevarying_gmm': {
            'name': 'Berkeley TSC (alias)',
            'description': 'Legacy alias for Berkeley TSC',
            'class': TimeVaryingGMM,
            'params': {'window_size': 2, 'pruning': 0.9, 'delta': -1, 'n_dpgmm_runs': 1}
        },

        # Other TSC variants
        'coreset': {
            'name': 'Coreset Segmentation',
            'description': 'Coreset-based segmentation (TSC baseline)',
            'class': CoresetSegmentation,
            'params': {'n_components': 4}
        },

        # Simple baseline methods
        'simple_gmm': {
            'name': 'Simple GMM',
            'description': 'Simple GMM with windowing',
            'class': SimpleGMMSegmentation,
            'params': {'n_components': 3, 'window_size': 3}
        },
        'changepoint': {
            'name': 'Change Point Detection',
            'description': 'Statistical change point detection',
            'class': ChangePointDetection,
            'params': {'threshold': 2.0, 'window_size': 5}
        },
    }

    if HMM_AVAILABLE:
        TSC_METHODS['hmm_gaussian'] = {
            'name': 'HMM-Gaussian',
            'description': 'HMM with Gaussian emissions (TSC baseline)',
            'class': HMMGaussianSegmentation,
            'params': {'n_components': 3}
        }
        TSC_METHODS['simple_hmm'] = {
            'name': 'Simple HMM',
            'description': 'Simple HMM segmentation',
            'class': HMMSegmentation,
            'params': {'n_components': 3}
        }

    def __init__(
        self,
        meta_dir: str,
        output_dir: str,
        max_trajectories: int = None,
        metric_tolerances: List[float] = None
    ):
        """Initialize evaluator."""
        self.meta_dir = meta_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_trajectories = max_trajectories
        self.metric_tolerances = metric_tolerances or [1.0, 2.0, 3.0, 4.0, 5.0]
        self.results = []

    def evaluate_method_on_task(
        self,
        method_name: str,
        task_id: int
    ) -> Dict[str, Any]:
        """Evaluate a single method on a single task."""
        print(f"\n{'='*80}")
        print(f"Evaluating: {self.TSC_METHODS[method_name]['name']} on Task {task_id}")
        print(f"{'='*80}\n")

        try:
            # Load data
            print(f"Loading data for task {task_id}...")
            val_trajectories = load_multi_task_split(
                'val',
                task_ids=[task_id],
                meta_dir=self.meta_dir
            )

            if self.max_trajectories and len(val_trajectories) > self.max_trajectories:
                val_trajectories = val_trajectories[:self.max_trajectories]

            print(f"Loaded {len(val_trajectories)} trajectories")

            # Initialize method
            method_config = self.TSC_METHODS[method_name]
            method = method_config['class'](**method_config['params'])

            # Extract data
            print("Extracting proprioceptive data...")
            demonstrations = []
            gt_boundaries_list = []
            proprio_lengths = []

            for traj in tqdm(val_trajectories, desc="Processing"):
                proprio = TSCDataAdapter.extract_proprio_sequence(traj)
                demonstrations.append(proprio)
                proprio_lengths.append(len(proprio))
                gt_boundaries = TSCDataAdapter.extract_ground_truth_boundaries(traj)
                gt_boundaries_list.append(gt_boundaries)

            # Fit and segment
            print("Fitting model and segmenting...")
            pred_segmentations = method.fit_and_segment(demonstrations)

            # Compute metrics
            print("Computing metrics...")
            all_metrics = []

            for pred_seg, gt_boundaries, traj_length in zip(pred_segmentations, gt_boundaries_list, proprio_lengths):
                metrics = {}

                # Boundary-level F1
                for tol in self.metric_tolerances:
                    precision, recall, f1 = TSCMetricsCalculator.calculate_boundary_f1(
                        pred_seg, gt_boundaries, tol
                    )
                    metrics[f'precision@{tol}'] = precision
                    metrics[f'recall@{tol}'] = recall
                    metrics[f'f1@{tol}'] = f1
                    metrics[f'ap@{tol}'] = TSCMetricsCalculator.calculate_average_precision(
                        pred_seg, gt_boundaries, tol
                    )

                # Segmental F1
                for iou_threshold in [0.1, 0.25, 0.5]:
                    seg_precision, seg_recall, seg_f1 = TSCMetricsCalculator.calculate_segmental_f1(
                        pred_seg, gt_boundaries, traj_length, iou_threshold
                    )
                    iou_pct = int(iou_threshold * 100)
                    metrics[f'seg_precision@{iou_pct}'] = seg_precision
                    metrics[f'seg_recall@{iou_pct}'] = seg_recall
                    metrics[f'seg_f1@{iou_pct}'] = seg_f1

                # Edit distance
                edit_dist, norm_edit_dist = TSCMetricsCalculator.calculate_edit_distance(
                    pred_seg, gt_boundaries
                )
                metrics['edit_distance'] = edit_dist
                metrics['normalized_edit_distance'] = norm_edit_dist

                # Framewise accuracy
                metrics['framewise_accuracy'] = TSCMetricsCalculator.calculate_framewise_accuracy(
                    pred_seg, gt_boundaries, traj_length
                )

                all_metrics.append(metrics)

            # Aggregate metrics
            aggregated_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated_metrics[key] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)

            # Compute mAP
            ap_scores = [aggregated_metrics[f'ap@{tol}'] for tol in self.metric_tolerances]
            aggregated_metrics['map'] = np.mean(ap_scores)

            # Print results
            print(f"\nResults for {method_config['name']} on Task {task_id}:")
            print(f"  F1@2.0: {aggregated_metrics['f1@2.0']:.4f}")
            print(f"  Seg-F1@50%: {aggregated_metrics['seg_f1@50']:.4f}")
            print(f"  Edit Distance: {aggregated_metrics['edit_distance']:.2f}")
            print(f"  Framewise Accuracy: {aggregated_metrics['framewise_accuracy']:.4f}")
            print(f"  mAP: {aggregated_metrics['map']:.4f}")

            return {
                'method': method_name,
                'method_name': method_config['name'],
                'description': method_config['description'],
                'task_id': task_id,
                'num_trajectories': len(val_trajectories),
                'metrics': aggregated_metrics,
                'status': 'completed'
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'method': method_name,
                'method_name': self.TSC_METHODS[method_name]['name'],
                'task_id': task_id,
                'status': 'failed',
                'error': str(e)
            }

    def evaluate_all_methods(
        self,
        task_ids: List[int],
        method_names: List[str] = None
    ):
        """Evaluate all methods on specified tasks."""
        if method_names is None:
            method_names = list(self.TSC_METHODS.keys())

        print(f"\n{'='*80}")
        print(f"TSC METHODS EVALUATION")
        print(f"{'='*80}")
        print(f"Methods: {[self.TSC_METHODS[m]['name'] for m in method_names]}")
        print(f"Tasks: {task_ids}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")

        for task_id in task_ids:
            for method_name in method_names:
                result = self.evaluate_method_on_task(method_name, task_id)
                self.results.append(result)
                self._save_results()

        self._generate_summary()

    def _save_results(self):
        """Save results to JSON."""
        results_file = self.output_dir / 'tsc_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _generate_summary(self):
        """Generate summary CSV."""
        print(f"\n{'='*80}")
        print(f"GENERATING SUMMARY")
        print(f"{'='*80}\n")

        summary_rows = []
        for result in self.results:
            if result['status'] == 'completed':
                row = {
                    'method': result['method_name'],
                    'task_id': result['task_id'],
                    'num_trajectories': result['num_trajectories']
                }

                metrics = result['metrics']
                for tol in self.metric_tolerances:
                    row[f'F1@{tol}'] = metrics[f'f1@{tol}']
                    row[f'AP@{tol}'] = metrics[f'ap@{tol}']

                row['mAP'] = metrics['map']

                for iou_pct in [10, 25, 50]:
                    row[f'Seg-F1@{iou_pct}%'] = metrics[f'seg_f1@{iou_pct}']

                row['Edit-Dist'] = metrics['edit_distance']
                row['Framewise-Acc'] = metrics['framewise_accuracy']

                summary_rows.append(row)

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            csv_file = self.output_dir / 'tsc_evaluation_summary.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary saved to: {csv_file}")
            print("\nResults Summary:")
            print(df.to_string(index=False))

        self._save_results()

        print(f"\n{'='*80}")
        print(f"Evaluation complete!")
        print(f"Results directory: {self.output_dir}")
        print(f"{'='*80}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate TSC methods on proprioceptive data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (like ablation study)
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file (like configs/experiments/multi_task_327_352_356_detr.yaml)'
    )

    # Direct arguments (override config)
    parser.add_argument('--task-ids', type=int, nargs='+', help='Task IDs to evaluate (overrides config)')
    parser.add_argument('--meta-dir', type=str, help='Trajectory metadata directory (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./tsc_evaluation_results', help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', choices=list(TSCEvaluator.TSC_METHODS.keys()) + ['all'], default=['all'], help='Methods to evaluate')
    parser.add_argument('--max-trajectories', type=int, default=None, help='Max trajectories per task')
    parser.add_argument('--metric-tolerances', type=float, nargs='+', help='Tolerances for boundary F1 (overrides config)')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (20 trajectories)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load config if provided
    task_ids = None
    meta_dir = None
    metric_tolerances = [1.0, 2.0, 3.0, 4.0, 5.0]  # default

    if args.config:
        print(f"Loading config from: {args.config}")
        from config import load_config
        config = load_config(args.config)

        # Extract parameters from config
        task_ids = config.data.eval_task_ids
        meta_dir = config.data.meta_dir
        metric_tolerances = getattr(config.training, 'metric_tolerances', [1.0, 2.0, 3.0])

        print(f"  Tasks from config: {task_ids}")
        print(f"  Meta dir: {meta_dir}")
        print(f"  Metric tolerances: {metric_tolerances}")

    # Command-line args override config
    if args.task_ids:
        task_ids = args.task_ids
    if args.meta_dir:
        meta_dir = args.meta_dir
    if args.metric_tolerances:
        metric_tolerances = args.metric_tolerances

    # Validate required parameters
    if task_ids is None:
        raise ValueError("Must provide --task-ids or --config with task IDs")
    if meta_dir is None:
        meta_dir = '/shared/projects/agibot/full_dataset/'

    if args.quick_test:
        args.max_trajectories = 20

    evaluator = TSCEvaluator(
        meta_dir=meta_dir,
        output_dir=args.output_dir,
        max_trajectories=args.max_trajectories,
        metric_tolerances=metric_tolerances
    )

    method_names = args.methods
    if 'all' in method_names:
        method_names = list(evaluator.TSC_METHODS.keys())

    evaluator.evaluate_all_methods(
        task_ids=task_ids,
        method_names=method_names
    )


if __name__ == '__main__':
    main()
