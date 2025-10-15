"""
Clustering-based segmentation methods.

This module wraps the existing clustering methods from trajectory_segmentation.py
to be compatible with the benchmark framework.
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import ruptures as rpt

from .base_method import WindowBasedSegmentationMethod


class SpectralClusteringMethod(WindowBasedSegmentationMethod):
    """Spectral Clustering with temporal constraints for segmentation."""

    def __init__(
        self,
        n_clusters: int = 5,
        gamma: float = 1.0,
        temporal_weight: float = 0.3,
        window_size: int = 15,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize Spectral Clustering method.

        Args:
            n_clusters: Number of clusters
            gamma: RBF kernel parameter
            temporal_weight: Weight for temporal proximity (0-1)
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        super().__init__(
            name="Spectral Clustering",
            window_size=window_size,
            stride=stride,
            n_clusters=n_clusters,
            gamma=gamma,
            temporal_weight=temporal_weight,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.temporal_weight = temporal_weight
        self.scaler = StandardScaler()

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        """Fit scaler on training data."""
        if embeddings is not None:
            self.scaler.fit(embeddings)

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Predict segment labels using spectral clustering.

        Args:
            trajectory_data: TrajectoryData object
            embeddings: Pre-computed embeddings for windows (n_windows, embedding_dim)

        Returns:
            Frame-level segment labels
        """
        if embeddings is None:
            raise ValueError("Spectral Clustering requires pre-computed embeddings")

        # Normalize embeddings
        embeddings_normalized = self.scaler.transform(embeddings)

        # Compute affinity matrix with temporal constraints
        n_samples = len(embeddings_normalized)

        # Spatial affinity (embedding similarity)
        spatial_distances = squareform(pdist(embeddings_normalized, 'euclidean'))
        spatial_affinity = np.exp(-self.gamma * spatial_distances ** 2)

        # Temporal affinity (penalize distant windows)
        temporal_affinity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                temporal_dist = abs(i - j) * self.stride
                temporal_affinity[i, j] = np.exp(-temporal_dist / self.window_size)

        # Combined affinity
        affinity = ((1 - self.temporal_weight) * spatial_affinity +
                   self.temporal_weight * temporal_affinity)

        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42
        )
        window_labels = clustering.fit_predict(affinity)

        # Convert to frame labels
        num_frames = trajectory_data.total_frames
        frame_labels = self.window_labels_to_frame_labels(window_labels, num_frames)

        return frame_labels


class ChangePointDetectionMethod(WindowBasedSegmentationMethod):
    """Change Point Detection using ruptures library."""

    def __init__(
        self,
        penalty: float = 3.0,
        min_size: int = 5,
        max_segments: int = 5,
        window_size: int = 15,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize Change Point Detection method.

        Args:
            penalty: Penalty value for change point detection
            min_size: Minimum segment size
            max_segments: Maximum number of segments
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        super().__init__(
            name="Change Point Detection",
            window_size=window_size,
            stride=stride,
            penalty=penalty,
            min_size=min_size,
            max_segments=max_segments,
            **kwargs
        )
        self.penalty = penalty
        self.min_size = min_size
        self.max_segments = max_segments
        self.scaler = StandardScaler()

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        """Fit scaler on training data."""
        if embeddings is not None:
            self.scaler.fit(embeddings)

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Predict segment labels using change point detection.

        Args:
            trajectory_data: TrajectoryData object
            embeddings: Pre-computed embeddings for windows

        Returns:
            Frame-level segment labels
        """
        if embeddings is None:
            raise ValueError("Change Point Detection requires pre-computed embeddings")

        # Normalize embeddings
        embeddings_normalized = self.scaler.transform(embeddings)

        if len(embeddings_normalized) < 2 * self.min_size:
            # Too short for segmentation
            return np.zeros(trajectory_data.total_frames, dtype=int)

        # Apply change point detection using Kernel method
        algo = rpt.KernelCPD(kernel="rbf", min_size=self.min_size)
        algo.fit(embeddings_normalized)

        # Find change points
        n_bkps = min(len(embeddings_normalized) // (2 * self.min_size), self.max_segments)
        if n_bkps < 1:
            return np.zeros(trajectory_data.total_frames, dtype=int)

        change_points = algo.predict(n_bkps=n_bkps)

        # Convert to window labels
        window_labels = np.zeros(len(embeddings_normalized), dtype=int)
        start = 0
        for seg_id, end in enumerate(change_points):
            window_labels[start:end] = seg_id
            start = end

        # Convert to frame labels
        frame_labels = self.window_labels_to_frame_labels(
            window_labels,
            trajectory_data.total_frames
        )

        return frame_labels


class GMMMethod(WindowBasedSegmentationMethod):
    """Gaussian Mixture Model for segmentation."""

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = 'full',
        window_size: int = 15,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize GMM method.

        Args:
            n_components: Number of Gaussian components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        super().__init__(
            name="Gaussian Mixture Model",
            window_size=window_size,
            stride=stride,
            n_components=n_components,
            covariance_type=covariance_type,
            **kwargs
        )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.scaler = StandardScaler()

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        """Fit scaler on training data."""
        if embeddings is not None:
            self.scaler.fit(embeddings)

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Predict segment labels using GMM.

        Args:
            trajectory_data: TrajectoryData object
            embeddings: Pre-computed embeddings for windows

        Returns:
            Frame-level segment labels
        """
        if embeddings is None:
            raise ValueError("GMM requires pre-computed embeddings")

        # Normalize embeddings
        embeddings_normalized = self.scaler.transform(embeddings)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=42,
            n_init=10
        )
        window_labels = gmm.fit_predict(embeddings_normalized)

        # Convert to frame labels
        frame_labels = self.window_labels_to_frame_labels(
            window_labels,
            trajectory_data.total_frames
        )

        return frame_labels


class HierarchicalClusteringMethod(WindowBasedSegmentationMethod):
    """Hierarchical Clustering for segmentation."""

    def __init__(
        self,
        n_clusters: int = 5,
        linkage_method: str = 'ward',
        window_size: int = 15,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize Hierarchical Clustering method.

        Args:
            n_clusters: Number of clusters
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        super().__init__(
            name="Hierarchical Clustering",
            window_size=window_size,
            stride=stride,
            n_clusters=n_clusters,
            linkage_method=linkage_method,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.scaler = StandardScaler()

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        """Fit scaler on training data."""
        if embeddings is not None:
            self.scaler.fit(embeddings)

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Predict segment labels using hierarchical clustering.

        Args:
            trajectory_data: TrajectoryData object
            embeddings: Pre-computed embeddings for windows

        Returns:
            Frame-level segment labels
        """
        if embeddings is None:
            raise ValueError("Hierarchical Clustering requires pre-computed embeddings")

        # Normalize embeddings
        embeddings_normalized = self.scaler.transform(embeddings)

        # Compute linkage
        linkage_matrix = linkage(embeddings_normalized, method=self.linkage_method)

        # Get clusters
        window_labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust') - 1

        # Convert to frame labels
        frame_labels = self.window_labels_to_frame_labels(
            window_labels,
            trajectory_data.total_frames
        )

        return frame_labels


class KMeansMethod(WindowBasedSegmentationMethod):
    """K-Means clustering for segmentation."""

    def __init__(
        self,
        n_clusters: int = 5,
        window_size: int = 15,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize K-Means method.

        Args:
            n_clusters: Number of clusters
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        super().__init__(
            name="K-Means",
            window_size=window_size,
            stride=stride,
            n_clusters=n_clusters,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def fit(self, trajectories: list, embeddings: np.ndarray = None):
        """Fit scaler on training data."""
        if embeddings is not None:
            self.scaler.fit(embeddings)

    def predict(self, trajectory_data, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Predict segment labels using K-Means.

        Args:
            trajectory_data: TrajectoryData object
            embeddings: Pre-computed embeddings for windows

        Returns:
            Frame-level segment labels
        """
        if embeddings is None:
            raise ValueError("K-Means requires pre-computed embeddings")

        # Normalize embeddings
        embeddings_normalized = self.scaler.transform(embeddings)

        # Fit K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        window_labels = kmeans.fit_predict(embeddings_normalized)

        # Convert to frame labels
        frame_labels = self.window_labels_to_frame_labels(
            window_labels,
            trajectory_data.total_frames
        )

        return frame_labels
