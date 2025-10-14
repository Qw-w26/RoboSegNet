import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import ruptures as rpt
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import h5py
from pathlib import Path
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class TrajectorySegmenter:
    """
    Segments robot trajectories based on embeddings from multiple modalities.
    Combines action patterns (Fast-DROID) and visual features (DINOv3).
    """
    
    def __init__(self, embeddings, window_info, window_size=15, stride=1):
        """
        Initialize the segmenter with embeddings and metadata.
        
        Args:
            embeddings: Array of shape (n_windows, embedding_dim)
            window_info: List of tuples (trajectory_id, window_index)
            window_size: Size of the sliding window
            stride: Stride used for window creation
        """
        self.embeddings = embeddings
        self.window_info = window_info
        self.window_size = window_size
        self.stride = stride
        
        # Normalize embeddings
        self.scaler = StandardScaler()
        self.embeddings_normalized = self.scaler.fit_transform(embeddings)
        
        # Group windows by trajectory
        self.trajectory_groups = self._group_by_trajectory()
        
    def _group_by_trajectory(self):
        """Group window indices by trajectory ID."""
        traj_groups = {}
        for idx, (traj_id, window_idx) in enumerate(self.window_info):
            if traj_id not in traj_groups:
                traj_groups[traj_id] = []
            traj_groups[traj_id].append(idx)
        return traj_groups
    
    def segment_with_spectral_clustering(self, n_clusters=5, gamma=1.0, temporal_weight=0.3):
        """
        Apply spectral clustering with temporal constraints.
        
        Args:
            n_clusters: Number of clusters
            gamma: RBF kernel parameter
            temporal_weight: Weight for temporal proximity (0-1)
        """
        print(f"Applying Spectral Clustering with {n_clusters} clusters...")
        
        # Compute affinity matrix with temporal constraints
        n_samples = len(self.embeddings_normalized)
        
        # Spatial affinity (embedding similarity)
        spatial_distances = squareform(pdist(self.embeddings_normalized, 'euclidean'))
        spatial_affinity = np.exp(-gamma * spatial_distances ** 2)
        
        # Temporal affinity (penalize distant windows)
        temporal_affinity = np.zeros((n_samples, n_samples))
        for traj_id, indices in self.trajectory_groups.items():
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    # Windows from same trajectory
                    temporal_dist = abs(i - j) * self.stride
                    temporal_affinity[idx_i, idx_j] = np.exp(-temporal_dist / self.window_size)
        
        # Combined affinity
        affinity = (1 - temporal_weight) * spatial_affinity + temporal_weight * temporal_affinity
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(affinity)
        
        return labels
    
    def segment_with_change_point_detection(self, penalty=3, min_size=5):
        """
        Detect change points in trajectory embeddings.
        
        Args:
            penalty: Penalty value for change point detection
            min_size: Minimum segment size
        """
        print("Detecting change points in trajectories...")
        
        all_change_points = {}
        all_segments = {}
        
        for traj_id, indices in self.trajectory_groups.items():
            # Get embeddings for this trajectory
            traj_embeddings = self.embeddings_normalized[indices]
            
            if len(traj_embeddings) < 2 * min_size:
                # Too short for segmentation
                all_segments[traj_id] = [0] * len(traj_embeddings)
                continue
            
            # Apply change point detection using Kernel method
            algo = rpt.KernelCPD(kernel="rbf", min_size=min_size)
            algo.fit(traj_embeddings)
            
            # Find change points
            n_bkps = min(len(traj_embeddings) // (2 * min_size), 5)  # Max 5 segments
            change_points = algo.predict(n_bkps=n_bkps)
            
            # Convert to segment labels
            segments = np.zeros(len(traj_embeddings), dtype=int)
            start = 0
            for seg_id, end in enumerate(change_points):
                segments[start:end] = seg_id
                start = end
            
            all_change_points[traj_id] = change_points
            all_segments[traj_id] = segments
        
        # Convert to global labels
        global_labels = np.zeros(len(self.embeddings), dtype=int)
        for traj_id, indices in self.trajectory_groups.items():
            if traj_id in all_segments:
                for local_idx, global_idx in enumerate(indices):
                    global_labels[global_idx] = all_segments[traj_id][local_idx]
        
        return global_labels, all_change_points
    
    def segment_with_gmm(self, n_components=5):
        """
        Apply Gaussian Mixture Model for soft clustering.
        
        Args:
            n_components: Number of Gaussian components
        """
        print(f"Applying GMM with {n_components} components...")
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        
        labels = gmm.fit_predict(self.embeddings_normalized)
        probabilities = gmm.predict_proba(self.embeddings_normalized)
        
        return labels, probabilities
    
    def hierarchical_segmentation(self, distance_threshold=None, n_clusters=None):
        """
        Apply hierarchical clustering for multi-level segmentation.
        
        Args:
            distance_threshold: Distance threshold for cutting dendrogram
            n_clusters: Number of clusters (alternative to distance_threshold)
        """
        print("Performing hierarchical segmentation...")
        
        # Compute linkage
        linkage_matrix = linkage(self.embeddings_normalized, method='ward')
        
        if n_clusters is not None:
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        elif distance_threshold is not None:
            labels = fcluster(linkage_matrix, distance_threshold, criterion='distance') - 1
        else:
            # Use elbow method to find optimal number of clusters
            distances = linkage_matrix[:, 2]
            diff = np.diff(distances)
            elbow_idx = np.argmax(diff) + 1
            distance_threshold = distances[elbow_idx]
            labels = fcluster(linkage_matrix, distance_threshold, criterion='distance') - 1
        
        return labels, linkage_matrix
    
    def evaluate_segmentation(self, labels):
        """
        Evaluate segmentation quality using multiple metrics.
        
        Args:
            labels: Cluster labels
        """
        # Skip if only one cluster
        n_unique = len(np.unique(labels))
        if n_unique <= 1:
            return {'silhouette': 0, 'davies_bouldin': float('inf'), 'temporal_coherence': 0}
        
        # Silhouette score (higher is better, range: -1 to 1)
        silhouette = silhouette_score(self.embeddings_normalized, labels)
        
        # Davies-Bouldin score (lower is better)
        davies_bouldin = davies_bouldin_score(self.embeddings_normalized, labels)
        
        # Temporal coherence (custom metric)
        temporal_coherence = self._compute_temporal_coherence(labels)
        
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'temporal_coherence': temporal_coherence,
            'n_clusters': n_unique
        }
    
    def _compute_temporal_coherence(self, labels):
        """
        Compute how well segments respect temporal continuity.
        Higher score means segments are more temporally coherent.
        """
        coherence_scores = []
        
        for traj_id, indices in self.trajectory_groups.items():
            if len(indices) <= 1:
                continue
                
            traj_labels = labels[indices]
            # Count transitions
            transitions = np.sum(np.diff(traj_labels) != 0)
            # Normalize by trajectory length
            coherence = 1.0 - (transitions / (len(traj_labels) - 1))
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0
    
    def visualize_segmentation(self, labels, method_name="", save_path=None):
        """
        Create comprehensive visualization of segmentation results.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Get 2D projection for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings_normalized)
        
        # 1. PCA scatter with segment colors
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=30)
        ax1.set_title(f'Segmentation Result - {method_name}')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        plt.colorbar(scatter, ax=ax1, label='Segment ID')
        
        # 2. Temporal evolution of segments
        ax2 = plt.subplot(2, 3, 2)
        for traj_id in sorted(self.trajectory_groups.keys())[:5]:  # Show first 5 trajectories
            indices = self.trajectory_groups[traj_id]
            traj_labels = labels[indices]
            window_positions = np.arange(len(traj_labels))
            ax2.plot(window_positions, traj_labels + traj_id * 0.1, 
                    marker='o', label=f'Traj {traj_id}', alpha=0.7)
        ax2.set_title('Segment Evolution Over Time')
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Segment ID (offset by trajectory)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Segment size distribution
        ax3 = plt.subplot(2, 3, 3)
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax3.bar(unique_labels, counts, color=plt.cm.tab10(unique_labels / max(unique_labels)))
        ax3.set_title('Segment Size Distribution')
        ax3.set_xlabel('Segment ID')
        ax3.set_ylabel('Number of Windows')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trajectory segmentation heatmap
        ax4 = plt.subplot(2, 3, 4)
        max_traj_len = max(len(indices) for indices in self.trajectory_groups.values())
        n_trajs = min(10, len(self.trajectory_groups))
        
        seg_matrix = np.full((n_trajs, max_traj_len), np.nan)
        for i, (traj_id, indices) in enumerate(list(self.trajectory_groups.items())[:n_trajs]):
            traj_labels = labels[indices]
            seg_matrix[i, :len(traj_labels)] = traj_labels
        
        sns.heatmap(seg_matrix, cmap='tab10', cbar_kws={'label': 'Segment ID'},
                   ax=ax4, mask=np.isnan(seg_matrix))
        ax4.set_title('Trajectory Segmentation Patterns')
        ax4.set_xlabel('Window Index')
        ax4.set_ylabel('Trajectory ID')
        
        # 5. Transition matrix
        ax5 = plt.subplot(2, 3, 5)
        n_segments = len(np.unique(labels))
        transition_matrix = np.zeros((n_segments, n_segments))
        
        for traj_id, indices in self.trajectory_groups.items():
            traj_labels = labels[indices]
            for i in range(len(traj_labels) - 1):
                transition_matrix[traj_labels[i], traj_labels[i+1]] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     where=row_sums != 0, out=transition_matrix)
        
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   ax=ax5, vmin=0, vmax=1, cbar_kws={'label': 'Transition Probability'})
        ax5.set_title('Segment Transition Matrix')
        ax5.set_xlabel('To Segment')
        ax5.set_ylabel('From Segment')
        
        # 6. Metrics display
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        metrics = self.evaluate_segmentation(labels)
        metrics_text = f"Segmentation Metrics - {method_name}\n\n"
        metrics_text += f"Number of Segments: {metrics['n_clusters']}\n"
        metrics_text += f"Silhouette Score: {metrics['silhouette']:.3f}\n"
        metrics_text += f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}\n"
        metrics_text += f"Temporal Coherence: {metrics['temporal_coherence']:.3f}\n"
        ax6.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Trajectory Segmentation Analysis - {method_name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig
    
    def compare_methods(self):
        """
        Compare multiple segmentation methods and select the best one.
        """
        results = {}
        
        # 1. Spectral Clustering with temporal constraints
        labels_spectral = self.segment_with_spectral_clustering(n_clusters=5, temporal_weight=0.3)
        results['spectral'] = {
            'labels': labels_spectral,
            'metrics': self.evaluate_segmentation(labels_spectral)
        }
        
        # 2. Change Point Detection
        labels_cpd, change_points = self.segment_with_change_point_detection(penalty=3)
        results['change_point'] = {
            'labels': labels_cpd,
            'metrics': self.evaluate_segmentation(labels_cpd),
            'change_points': change_points
        }
        
        # 3. Gaussian Mixture Model
        labels_gmm, probs = self.segment_with_gmm(n_components=5)
        results['gmm'] = {
            'labels': labels_gmm,
            'metrics': self.evaluate_segmentation(labels_gmm),
            'probabilities': probs
        }
        
        # 4. Hierarchical Clustering
        labels_hier, linkage_matrix = self.hierarchical_segmentation(n_clusters=5)
        results['hierarchical'] = {
            'labels': labels_hier,
            'metrics': self.evaluate_segmentation(labels_hier),
            'linkage': linkage_matrix
        }
        
        # Compare and select best method
        print("\n" + "="*50)
        print("Method Comparison:")
        print("="*50)
        
        best_method = None
        best_score = -float('inf')
        
        for method, result in results.items():
            metrics = result['metrics']
            # Combined score (weighted average)
            score = (metrics['silhouette'] + metrics['temporal_coherence']) / 2
            
            print(f"\n{method.upper()}:")
            print(f"  Clusters: {metrics['n_clusters']}")
            print(f"  Silhouette: {metrics['silhouette']:.3f}")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
            print(f"  Temporal Coherence: {metrics['temporal_coherence']:.3f}")
            print(f"  Combined Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_method = method
        
        print(f"\n{'='*50}")
        print(f"Best Method: {best_method.upper()} (Score: {best_score:.3f})")
        print("="*50)
        
        return results, best_method


def main():
    """
    Main function to run trajectory segmentation on saved embeddings.
    """
    import pickle
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment trajectories based on embeddings')
    parser.add_argument('--embeddings_path', type=str, 
                       default='/home/syx/ucsf/task_seg/embeddings_cache.pkl',
                       help='Path to saved embeddings')
    parser.add_argument('--window_size', type=int, default=20,
                       help='Window size used for embedding creation')
    parser.add_argument('--stride', type=int, default=5,
                       help='Stride used for embedding creation')
    args = parser.parse_args()
    
    # Load pre-computed embeddings (we'll need to save these from the original script)
    # For now, let's run the embedding generation first
    print("Loading and processing trajectory data...")
    
    # Import and run the embedding generation
    from plot_fast_droid_embeddings import (
        load_fast_droid_tokenizer, 
        extract_data_from_h5, 
        create_windows,
        create_embeddings_from_windows_with_dino,
        extract_dino_features,
        extract_frame_from_video
    )
    
    import glob
    import ray
    
    # Initialize Ray for parallel processing
    ray.init(ignore_reinit_error=True)
    
    # Dataset path
    dataset_path = "/home/kushtimusprime/ur5_policy_training/data_tiger_picking_90_good_and_bad_cleaned"
    
    # Load tokenizer
    tokenizer = load_fast_droid_tokenizer()
    
    # Find HDF5 files
    h5_files = sorted(glob.glob(f"{dataset_path}/trajectory_*/joints.h5"))[85:]  # Use 10 trajectories
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Process trajectories
    all_windows = []
    all_dino_features = []
    window_info = []
    
    for h5_path in tqdm(h5_files, desc="Processing trajectories"):
        trajectory = extract_data_from_h5(h5_path)
        if trajectory is None or len(trajectory) < args.window_size:
            continue
        
        # Extract trajectory ID
        traj_id = int(Path(h5_path).parent.name.split('_')[-1])
        
        # Get video path
        video_path = Path(h5_path).parent / "wrist.mp4"
        if not video_path.exists():
            video_path = Path(h5_path).parent / "exo.mp4"
        
        # Create windows
        windows = create_windows(trajectory, args.window_size, args.stride, use_joints=False)
        
        # Extract DINO features for center frames
        for window, center_frame_idx in windows:
            all_windows.append(window)
            window_info.append((traj_id, len(all_windows) - 1))
            
            # Extract DINO features (simplified for now - you may want to parallelize)
            if video_path.exists():
                image = extract_frame_from_video(video_path, center_frame_idx)
                if image:
                    dino_features = extract_dino_features(image)
                else:
                    dino_features = np.zeros(1024)
            else:
                dino_features = np.zeros(1024)
            all_dino_features.append(dino_features)
    
    print(f"Created {len(all_windows)} windows from {len(h5_files)} trajectories")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings_from_windows_with_dino(
        all_windows, tokenizer, dino_features_list=all_dino_features
    )
    
    # Save embeddings for future use
    cache_path = '/home/syx/ucsf/task_seg/embeddings_cache.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'window_info': window_info,
            'window_size': args.window_size,
            'stride': args.stride
        }, f)
    print(f"Saved embeddings to {cache_path}")
    
    # Initialize segmenter
    segmenter = TrajectorySegmenter(
        embeddings, 
        window_info, 
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Compare segmentation methods
    results, best_method = segmenter.compare_methods()
    
    # Visualize best method
    best_labels = results[best_method]['labels']
    fig = segmenter.visualize_segmentation(
        best_labels, 
        method_name=best_method.upper(),
        save_path=f'/home/syx/ucsf/task_seg/segmentation_{best_method}.png'
    )
    
    # Save segmentation results
    results_path = '/home/syx/ucsf/task_seg/segmentation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved segmentation results to {results_path}")
    
    plt.show()
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()