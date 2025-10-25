"""
Trajectory embedding visualization utility.

Visualizes embeddings across windows within trajectories to show
temporal progression through the embedding space.

Based on plot_fast_droid_embeddings.py but integrated with the training pipeline.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch


def extract_trajectory_windows_embeddings(
    trajectory_data: Any,
    dataset,
    max_windows: Optional[int] = None
) -> Tuple[np.ndarray, List[int], str]:
    """
    Extract embeddings for all windows in a trajectory.

    Args:
        trajectory_data: Trajectory from parquet loader
        dataset: BoundaryDataset instance
        max_windows: Optional limit on number of windows

    Returns:
        Tuple of (embeddings, window_indices, trajectory_id)
        - embeddings: (N_windows, embedding_dim) array
        - window_indices: List of window begin frame indices
        - trajectory_id: String identifier for trajectory
    """
    task_id = getattr(trajectory_data, 'task_id', 0)
    episode_id = getattr(trajectory_data, 'episode_id', 0)
    trajectory_id = f"T{task_id}-E{episode_id}"

    # Find all samples from this trajectory in the dataset
    trajectory_samples = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['task_id'] == task_id and sample['episode_id'] == episode_id:
            trajectory_samples.append((idx, sample))

    # Sort by window begin frame
    trajectory_samples.sort(key=lambda x: x[1]['window'].begin_frame)

    if max_windows:
        trajectory_samples = trajectory_samples[:max_windows]

    # Extract embeddings
    embeddings_list = []
    window_indices = []

    for idx, sample in trajectory_samples:
        try:
            # Extract proprio tokens
            proprio_tokens = sample['proprio_features'].numpy()

            # Extract DINO features (3 frames)
            dino_features = torch.cat([
                sample['dino_features']['begin_frame'],
                sample['dino_features']['middle_frame'],
                sample['dino_features']['end_frame']
            ]).numpy()

            # Concatenate
            embedding = np.concatenate([proprio_tokens, dino_features])

            embeddings_list.append(embedding)
            window_indices.append(sample['window'].begin_frame)
        except Exception as e:
            # Skip samples with mismatched dimensions
            continue

    if not embeddings_list:
        raise ValueError("No valid embeddings extracted from trajectory")

    embeddings = np.stack(embeddings_list, axis=0)

    return embeddings, window_indices, trajectory_id


def visualize_trajectory_embeddings(
    trajectories_data: List[Any],
    dataset,
    method: str = 'pca',
    num_trajectories: int = 3,
    max_windows_per_traj: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize embeddings from multiple trajectories showing temporal progression.

    Args:
        trajectories_data: List of trajectory objects from parquet loader
        dataset: BoundaryDataset instance
        method: 'pca' or 'tsne'
        num_trajectories: Number of trajectories to visualize
        max_windows_per_traj: Maximum windows per trajectory
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract embeddings from multiple trajectories
    all_embeddings = []
    all_window_indices = []
    all_trajectory_ids = []

    for traj_data in trajectories_data[:num_trajectories]:
        try:
            embeddings, window_indices, traj_id = extract_trajectory_windows_embeddings(
                traj_data, dataset, max_windows_per_traj
            )

            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
                all_window_indices.extend(window_indices)
                all_trajectory_ids.extend([traj_id] * len(embeddings))
        except Exception as e:
            print(f"Warning: Failed to extract trajectory: {e}")
            continue

    if not all_embeddings:
        raise ValueError("No embeddings extracted from trajectories")

    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)

    # Normalize
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_array)

    # Reduce dimensionality
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_normalized)
        explained_var = reducer.explained_variance_ratio_
        var_text = f"(Var: {explained_var[0]:.1%}, {explained_var[1]:.1%})"
    else:  # tsne
        perplexity = min(30, len(embeddings_array) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = reducer.fit_transform(embeddings_normalized)
        var_text = ""

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot colored by window index (timestep)
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=all_window_indices,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='k',
        linewidths=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Window/Segment Index (Timestep)', rotation=270, labelpad=20, fontsize=11)

    # Add trajectory labels
    n_labels = min(15, len(all_trajectory_ids))
    label_indices = np.linspace(0, len(all_trajectory_ids) - 1, n_labels, dtype=int)
    for i in label_indices:
        ax.annotate(
            f'{all_trajectory_ids[i]}-W{all_window_indices[i]}',
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=7,
            alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )

    # Set labels and title
    method_name = method.upper()
    if title is None:
        title = f'Trajectory Embeddings ({method_name}) {var_text}'

    ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def create_embedding_plot_for_logging(
    train_trajectories: List[Any],
    val_trajectories: List[Any],
    train_dataset,
    val_dataset,
    method: str = 'pca',
    num_trajectories_per_split: int = 2,
    max_windows_per_traj: Optional[int] = None
) -> Dict[str, plt.Figure]:
    """
    Create embedding plots for both train and val sets for logging.

    Args:
        train_trajectories: Training trajectory data
        val_trajectories: Validation trajectory data
        train_dataset: Training BoundaryDataset
        val_dataset: Validation BoundaryDataset
        method: 'pca' or 'tsne'
        num_trajectories_per_split: Number of trajectories to plot per split
        max_windows_per_traj: Max windows per trajectory

    Returns:
        Dictionary with 'train' and 'val' figures
    """
    figures = {}

    # Create training plot
    if train_trajectories and train_dataset:
        try:
            fig_train = visualize_trajectory_embeddings(
                train_trajectories,
                train_dataset,
                method=method,
                num_trajectories=num_trajectories_per_split,
                max_windows_per_traj=max_windows_per_traj,
                title=f'Training Trajectory Embeddings ({method.upper()})'
            )
            figures['train'] = fig_train
        except Exception as e:
            print(f"Warning: Failed to create training plot: {e}")

    # Create validation plot
    if val_trajectories and val_dataset:
        try:
            fig_val = visualize_trajectory_embeddings(
                val_trajectories,
                val_dataset,
                method=method,
                num_trajectories=num_trajectories_per_split,
                max_windows_per_traj=max_windows_per_traj,
                title=f'Validation Trajectory Embeddings ({method.upper()})'
            )
            figures['val'] = fig_val
        except Exception as e:
            print(f"Warning: Failed to create validation plot: {e}")

    return figures
