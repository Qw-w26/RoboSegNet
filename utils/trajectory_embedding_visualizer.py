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


def _get_action_label_for_window(sample: Dict[str, Any], freq_multiplier: int = 1) -> str:
    """
    Determine the action label for a window based on its frame position.

    Args:
        sample: Dataset sample containing 'window' and 'intervals'
        freq_multiplier: Frequency multiplier to convert window frames to proprio frames
                        (e.g., if window is at 2Hz and intervals at 30Hz, multiplier is 15)

    Returns:
        Action skill label (e.g., 'Pick', 'Place', 'Move')
    """
    window = sample['window']
    intervals = sample.get('intervals', [])

    if not intervals:
        return 'Unknown'

    # Find which interval this window's middle frame belongs to
    # Window frames are at window_fps, but intervals are at proprio_fps
    # So we need to scale the window frame to proprio_fps
    middle_frame_proprio = window.middle_frame * freq_multiplier

    for interval in intervals:
        if interval.start_frame <= middle_frame_proprio < interval.end_frame:
            # Return the skill name if available, otherwise action_text
            return interval.skill if interval.skill else interval.action_text

    # Default if no matching interval found
    return 'Unknown'


def _extract_layer_embedding(
    sample: Dict[str, Any],
    model: torch.nn.Module,
    layer: str,
    device: str
) -> np.ndarray:
    """
    Extract intermediate layer embedding from model.

    Args:
        sample: Dataset sample with input features
        model: Trained model
        layer: Layer to extract ('token_embed', 'proprio_encoded', 'dino_encoded',
               'combined', 'hidden1', 'hidden2', 'hidden3')
        device: Device to run on

    Returns:
        Embedding array
    """
    model.eval()
    with torch.no_grad():
        # Prepare inputs (batch size = 1)
        proprio_tokens = sample['proprio_features'].unsqueeze(0).to(device)
        dino_features_cat = torch.cat([
            sample['dino_features']['begin_frame'],
            sample['dino_features']['middle_frame'],
            sample['dino_features']['end_frame']
        ]).unsqueeze(0).to(device)

        if layer == 'token_embed':
            # Extract token embeddings (before transformer)
            if hasattr(model, 'proprio_encoder'):
                token_ids = proprio_tokens
                x = model.proprio_encoder.embedding(token_ids)  # (1, seq_len, embed_dim)
                # Mean pool over sequence
                key_padding_mask = (token_ids == model.proprio_encoder.pad_token_idx)
                lengths = (~key_padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
                x_masked = x * (~key_padding_mask).unsqueeze(-1)
                embedding = (x_masked.sum(dim=1) / lengths).squeeze(0)  # (embed_dim,)
            else:
                raise ValueError("Model does not have proprio_encoder")

        elif layer == 'proprio_encoded':
            # After transformer encoder
            if hasattr(model, 'proprio_encoder'):
                embedding = model.proprio_encoder(proprio_tokens).squeeze(0)
            else:
                raise ValueError("Model does not have proprio_encoder")

        elif layer == 'dino_encoded':
            # After DINO encoder
            if hasattr(model, 'dino_encoder'):
                embedding = model.dino_encoder(dino_features_cat).squeeze(0)
            else:
                raise ValueError("Model does not have dino_encoder")

        elif layer == 'combined':
            # After concatenation of encoders
            encoded_features = []
            if model.use_proprio:
                encoded_features.append(model.proprio_encoder(proprio_tokens))
            if model.use_dino:
                encoded_features.append(model.dino_encoder(dino_features_cat))
            embedding = torch.cat(encoded_features, dim=1).squeeze(0)

        elif layer in ['hidden1', 'hidden2', 'hidden3']:
            # Extract from MLP hidden layers
            encoded_features = []
            if model.use_proprio:
                encoded_features.append(model.proprio_encoder(proprio_tokens))
            if model.use_dino:
                encoded_features.append(model.dino_encoder(dino_features_cat))
            combined = torch.cat(encoded_features, dim=1)

            # Pass through MLP layers
            layer_idx = int(layer[-1]) - 1  # 'hidden1' -> 0, 'hidden2' -> 1, 'hidden3' -> 2

            # Each hidden layer consists of: Linear, BatchNorm, ReLU, Dropout
            # So layer 1 is at indices 0-3, layer 2 at 4-7, layer 3 at 8-11
            target_layer = (layer_idx + 1) * 4 - 1  # Get output after dropout

            x = combined
            for i, module in enumerate(model.mlp[:target_layer + 1]):
                x = module(x)

            embedding = x.squeeze(0)

        else:
            raise ValueError(f"Unknown layer: {layer}")

        return embedding.cpu().numpy()


def extract_trajectory_windows_embeddings(
    trajectory_data: Any,
    dataset,
    max_windows: Optional[int] = None,
    model: Optional[torch.nn.Module] = None,
    layer: str = 'input',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, List[int], str, List[str]]:
    """
    Extract embeddings for all windows in a trajectory.

    Args:
        trajectory_data: Trajectory from parquet loader
        dataset: BoundaryDataset instance
        max_windows: Optional limit on number of windows
        model: Optional model for extracting intermediate layer embeddings
        layer: Which layer to extract ('input', 'token_embed', 'proprio_encoded',
               'dino_encoded', 'combined', 'hidden1', 'hidden2', 'hidden3')
        device: Device to run model on

    Returns:
        Tuple of (embeddings, window_indices, trajectory_id, action_labels)
        - embeddings: (N_windows, embedding_dim) array
        - window_indices: List of window begin frame indices
        - trajectory_id: String identifier for trajectory
        - action_labels: List of action skill labels for each window
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

    # Get frequency multiplier from dataset (for converting window frames to proprio frames)
    freq_multiplier = getattr(dataset, 'freq_multiplier', 1)

    # Extract embeddings
    embeddings_list = []
    window_indices = []
    action_labels = []

    for idx, sample in trajectory_samples:
        try:
            # Extract action label for this window
            action_label = _get_action_label_for_window(sample, freq_multiplier=freq_multiplier)
            action_labels.append(action_label)

            if layer == 'input':
                # Original behavior: raw input features
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

            elif model is not None:
                # Extract intermediate layer embedding using model
                embedding = _extract_layer_embedding(sample, model, layer, device)
            else:
                raise ValueError(f"Model required for layer '{layer}' but not provided")

            embeddings_list.append(embedding)
            window_indices.append(sample['window'].begin_frame)
        except Exception as e:
            # Skip samples with mismatched dimensions
            continue

    if not embeddings_list:
        raise ValueError("No valid embeddings extracted from trajectory")

    embeddings = np.stack(embeddings_list, axis=0)

    return embeddings, window_indices, trajectory_id, action_labels


def visualize_trajectory_embeddings(
    trajectories_data: List[Any],
    dataset,
    method: str = 'pca',
    num_trajectories: int = 3,
    max_windows_per_traj: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    model: Optional[torch.nn.Module] = None,
    layer: str = 'input',
    color_by: str = 'timestep',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        model: Optional model for extracting intermediate layer embeddings
        layer: Which layer to extract ('input', 'token_embed', 'proprio_encoded', etc.)
        color_by: How to color points ('timestep' or 'action')
        device: Device to run model on

    Returns:
        Matplotlib figure
    """
    # Extract embeddings from multiple trajectories
    all_embeddings = []
    all_window_indices = []
    all_trajectory_ids = []
    all_action_labels = []

    for traj_data in trajectories_data[:num_trajectories]:
        try:
            embeddings, window_indices, traj_id, action_labels = extract_trajectory_windows_embeddings(
                traj_data, dataset, max_windows_per_traj, model, layer, device
            )

            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
                all_window_indices.extend(window_indices)
                all_trajectory_ids.extend([traj_id] * len(embeddings))
                all_action_labels.extend(action_labels)
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

    if color_by == 'action':
        # Color by action labels
        unique_actions = sorted(list(set(all_action_labels)))
        action_to_idx = {action: i for i, action in enumerate(unique_actions)}
        color_values = [action_to_idx[action] for action in all_action_labels]

        # Create scatter plot
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=color_values,
            cmap='tab10' if len(unique_actions) <= 10 else 'tab20',
            alpha=0.6,
            s=50,
            edgecolors='k',
            linewidths=0.5
        )

        # Add colorbar with action labels
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Action Skill', rotation=270, labelpad=20, fontsize=11)

        # Set colorbar ticks to show action names
        if len(unique_actions) <= 10:
            cbar.set_ticks(range(len(unique_actions)))
            cbar.set_ticklabels(unique_actions)
    else:
        # Color by window index (timestep) - original behavior
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
        if color_by == 'action':
            label_text = f'{all_trajectory_ids[i]}-{all_action_labels[i]}'
        else:
            label_text = f'{all_trajectory_ids[i]}-W{all_window_indices[i]}'

        ax.annotate(
            label_text,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=7,
            alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )

    # Set labels and title
    method_name = method.upper()
    layer_name = layer.replace('_', ' ').title()
    if title is None:
        title = f'Trajectory Embeddings - {layer_name} ({method_name}) {var_text}'

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
    max_windows_per_traj: Optional[int] = None,
    model: Optional[torch.nn.Module] = None,
    layer: str = 'input',
    color_by: str = 'timestep',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        model: Optional model for extracting intermediate layer embeddings
        layer: Which layer to extract ('input', 'token_embed', 'proprio_encoded', etc.)
        color_by: How to color points ('timestep' or 'action')
        device: Device to run model on

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
                title=f'Training Trajectory Embeddings ({method.upper()})',
                model=model,
                layer=layer,
                color_by=color_by,
                device=device
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
                title=f'Validation Trajectory Embeddings ({method.upper()})',
                model=model,
                layer=layer,
                color_by=color_by,
                device=device
            )
            figures['val'] = fig_val
        except Exception as e:
            print(f"Warning: Failed to create validation plot: {e}")

    return figures
