import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer
import torch
from pathlib import Path
from tqdm import tqdm
import glob
from sklearn.preprocessing import StandardScaler
import yourdfpy

# Initialize URDF model globally
URDF_PATH = "/home/kushtimusprime/ur5_policy_training/urdf/ur5_with_robotiq_gripper.urdf"
robot_model = None

def initialize_robot_model():
    """Initialize the robot model from URDF."""
    global robot_model
    if robot_model is None:
        robot_model = yourdfpy.URDF.load(URDF_PATH)
    return robot_model

def ur5_forward_kinematics(joint_angles, normalize=False, 
                          joint_min=None, joint_max=None):
    """
    Compute forward kinematics for UR5 robot using URDF and append gripper value.
    
    Args:
        joint_angles: Array of shape (n_timesteps, n_joints) with joint angles
                     Last column is gripper value (0-255 range)
        normalize: If True, normalize output to [-1, 1] range
        joint_min: Min bounds for normalization (Cartesian + gripper)
        joint_max: Max bounds for normalization (Cartesian + gripper)
    
    Returns:
        Array of shape (n_timesteps, 7) with [x, y, z, roll, pitch, yaw, gripper]
    """
    robot = initialize_robot_model()
    
    n_timesteps = joint_angles.shape[0]
    # Extract and normalize gripper values (last column) from 0-255 to 0-1
    gripper_values = joint_angles[:, -1] / 255.0 if joint_angles.shape[1] >= 7 else np.zeros(n_timesteps)
    
    # Output includes 6 pose values + 1 gripper value
    cartesian_poses = np.zeros((n_timesteps, 7))
    
    # Get joint names from the robot model (only revolute/continuous joints)
    joint_names = [joint.name for joint in robot.joint_map.values() 
                   if joint.type in ['revolute', 'continuous']]
    
    # Use only the first 6 joints for UR5 FK (exclude gripper)
    n_joints_to_use = min(6, joint_angles.shape[1], len(joint_names))
    
    for t in range(n_timesteps):
        try:
            # Create configuration dictionary (first 6 joints only)
            cfg = {joint_names[i]: float(joint_angles[t, i]) 
                   for i in range(n_joints_to_use)}
            
            # Update robot configuration
            robot.update_cfg(cfg)
            
            # Get end-effector link (usually the last link)
            # Try common end-effector link names
            ee_link_names = ['tool0', 'ee_link', 'wrist_3_link', 'robotiq_85_base_link']
            ee_matrix = None
            
            for ee_name in ee_link_names:
                if ee_name in robot.link_map:
                    ee_matrix = robot.get_transform(ee_name, 'base_link')
                    break
            
            if ee_matrix is None:
                # Fallback: use the last link
                link_names = list(robot.link_map.keys())
                ee_matrix = robot.get_transform(link_names[-1], 'base_link')
            
            # Extract position (x, y, z)
            cartesian_poses[t, 0:3] = ee_matrix[0:3, 3]
            
            # Extract orientation (roll, pitch, yaw) from rotation matrix
            R = ee_matrix[0:3, 0:3]
            
            # Convert rotation matrix to Euler angles (ZYX convention)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = np.arctan2(R[1, 0], R[0, 0])
            else:
                roll = np.arctan2(-R[1, 2], R[1, 1])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = 0
            
            cartesian_poses[t, 3] = roll
            cartesian_poses[t, 4] = pitch
            cartesian_poses[t, 5] = yaw
            
            # Append gripper value (already normalized to 0-1)
            cartesian_poses[t, 6] = gripper_values[t]
            
        except Exception as e:
            # If forward kinematics fails for this timestep, keep zeros
            continue
    
    # Optionally normalize to [-1, 1] range
    if normalize and joint_min is not None and joint_max is not None:
        cartesian_poses = 2.0 * ((cartesian_poses - joint_min) / (joint_max - joint_min)) - 1.0
    
    return cartesian_poses

def load_fast_droid_tokenizer():
    """Load the fast_droid tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained("KarlP/fast-droid")
    # Add special tokens since this tokenizer doesn't have them
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'bos_token': '[BOS]',
        'unk_token': '[UNK]'
    })
    return tokenizer

def extract_data_from_h5(h5_path):
    """Extract relevant data from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Extract joint states
        if 'joint_states' in f:
            joint_states = f['joint_states'][:]
        else:
            # Try to find any dataset in the file
            keys = list(f.keys())
            if keys:
                joint_states = f[keys[0]][:]
            else:
                return None
        
        return joint_states

def create_windows(trajectory, window_size, stride):
    """Create sliding windows from a trajectory and convert to Cartesian.
    
    Args:
        trajectory: Joint state trajectory
        window_size: Size of the sliding window
        stride: Stride for the sliding window
    
    Returns:
        List of windows in Cartesian space with gripper values
    """
    windows = []
    n_timesteps = len(trajectory)
    
    # Create windows with the specified stride
    for start_idx in range(0, n_timesteps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = trajectory[start_idx:end_idx]
        
        # Convert joint states to Cartesian poses (always)
        window = ur5_forward_kinematics(window)
        
        windows.append(window)
    
    # Handle the last window if there's remaining data
    if n_timesteps > window_size and (n_timesteps - window_size) % stride != 0:
        window = trajectory[-window_size:]
        
        # Convert joint states to Cartesian poses
        window = ur5_forward_kinematics(window)
        
        windows.append(window)
    
    return windows

def create_embeddings_from_windows(windows_list, tokenizer):
    """Create embeddings from trajectory windows using tokenizer."""
    embeddings = []
    max_token_length = None
    
    # First pass: tokenize all windows to find max token length
    all_tokens = []
    for window in tqdm(windows_list, desc="Tokenizing windows"):
        if len(window) > 1:
            actions = np.diff(window, axis=0)
            # Convert entire action sequence to string format
            action_str = " ".join([f"{x:.4f}" for x in actions.flatten()])
            
            # Tokenize the full action sequence
            tokens = tokenizer(action_str, truncation=False, return_tensors='np')
            all_tokens.append(tokens['input_ids'].flatten())
            
            if max_token_length is None:
                max_token_length = len(tokens['input_ids'].flatten())
            else:
                max_token_length = max(max_token_length, len(tokens['input_ids'].flatten()))
        else:
            all_tokens.append(np.array([]))
    
    print(f"Max token length: {max_token_length}")
    
    # Second pass: create embeddings with consistent size
    for window, tokens in tqdm(zip(windows_list, all_tokens), 
                               desc="Creating embeddings", 
                               total=len(windows_list)):
        if len(window) > 1 and len(tokens) > 0:
            # Pad or truncate tokens to max_token_length
            if len(tokens) < max_token_length:
                # Pad with tokenizer's pad token ID
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                token_embedding = np.pad(tokens, (0, max_token_length - len(tokens)), 
                                        constant_values=pad_id)
            else:
                token_embedding = tokens[:max_token_length]
            
            # Add statistical features of the window
            window_features = np.concatenate([
                window.mean(axis=0),
                window.std(axis=0),
                window.min(axis=0),
                window.max(axis=0),
                np.percentile(window, 25, axis=0),
                np.percentile(window, 75, axis=0)
            ])
            
            # Combine full token embedding with window features
            combined_embedding = np.concatenate([token_embedding, window_features])
        else:
            # Handle edge case of very small windows
            combined_embedding = np.zeros(max_token_length + 6 * window.shape[1])
        
        embeddings.append(combined_embedding)
    
    return np.array(embeddings)

def plot_embeddings(embeddings, window_info, method='pca', window_size=None, stride=None):
    """Create visualization of embeddings using PCA or t-SNE."""
    
    # Reduce dimensionality to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = f"Fast-DROID Window Embeddings (PCA) - Window: {window_size}, Stride: {stride}"
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
        title = f"Fast-DROID Window Embeddings (t-SNE) - Window: {window_size}, Stride: {stride}"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract trajectory IDs and window indices
    traj_ids = [info[0] for info in window_info]
    window_indices = [info[1] for info in window_info]
    
    # Create scatter plot with color coding by window index (segment number)
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=window_indices, #cmap='plasma', 
                        alpha=0.6, s=30)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Window/Segment Index', rotation=270, labelpad=15)
    
    # Add labels for some windows
    n_labels = min(20, len(traj_ids))
    label_indices = np.linspace(0, len(traj_ids)-1, n_labels, dtype=int)
    for i in label_indices:
        ax.annotate(f'T{traj_ids[i]}-W{window_indices[i]}', 
                   (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   fontsize=6, alpha=0.5)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main(window_size=20, stride=10):
    """
    Main function to create embeddings with sliding windows in Cartesian space.
    
    Args:
        window_size: Size of the sliding window (number of timesteps)
        stride: Stride for the sliding window (overlap = window_size - stride)
    """
    # Dataset path
    dataset_path = "/home/kushtimusprime/ur5_policy_training/data_tiger_picking_90_good_and_bad_cleaned"
    
    print(f"Configuration: Window Size = {window_size}, Stride = {stride}")
    print(f"Overlap = {window_size - stride} timesteps")
    print(f"Using Cartesian space with gripper values")
    print()
    
    print("Loading Fast-DROID tokenizer...")
    tokenizer = load_fast_droid_tokenizer()
    
    print("Initializing robot model from URDF...")
    initialize_robot_model()
    
    # Find all HDF5 files
    h5_files = sorted(glob.glob(f"{dataset_path}/trajectory_*/joints.h5"))[:50]
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Extract data from HDF5 files and create windows
    print("Extracting data and creating windows...")
    all_windows = []
    window_info = []  # Store (trajectory_id, window_index) for each window
    
    for h5_path in tqdm(h5_files, desc="Processing trajectories"):
        trajectory = extract_data_from_h5(h5_path)
        if trajectory is not None and len(trajectory) >= window_size:
            # Extract trajectory ID from path
            traj_id = int(Path(h5_path).parent.name.split('_')[-1])
            
            # Create windows for this trajectory (always converts to Cartesian)
            windows = create_windows(trajectory, window_size, stride)
            
            # Add windows and their info
            for window_idx, window in enumerate(windows):
                all_windows.append(window)
                window_info.append((traj_id, window_idx))
    
    print(f"Created {len(all_windows)} windows from trajectories")
    
    # Create embeddings for all windows
    print("Creating embeddings for windows...")
    embeddings = create_embeddings_from_windows(all_windows, tokenizer)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Normalize embeddings for better visualization
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # PCA plot
    fig_pca = plot_embeddings(embeddings_normalized, window_info, method='pca', 
                              window_size=window_size, stride=stride)
    fig_pca.savefig(f'fast_droid_window_embeddings_cartesian_pca_w{window_size}_s{stride}.png', 
                    dpi=150, bbox_inches='tight')
    print(f"Saved PCA plot to fast_droid_window_embeddings_cartesian_pca_w{window_size}_s{stride}.png")
    
    # t-SNE plot (if dataset is not too large)
    if len(embeddings) < 5000:
        fig_tsne = plot_embeddings(embeddings_normalized, window_info, method='tsne',
                                  window_size=window_size, stride=stride)
        fig_tsne.savefig(f'fast_droid_window_embeddings_cartesian_tsne_w{window_size}_s{stride}.png', 
                        dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to fast_droid_window_embeddings_cartesian_tsne_w{window_size}_s{stride}.png")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total trajectories processed: {len(h5_files)}")
    print(f"Total windows created: {len(all_windows)}")
    print(f"Average windows per trajectory: {len(all_windows) / len(h5_files):.2f}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create embeddings with sliding windows in Cartesian space')
    parser.add_argument('--window_size', type=int, default=10, 
                        help='Size of the sliding window (default: 30)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for the sliding window (default: 15)')
    args = parser.parse_args()
    
    main(window_size=args.window_size, stride=args.stride)