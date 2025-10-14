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
import cv2
from PIL import Image
import torchvision.transforms as transforms
import ray

# Initialize URDF model globally
URDF_PATH = "/home/kushtimusprime/ur5_policy_training/urdf/ur5_with_robotiq_gripper.urdf"
robot_model = None

# Initialize DINOv2 model globally
dino_model = None
dino_transform = None

def initialize_robot_model():
    """Initialize the robot model from URDF."""
    global robot_model
    if robot_model is None:
        robot_model = yourdfpy.URDF.load(URDF_PATH)
    return robot_model

def initialize_dino_model():
    """Initialize DINOv3 model and transforms."""
    global dino_model, dino_transform
    if dino_model is None:
        from transformers import AutoModel, AutoProcessor
        
        # Load DINOv3 model from HuggingFace
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        dino_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        dino_model.eval()
        
        # Load the processor for preprocessing
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        if torch.cuda.is_available():
            dino_model = dino_model.cuda()
        
        # Use the processor's transform
        dino_transform = processor
        
    return dino_model, dino_transform

def extract_frame_from_video(video_path, frame_idx):
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def extract_dino_features(image):
    """Extract DINOv3 features from an image."""
    model, processor = initialize_dino_model()
    
    # Process image using the processor
    inputs = processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the CLS token features (first token)
        features = outputs.last_hidden_state[:, 0, :]
    
    return features.cpu().numpy().flatten()

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

def create_windows(trajectory, window_size, stride, use_joints=False):
    """Create sliding windows from a trajectory, optionally convert to Cartesian.
    
    Args:
        trajectory: Joint state trajectory
        window_size: Size of the sliding window
        stride: Stride for the sliding window
        use_joints: If True, use joint states directly without FK conversion
    
    Returns:
        List of tuples (window, center_frame_idx) where window is either joint or Cartesian space
    """
    windows = []
    n_timesteps = len(trajectory)
    
    # Create windows with the specified stride
    for start_idx in range(0, n_timesteps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = trajectory[start_idx:end_idx]
        
        # Optionally convert joint states to Cartesian poses
        if not use_joints:
            window = ur5_forward_kinematics(window)
        
        # Calculate center frame index
        center_frame_idx = start_idx + window_size // 2
        
        windows.append((window, center_frame_idx))
    
    # Handle the last window if there's remaining data
    if n_timesteps > window_size and (n_timesteps - window_size) % stride != 0:
        start_idx = n_timesteps - window_size
        window = trajectory[-window_size:]
        
        # Optionally convert joint states to Cartesian poses
        if not use_joints:
            window = ur5_forward_kinematics(window)
        
        # Calculate center frame index
        center_frame_idx = start_idx + window_size // 2
        
        windows.append((window, center_frame_idx))
    
    return windows

def create_embeddings_from_windows_with_dino(windows_list, tokenizer, dino_features_list=None):
    """Create embeddings from trajectory windows using tokenizer and pre-computed DINOv2 features.
    
    Args:
        windows_list: List of tuples (window, center_frame_idx) or just windows
        tokenizer: Tokenizer for action sequences
        dino_features_list: Pre-computed DINOv2 features for each window
    """
    embeddings = []
    max_token_length = None
    
    # Extract windows and frame indices
    if isinstance(windows_list[0], tuple):
        windows = [w[0] for w in windows_list]
        frame_indices = [w[1] for w in windows_list]
    else:
        windows = windows_list
        frame_indices = [None] * len(windows_list)
    
    # First pass: tokenize all windows to find max token length
    all_tokens = []
    for window in tqdm(windows, desc="Tokenizing windows"):
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
    for idx, (window, tokens) in tqdm(enumerate(zip(windows, all_tokens)), 
                               desc="Creating embeddings", 
                               total=len(windows)):
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
            if dino_features_list is not None and idx < len(dino_features_list):
                combined_embedding = np.concatenate([token_embedding, window_features, dino_features_list[idx]])
            else:
                combined_embedding = np.concatenate([token_embedding, window_features])
        else:
            # Handle edge case of very small windows
            if dino_features_list is not None:
                combined_embedding = np.zeros(max_token_length + 6 * window.shape[1] + 1024)  # DINOv3 ViT-L feature dimension
            else:
                combined_embedding = np.zeros(max_token_length + 6 * window.shape[1])
        
        embeddings.append(combined_embedding)
    
    return np.array(embeddings)

@ray.remote(num_gpus=0.1)
def process_trajectory(h5_path, window_size, stride, use_joints=False):
    """Process a single trajectory and return windows with embeddings.
    
    This function is designed to run in parallel using Ray.
    
    Args:
        h5_path: Path to the HDF5 file
        window_size: Size of the sliding window
        stride: Stride for the sliding window
        use_joints: If True, use joint states directly without FK conversion
    
    Returns:
        Tuple of (windows_with_indices, window_info, video_paths, traj_id)
    """
    # Initialize models locally for this worker
    import torch
    import torchvision.transforms as transforms
    import yourdfpy
    
    # Load robot model
    robot_model = yourdfpy.URDF.load(URDF_PATH)
    
    # Load DINOv3 model from HuggingFace - requires GPU
    from transformers import AutoModel, AutoProcessor
    
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    dino_model.eval()
    
    # Load the processor for preprocessing
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Must use GPU - fail if not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DINOv3 requires GPU.")
    
    dino_model = dino_model.cuda()
    
    # Helper function for forward kinematics (local to worker)
    def local_ur5_forward_kinematics(joint_angles):
        n_timesteps = joint_angles.shape[0]
        gripper_values = joint_angles[:, -1] / 255.0 if joint_angles.shape[1] >= 7 else np.zeros(n_timesteps)
        cartesian_poses = np.zeros((n_timesteps, 7))
        
        joint_names = [joint.name for joint in robot_model.joint_map.values() 
                      if joint.type in ['revolute', 'continuous']]
        n_joints_to_use = min(6, joint_angles.shape[1], len(joint_names))
        
        for t in range(n_timesteps):
            try:
                cfg = {joint_names[i]: float(joint_angles[t, i]) 
                      for i in range(n_joints_to_use)}
                robot_model.update_cfg(cfg)
                
                ee_link_names = ['tool0', 'ee_link', 'wrist_3_link', 'robotiq_85_base_link']
                ee_matrix = None
                
                for ee_name in ee_link_names:
                    if ee_name in robot_model.link_map:
                        ee_matrix = robot_model.get_transform(ee_name, 'base_link')
                        break
                
                if ee_matrix is None:
                    link_names = list(robot_model.link_map.keys())
                    ee_matrix = robot_model.get_transform(link_names[-1], 'base_link')
                
                cartesian_poses[t, 0:3] = ee_matrix[0:3, 3]
                R = ee_matrix[0:3, 0:3]
                
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
                cartesian_poses[t, 6] = gripper_values[t]
            except:
                continue
        
        return cartesian_poses
    
    # Helper function for extracting frame and DINO features
    def local_extract_dino_features(video_path, frame_idx):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            
            # Process image using the processor
            inputs = processor(images=image, return_tensors="pt")
            # Use GPU for DINOv3
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
                # Get the CLS token features (first token)
                features = outputs.last_hidden_state[:, 0, :]
            
            return features.cpu().numpy().flatten()
        return np.zeros(1024)  # DINOv3 ViT-L feature dimension
    
    # Extract trajectory data
    trajectory = extract_data_from_h5(h5_path)
    if trajectory is None or len(trajectory) < window_size:
        return None, None, None, None
    
    # Extract trajectory ID from path
    traj_id = int(Path(h5_path).parent.name.split('_')[-1])
    
    # Get video path
    video_path = Path(h5_path).parent / "wrist.mp4"
    if not video_path.exists():
        video_path = Path(h5_path).parent / "exo.mp4"
    
    # Create windows
    windows_with_indices = []
    window_info = []
    video_paths = []
    dino_features_list = []
    
    n_timesteps = len(trajectory)
    
    # Create windows with the specified stride
    for start_idx in range(0, n_timesteps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = trajectory[start_idx:end_idx]
        
        # Optionally convert joint states to Cartesian poses
        if not use_joints:
            window = local_ur5_forward_kinematics(window)
        
        # Calculate center frame index
        center_frame_idx = start_idx + window_size // 2
        
        # Extract DINOv2 features for center frame
        if video_path.exists():
            dino_features = local_extract_dino_features(video_path, center_frame_idx)
        else:
            dino_features = np.zeros(1024)  # DINOv3 ViT-L feature dimension
        
        windows_with_indices.append((window, center_frame_idx))
        window_info.append((traj_id, len(windows_with_indices) - 1))
        video_paths.append(video_path if video_path.exists() else None)
        dino_features_list.append(dino_features)
    
    # Handle the last window if there's remaining data
    if n_timesteps > window_size and (n_timesteps - window_size) % stride != 0:
        start_idx = n_timesteps - window_size
        window = trajectory[-window_size:]
        
        # Optionally convert joint states to Cartesian poses
        if not use_joints:
            window = local_ur5_forward_kinematics(window)
        
        # Calculate center frame index
        center_frame_idx = start_idx + window_size // 2
        
        # Extract DINOv2 features for center frame
        if video_path.exists():
            dino_features = local_extract_dino_features(video_path, center_frame_idx)
        else:
            dino_features = np.zeros(1024)  # DINOv3 ViT-L feature dimension
        
        windows_with_indices.append((window, center_frame_idx))
        window_info.append((traj_id, len(windows_with_indices) - 1))
        video_paths.append(video_path if video_path.exists() else None)
        dino_features_list.append(dino_features)
    
    return windows_with_indices, window_info, video_paths, dino_features_list

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

def main(window_size=20, stride=10, use_joints=False):
    """
    Main function to create embeddings with sliding windows.
    
    Args:
        window_size: Size of the sliding window (number of timesteps)
        stride: Stride for the sliding window (overlap = window_size - stride)
        use_joints: If True, use joint states directly without FK conversion
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Dataset path
    dataset_path = "/home/kushtimusprime/ur5_policy_training/data_tiger_picking_90_good_and_bad_cleaned"
    
    print(f"Configuration: Window Size = {window_size}, Stride = {stride}")
    print(f"Overlap = {window_size - stride} timesteps")
    space_type = "joint space" if use_joints else "Cartesian space"
    print(f"Using {space_type} with gripper values and DINOv3 features")
    print()
    
    print("Loading Fast-DROID tokenizer...")
    tokenizer = load_fast_droid_tokenizer()
    
    # Find all HDF5 files
    h5_files = sorted(glob.glob(f"{dataset_path}/trajectory_*/joints.h5"))[:1]
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Process trajectories in parallel using Ray
    print("Processing trajectories in parallel with Ray...")
    
    # Submit all trajectory processing tasks to Ray
    futures = []
    for h5_path in h5_files:
        future = process_trajectory.remote(h5_path, window_size, stride, use_joints)
        futures.append(future)
    
    # Wait for all tasks to complete and collect results
    all_windows = []
    all_dino_features = []
    window_info = []
    
    for future in tqdm(futures, desc="Processing trajectories"):
        result = ray.get(future)
        windows_with_indices, info, video_paths, dino_features = result
        
        if windows_with_indices is not None:
            # Extract just the windows (without frame indices)
            windows = [w[0] for w in windows_with_indices]
            all_windows.extend(windows)
            all_dino_features.extend(dino_features)
            window_info.extend(info)
    
    print(f"Created {len(all_windows)} windows from trajectories")
    
    # Create embeddings for all windows
    print("Creating embeddings for windows with pre-computed DINOv3 features...")
    embeddings = create_embeddings_from_windows_with_dino(all_windows, tokenizer, dino_features_list=all_dino_features)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Normalize embeddings for better visualization
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # PCA plot
    space_suffix = "joint" if use_joints else "cartesian"
    fig_pca = plot_embeddings(embeddings_normalized, window_info, method='pca', 
                              window_size=window_size, stride=stride)
    fig_pca.savefig(f'fast_droid_window_embeddings_{space_suffix}_pca_w{window_size}_s{stride}.png', 
                    dpi=150, bbox_inches='tight')
    print(f"Saved PCA plot to fast_droid_window_embeddings_{space_suffix}_pca_w{window_size}_s{stride}.png")
    
    # t-SNE plot (if dataset is not too large)
    if len(embeddings) < 5000:
        fig_tsne = plot_embeddings(embeddings_normalized, window_info, method='tsne',
                                  window_size=window_size, stride=stride)
        fig_tsne.savefig(f'fast_droid_window_embeddings_{space_suffix}_tsne_w{window_size}_s{stride}.png', 
                        dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to fast_droid_window_embeddings_{space_suffix}_tsne_w{window_size}_s{stride}.png")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total trajectories processed: {len(h5_files)}")
    print(f"Total windows created: {len(all_windows)}")
    print(f"Average windows per trajectory: {len(all_windows) / len(h5_files):.2f}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"  - Token dimensions: varies")
    print(f"  - Window statistics: {6 * 7}")
    print(f"  - DINOv3 features: 1024")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create embeddings with sliding windows')
    parser.add_argument('--window_size', type=int, default=15, 
                        help='Size of the sliding window (default: 15)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for the sliding window (default: 1)')
    parser.add_argument('--use_joints', action='store_true',
                        help='Use joint states directly without forward kinematics')
    args = parser.parse_args()
    
    main(window_size=args.window_size, stride=args.stride, use_joints=args.use_joints)