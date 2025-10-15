"""
Fast-DROID + DINOv3 embedding module for trajectory representation.

This module provides embedding functionality using:
- Fast-DROID tokenizer for action sequences
- DINOv3 for visual features (when videos available)
- Forward kinematics for Cartesian space conversion (when robot model available)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import h5py

try:
    import cv2
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Video features will be disabled.")

try:
    import yourdfpy
    URDF_AVAILABLE = True
except ImportError:
    URDF_AVAILABLE = False
    print("Warning: yourdfpy not available. Forward kinematics will be disabled.")


class FastDROIDEmbedder:
    """
    Embedding generator using Fast-DROID tokenizer and DINOv3 features.

    This class handles:
    - Loading proprioceptive data from H5 files
    - Converting joint space to Cartesian space (optional)
    - Creating sliding windows
    - Tokenizing action sequences with Fast-DROID
    - Extracting visual features with DINOv3 (optional)
    - Combining all features into embeddings
    """

    def __init__(
        self,
        window_size: int = 15,
        stride: int = 1,
        use_cartesian: bool = True,
        use_dino: bool = True,
        urdf_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Fast-DROID embedder.

        Args:
            window_size: Size of sliding window
            stride: Stride for sliding window
            use_cartesian: If True, convert joint space to Cartesian space
            use_dino: If True, extract DINOv3 visual features
            urdf_path: Path to URDF file for forward kinematics
            device: Device to use for DINOv3 model
        """
        self.window_size = window_size
        self.stride = stride
        self.use_cartesian = use_cartesian
        self.use_dino = use_dino
        self.urdf_path = urdf_path
        self.device = device

        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()

        # Initialize DINOv3 model if needed
        self.dino_model = None
        self.dino_processor = None
        if use_dino and CV2_AVAILABLE:
            self._init_dino_model()

        # Initialize robot model if needed
        self.robot_model = None
        if use_cartesian and URDF_AVAILABLE and urdf_path:
            self._init_robot_model(urdf_path)

    def _load_tokenizer(self):
        """Load Fast-DROID tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained("KarlP/fast-droid")
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'eos_token': '[EOS]',
            'bos_token': '[BOS]',
            'unk_token': '[UNK]'
        })
        return tokenizer

    def _init_dino_model(self):
        """Initialize DINOv3 model."""
        try:
            model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            self.dino_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.dino_model.eval()
            self.dino_model = self.dino_model.to(self.device)

            self.dino_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"Loaded DINOv3 model on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load DINOv3 model: {e}")
            self.use_dino = False

    def _init_robot_model(self, urdf_path: str):
        """Initialize robot model from URDF."""
        try:
            self.robot_model = yourdfpy.URDF.load(urdf_path)
            print(f"Loaded robot model from {urdf_path}")
        except Exception as e:
            print(f"Warning: Failed to load robot model: {e}")
            self.use_cartesian = False

    def load_proprioception(self, h5_path: Path) -> Optional[np.ndarray]:
        """Load proprioceptive data from H5 file."""
        if not h5_path.exists():
            return None

        try:
            with h5py.File(h5_path, 'r') as f:
                # Try AgiBotWorld format first
                if 'state' in f and 'joint' in f['state'] and 'position' in f['state/joint']:
                    return np.array(f['state/joint/position'])
                # Try simple format
                elif 'joint_states' in f:
                    return np.array(f['joint_states'])
                # Try action format
                elif 'action' in f and 'joint' in f['action'] and 'position' in f['action/joint']:
                    return np.array(f['action/joint/position'])
                else:
                    # Fallback: try first key
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]]
                        if hasattr(data, 'shape'):
                            return np.array(data)
            return None
        except Exception as e:
            print(f"Warning: Failed to load {h5_path}: {e}")
            return None

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Convert joint angles to Cartesian poses using forward kinematics.

        Args:
            joint_angles: (n_timesteps, n_joints) array

        Returns:
            (n_timesteps, 7) array with [x, y, z, roll, pitch, yaw, gripper]
        """
        if not self.use_cartesian or self.robot_model is None:
            return joint_angles

        n_timesteps = joint_angles.shape[0]
        gripper_values = joint_angles[:, -1] / 255.0 if joint_angles.shape[1] >= 7 else np.zeros(n_timesteps)
        cartesian_poses = np.zeros((n_timesteps, 7))

        joint_names = [joint.name for joint in self.robot_model.joint_map.values()
                      if joint.type in ['revolute', 'continuous']]
        n_joints_to_use = min(6, joint_angles.shape[1], len(joint_names))

        for t in range(n_timesteps):
            try:
                cfg = {joint_names[i]: float(joint_angles[t, i])
                      for i in range(n_joints_to_use)}
                self.robot_model.update_cfg(cfg)

                # Try common end-effector link names
                ee_link_names = ['tool0', 'ee_link', 'wrist_3_link', 'robotiq_85_base_link']
                ee_matrix = None

                for ee_name in ee_link_names:
                    if ee_name in self.robot_model.link_map:
                        ee_matrix = self.robot_model.get_transform(ee_name, 'base_link')
                        break

                if ee_matrix is None:
                    link_names = list(self.robot_model.link_map.keys())
                    ee_matrix = self.robot_model.get_transform(link_names[-1], 'base_link')

                # Extract position
                cartesian_poses[t, 0:3] = ee_matrix[0:3, 3]

                # Extract orientation (Euler angles)
                R = ee_matrix[0:3, 0:3]
                sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

                if sy >= 1e-6:
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

    def create_windows(self, trajectory: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Create sliding windows from trajectory.

        Args:
            trajectory: Trajectory data (n_timesteps, n_features)

        Returns:
            List of (window, center_frame_idx) tuples
        """
        windows = []
        n_timesteps = len(trajectory)

        # Convert to Cartesian if needed
        if self.use_cartesian:
            trajectory = self.forward_kinematics(trajectory)

        # Create windows
        for start_idx in range(0, n_timesteps - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = trajectory[start_idx:end_idx]
            center_frame_idx = start_idx + self.window_size // 2
            windows.append((window, center_frame_idx))

        # Handle last window
        if n_timesteps > self.window_size and (n_timesteps - self.window_size) % self.stride != 0:
            window = trajectory[-self.window_size:]
            center_frame_idx = n_timesteps - self.window_size // 2
            windows.append((window, center_frame_idx))

        return windows

    def extract_dino_features(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Extract DINOv3 features from a video frame."""
        if not self.use_dino or not CV2_AVAILABLE or self.dino_model is None:
            return None

        if not video_path.exists():
            return None

        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            # Process image
            inputs = self.dino_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]

            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Warning: Failed to extract DINO features: {e}")
            return None

    def create_embeddings(
        self,
        windows: List[Tuple[np.ndarray, int]],
        video_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create embeddings from windows using Fast-DROID tokenizer and DINOv3.

        Args:
            windows: List of (window, center_frame_idx) tuples
            video_path: Optional path to video for visual features

        Returns:
            Embeddings array of shape (n_windows, embedding_dim)
        """
        embeddings = []
        all_tokens = []
        max_token_length = 0

        # First pass: tokenize all windows
        for window, _ in tqdm(windows, desc="Tokenizing windows", leave=False):
            if len(window) > 1:
                # Compute actions (differences)
                actions = np.diff(window, axis=0)
                action_str = " ".join([f"{x:.4f}" for x in actions.flatten()])

                # Tokenize
                tokens = self.tokenizer(action_str, truncation=False, return_tensors='np')
                token_ids = tokens['input_ids'].flatten()
                all_tokens.append(token_ids)
                max_token_length = max(max_token_length, len(token_ids))
            else:
                all_tokens.append(np.array([]))

        # Second pass: create embeddings
        for idx, (window, center_frame_idx) in enumerate(tqdm(windows, desc="Creating embeddings", leave=False)):
            tokens = all_tokens[idx]

            if len(window) > 1 and len(tokens) > 0:
                # Pad/truncate tokens
                if len(tokens) < max_token_length:
                    pad_id = self.tokenizer.pad_token_id or 0
                    token_embedding = np.pad(tokens, (0, max_token_length - len(tokens)),
                                            constant_values=pad_id)
                else:
                    token_embedding = tokens[:max_token_length]

                # Statistical features
                window_features = np.concatenate([
                    window.mean(axis=0),
                    window.std(axis=0),
                    window.min(axis=0),
                    window.max(axis=0),
                    np.percentile(window, 25, axis=0),
                    np.percentile(window, 75, axis=0)
                ])

                # DINOv3 features
                dino_features = None
                if self.use_dino and video_path is not None:
                    dino_features = self.extract_dino_features(video_path, center_frame_idx)

                # Combine features
                if dino_features is not None:
                    combined = np.concatenate([token_embedding, window_features, dino_features])
                else:
                    combined = np.concatenate([token_embedding, window_features])
            else:
                # Empty window
                feature_dim = max_token_length + 6 * window.shape[1]
                if self.use_dino:
                    feature_dim += 1024  # DINOv3 ViT-L dimension
                combined = np.zeros(feature_dim)

            embeddings.append(combined)

        return np.array(embeddings)

    def embed_trajectory(
        self,
        trajectory_data,
        h5_path: Optional[Path] = None,
        video_path: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """
        Create embeddings for a complete trajectory.

        Args:
            trajectory_data: TrajectoryData object
            h5_path: Path to H5 file with proprioception
            video_path: Path to video file

        Returns:
            Embeddings array or None if no data available
        """
        # Load proprioception
        if h5_path is None and hasattr(trajectory_data, 'proprioception_path'):
            h5_path = trajectory_data.proprioception_path

        if h5_path is None:
            print(f"Warning: No proprioception data for {trajectory_data.trajectory_id}")
            return None

        proprio = self.load_proprioception(h5_path)
        if proprio is None:
            return None

        # Create windows
        windows = self.create_windows(proprio)
        if not windows:
            return None

        # Create embeddings
        embeddings = self.create_embeddings(windows, video_path)

        return embeddings


def load_fast_droid_tokenizer():
    """Load the Fast-DROID tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("KarlP/fast-droid")
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'bos_token': '[BOS]',
        'unk_token': '[UNK]'
    })
    return tokenizer


def create_embedding_function(
    window_size: int = 15,
    stride: int = 1,
    use_cartesian: bool = True,
    use_dino: bool = True,
    urdf_path: Optional[str] = None
):
    """
    Create an embedding function for use with the benchmark.

    Args:
        window_size: Size of sliding window
        stride: Stride for sliding window
        use_cartesian: If True, use Cartesian space
        use_dino: If True, extract visual features
        urdf_path: Path to URDF file

    Returns:
        Function that takes TrajectoryData and returns embeddings
    """
    embedder = FastDROIDEmbedder(
        window_size=window_size,
        stride=stride,
        use_cartesian=use_cartesian,
        use_dino=use_dino,
        urdf_path=urdf_path
    )

    def embedding_fn(trajectory_data):
        """Embedding function for trajectory data."""
        # Get paths
        h5_path = getattr(trajectory_data, 'proprioception_path', None)

        video_paths = getattr(trajectory_data, 'video_paths', None)
        video_path = None
        if video_paths:
            # Try to get wrist camera first
            video_path = video_paths.get('wrist') or video_paths.get('exo') or list(video_paths.values())[0]

        return embedder.embed_trajectory(trajectory_data, h5_path, video_path)

    return embedding_fn
