"""
Fast-DROID + DINOv3 embedding module for trajectory representation.

This module provides embedding functionality using:
- Fast-DROID tokenizer for action sequences
- DINOv3 for visual features (when videos available)
- End effector poses directly from H5 files
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


class FastDROIDEmbedder:
    """
    Embedding generator using Fast-DROID tokenizer and DINOv3 features.

    This class handles:
    - Loading end effector poses from H5 files
    - Creating sliding windows
    - Tokenizing action sequences with Fast-DROID
    - Extracting visual features with DINOv3 (optional)
    - Combining all features into embeddings
    """

    def __init__(
        self,
        window_size: int = 15,
        stride: int = 1,
        use_dino: bool = True,
        max_token_length: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Fast-DROID embedder.

        Args:
            window_size: Size of sliding window
            stride: Stride for sliding window
            use_dino: If True, extract DINOv3 visual features
            max_token_length: Fixed maximum token length (for consistent embeddings)
            device: Device to use for DINOv3 model
        """
        self.window_size = window_size
        self.stride = stride
        self.use_dino = use_dino
        self.device = device
        self.max_token_length = max_token_length  # Fixed max token length

        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()

        # Initialize DINOv3 model if needed
        self.dino_model = None
        self.dino_processor = None
        if use_dino and CV2_AVAILABLE:
            self._init_dino_model()

    def _load_tokenizer(self):
        """Load Fast-DROID tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained("physical-intelligence/fast")
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

    def load_proprioception(self, h5_path: Path) -> Optional[np.ndarray]:
        """
        Load end effector poses from H5 file.

        Returns:
            Array of shape (N, 14) containing:
            - Left end effector: position (3) + orientation (4) = 7 values
            - Right end effector: position (3) + orientation (4) = 7 values
        """
        if not h5_path.exists():
            return None

        try:
            with h5py.File(h5_path, 'r') as f:
                # Load end effector poses from AgiBotWorld format
                # /state/end/position: [N, 2, 3] - left [:, 0, :], right [:, 1, :], xyz in meters
                # /state/end/orientation: [N, 2, 4] - left [:, 0, :], right [:, 1, :], quaternion xyzw
                if 'state' in f and 'end' in f['state']:
                    position = np.array(f['state/end/position'])  # Shape: (N, 2, 3)
                    orientation = np.array(f['state/end/orientation'])  # Shape: (N, 2, 4)

                    # Flatten to (N, 14): [left_pos(3), left_ori(4), right_pos(3), right_ori(4)]
                    n_timesteps = position.shape[0]
                    ee_poses = np.zeros((n_timesteps, 14))
                    ee_poses[:, 0:3] = position[:, 0, :]  # Left position
                    ee_poses[:, 3:7] = orientation[:, 0, :]  # Left orientation
                    ee_poses[:, 7:10] = position[:, 1, :]  # Right position
                    ee_poses[:, 10:14] = orientation[:, 1, :]  # Right orientation

                    return ee_poses
                else:
                    raise ValueError("H5 file does not contain end effector pose data at 'state/end'")

        except Exception as e:
            print(f"Warning: Failed to load {h5_path}: {e}")
            return None

    def create_windows(self, trajectory: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Create sliding windows from trajectory.

        Args:
            trajectory: Trajectory data (n_timesteps, 14)  - end effector poses

        Returns:
            List of (window, center_frame_idx) tuples
        """
        windows = []
        n_timesteps = len(trajectory)

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
        """Extract DINOv3 features from a video frame using PyAV for AV1 support."""
        if not self.use_dino or not CV2_AVAILABLE or self.dino_model is None:
            raise RuntimeError("DINO feature extraction is not available")

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            import av
        except ImportError:
            raise ImportError("PyAV is required for video reading. Install with: pip install av")

        try:
            # Open video with PyAV (supports AV1 codec)
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Check frame index
            if stream.frames > 0 and frame_idx >= stream.frames:
                container.close()
                raise ValueError(f"Frame index {frame_idx} out of range (video has {stream.frames} frames)")

            # Seek to approximate position first for efficiency
            if frame_idx > 0:
                # Calculate timestamp for seeking
                time_base = float(stream.time_base)
                avg_frame_rate = float(stream.average_rate)
                if avg_frame_rate > 0:
                    timestamp = int(frame_idx / avg_frame_rate / time_base)
                    container.seek(timestamp, stream=stream)

            # Decode frames until we reach the target
            current_idx = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if current_idx == frame_idx:
                        # Convert frame to numpy array (RGB format)
                        img = frame.to_ndarray(format='rgb24')
                        container.close()

                        # Convert to PIL Image
                        image = Image.fromarray(img)

                        # Process image with DINO
                        inputs = self.dino_processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.dino_model(**inputs)
                            features = outputs.last_hidden_state[:, 0, :]

                        return features.cpu().numpy().flatten()

                    current_idx += 1
                    if current_idx > frame_idx:
                        break

            # If we get here, frame wasn't found
            container.close()
            raise RuntimeError(f"Failed to read frame {frame_idx} from video: {video_path}")

        except Exception as e:
            if 'container' in locals():
                container.close()
            raise RuntimeError(f"Error reading video {video_path} frame {frame_idx}: {str(e)}") from e

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

        # Use fixed max_token_length if provided, otherwise compute it
        if self.max_token_length is not None:
            max_token_length = self.max_token_length
        else:
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

                # Only compute max if not using fixed length
                if self.max_token_length is None:
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
    use_dino: bool = True,
    max_token_length: int = 600
):
    """
    Create an embedding function for use with the benchmark.

    Args:
        window_size: Size of sliding window
        stride: Stride for sliding window
        use_dino: If True, extract visual features
        max_token_length: Fixed maximum token length for consistent embeddings

    Returns:
        Function that takes TrajectoryData and returns embeddings
    """
    embedder = FastDROIDEmbedder(
        window_size=window_size,
        stride=stride,
        use_dino=use_dino,
        max_token_length=max_token_length
    )

    def embedding_fn(trajectory_data):
        """Embedding function for trajectory data."""
        # Get paths
        h5_path = getattr(trajectory_data, 'proprioception_path', None)

        video_paths = getattr(trajectory_data, 'video_paths', None)
        video_path = None
        if video_paths:
            # Try to get head camera or wrist camera first
            video_path = video_paths.get('head_color') or video_paths.get('hand_left_color') or list(video_paths.values())[0]

        return embedder.embed_trajectory(trajectory_data, h5_path, video_path)

    return embedding_fn
