"""
Ray-based parallel processing for trajectory data.

This module provides Ray remote functions and actors for parallel processing of:
- Video frame extraction
- DINO feature extraction
- Proprioception data loading
- Dataset building
"""

import ray
import torch
import numpy as np
import av
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote(num_gpus=0.25)  # Allocate partial GPU per task
class TrajectoryProcessor:
    """Ray actor for processing trajectories with persistent DINO model."""

    def __init__(
        self,
        extract_dino: bool = True,
        extract_frames: bool = False,
        dino_model_name: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        target_fps: float = 5.0,
        device: str = None,
        frames_dir: str = None
    ):
        """Initialize the processor with optional DINO model."""
        self.extract_dino = extract_dino
        self.extract_frames = extract_frames
        self.target_fps = target_fps
        self.dino_model_name = dino_model_name
        self.frames_dir = Path(frames_dir) if frames_dir else None

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize DINO model if needed
        self.dino_model = None
        self.dino_processor = None
        if extract_dino:
            self._init_dino_model()

    def _init_dino_model(self):
        """Initialize DINOv3 model for feature extraction."""
        try:
            from transformers import AutoModel, AutoProcessor

            # Load DINOv3 model from HuggingFace
            self.dino_model = AutoModel.from_pretrained(
                self.dino_model_name,
                trust_remote_code=True
            )
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()

            # Load processor for preprocessing
            self.dino_processor = AutoProcessor.from_pretrained(
                self.dino_model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not initialize DINO model: {e}")
            self.extract_dino = False

    def process_trajectory(self, trajectory_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single trajectory."""
        try:
            trajectory_id = trajectory_data['trajectory_id']
            task_id = trajectory_data['task_id']
            episode_id = trajectory_data.get('episode_id', 0)

            # Extract proprioception if path is provided
            proprio_data = None
            proprio_timestamps = None
            if 'proprioception_path' in trajectory_data and trajectory_data['proprioception_path']:
                proprio_data, proprio_timestamps = self._extract_proprioception(
                    Path(trajectory_data['proprioception_path'])
                )

            # Extract visual features if video paths are provided
            dino_features = None
            image_paths = None

            # Check if DINO/frame extraction is requested
            if self.extract_dino or self.extract_frames:
                # Validate video paths exist
                if 'video_paths' not in trajectory_data or not trajectory_data['video_paths']:
                    raise ValueError(
                        f"Trajectory {trajectory_id} has no video_paths. "
                        f"Cannot extract DINO features or frames without videos. "
                        f"Please ensure video files are available in the source data."
                    )

                video_paths = trajectory_data['video_paths']
                if not isinstance(video_paths, dict) or 'head_color' not in video_paths:
                    raise ValueError(
                        f"Trajectory {trajectory_id} missing 'head_color' camera in video_paths. "
                        f"Available cameras: {list(video_paths.keys()) if isinstance(video_paths, dict) else 'none'}"
                    )

                video_path = Path(video_paths['head_color'])

                # Validate video file exists
                if not video_path.exists():
                    raise FileNotFoundError(
                        f"Video file not found for trajectory {trajectory_id}: {video_path}\n"
                        f"Please ensure video files are available at the correct location."
                    )

                if not video_path.is_file():
                    raise ValueError(
                        f"Video path is not a file for trajectory {trajectory_id}: {video_path}"
                    )

                # Extract features from validated video path
                if self.extract_dino:
                    dino_features = self._extract_dino_features(video_path)
                if self.extract_frames:
                    image_paths = self._extract_and_save_frames(video_path, trajectory_id)

            # If not extracting visual features, still store video paths if available
            elif 'video_paths' in trajectory_data and trajectory_data['video_paths']:
                # Just store the video paths without validation (for reference only)
                pass

            # Create frame labels from segments
            frame_labels = self._create_frame_labels(trajectory_data.get('segments', []))

            # Return processed data
            result = {
                'trajectory_id': trajectory_id,
                'task_id': task_id,
                'episode_id': episode_id,
                'task_name': trajectory_data.get('task_name', ''),
                'segments': trajectory_data.get('segments', []),
                'frame_labels': frame_labels.tolist() if frame_labels is not None else [],
                'proprioception': proprio_data.tolist() if proprio_data is not None else None,
                'proprio_timestamps': proprio_timestamps.tolist() if proprio_timestamps is not None else None,
                'dino_features': dino_features.tolist() if dino_features is not None else None,
                'image_paths': image_paths,  # List of relative path strings or None
                'video_paths': trajectory_data.get('video_paths', {}),
                'total_frames': trajectory_data.get('total_frames', 0),
                'fps': self.target_fps,  # Store actual target fps used for subsampling
                'metadata': trajectory_data.get('metadata', {})
            }

            return result

        except Exception as e:
            traj_id = trajectory_data.get('trajectory_id', 'unknown')
            print(f"\n❌ ERROR: Failed to process trajectory {traj_id}")
            print(f"   Reason: {type(e).__name__}: {e}")
            return None

    def _create_frame_labels(self, segments: List[Dict]) -> np.ndarray:
        """Create frame-level labels from segment annotations."""
        if not segments:
            return np.array([])

        max_frame = max(seg['end_frame'] for seg in segments)
        labels = np.zeros(max_frame, dtype=int)

        for seg in segments:
            labels[seg['start_frame']:seg['end_frame']] = seg.get('segment_id', 0) + 1

        return labels

    def _compute_frequency(self, timestamps: np.ndarray) -> float:
        """Compute sampling frequency from timestamps."""
        if timestamps is None or len(timestamps) < 2:
            return 30.0  # Default to 30 Hz if no timestamps
        
        # Calculate time differences and use median to be robust to outliers
        dt = np.diff(timestamps)
        median_dt = np.median(dt)
        
        if median_dt <= 0:
            return 30.0  # Fallback to default
        
        frequency = 1.0 / median_dt
        return float(frequency)

    def _extract_proprioception(self, proprio_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract and subsample proprioception data from H5 file to match target_fps.
        
        This ensures that proprioception data is aligned with visual data (images/DINO features)
        which are extracted at target_fps. The actual proprioception frequency is computed from
        timestamps, and the data is subsampled accordingly.
        
        Returns:
            Tuple of (subsampled_proprio_data, subsampled_timestamps)
        """
        if not proprio_path.exists():
            return None, None

        try:
            with h5py.File(proprio_path, 'r') as f:
                proprio_data = None
                timestamps = None

                if 'state' in f and 'joint' in f['state'] and 'position' in f['state']['joint']:
                    proprio_data = f['state']['joint']['position'][:]

                if 'timestamp' in f:
                    timestamps = f['timestamp'][:]
                elif 'timestamps' in f:
                    timestamps = f['timestamps'][:]

                # Return None if no data
                if proprio_data is None:
                    return None, None

                # Subsample to match target_fps
                if timestamps is not None and len(timestamps) == len(proprio_data):
                    # Compute actual frequency
                    actual_freq = self._compute_frequency(timestamps)
                    sample_rate = max(1, int(actual_freq / self.target_fps))
                    
                    # Subsample data and timestamps
                    indices = np.arange(0, len(proprio_data), sample_rate)
                    proprio_data = proprio_data[indices]
                    timestamps = timestamps[indices]
                    
                    # Debug info (optional)
                    # print(f"Proprioception: {actual_freq:.1f} Hz -> {self.target_fps} Hz (sample_rate={sample_rate})")
                else:
                    # If no timestamps or mismatch, assume 30 Hz and subsample
                    assumed_freq = 30.0
                    sample_rate = max(1, int(assumed_freq / self.target_fps))
                    indices = np.arange(0, len(proprio_data), sample_rate)
                    proprio_data = proprio_data[indices]
                    if timestamps is not None:
                        # Try to subsample timestamps too if available
                        if len(timestamps) >= len(indices):
                            timestamps = timestamps[indices]

                return proprio_data, timestamps
        except Exception as e:
            print(f"Warning: Could not load proprioception from {proprio_path}: {e}")
            return None, None

    def _extract_dino_features(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract DINOv3 features from video using PyAV."""
        if not self.dino_model:
            return None

        try:
            from PIL import Image as PILImage

            container = av.open(str(video_path))
            stream = container.streams.video[0]

            fps = float(stream.average_rate) if stream.average_rate else 30.0
            sample_rate = max(1, int(fps / self.target_fps))

            features = []
            frame_idx = 0

            with torch.no_grad():
                for frame in container.decode(stream):
                    if frame_idx % sample_rate == 0:
                        # Convert to RGB numpy array
                        img = frame.to_ndarray(format='rgb24')

                        # Convert to PIL Image for processor
                        pil_img = PILImage.fromarray(img)

                        # Process image with DINOv3 processor
                        inputs = self.dino_processor(images=pil_img, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        # Extract features using [CLS] token
                        outputs = self.dino_model(**inputs)
                        feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        features.append(feature.squeeze())

                    frame_idx += 1

            container.close()

            if features:
                return np.stack(features)
            return None

        except Exception as e:
            print(f"Warning: Could not extract DINO features from {video_path}: {e}")
            return None

    def _extract_and_save_frames(self, video_path: Path, trajectory_id: str) -> Optional[List[str]]:
        """Extract frames from video and save to disk, returning relative paths."""
        if self.frames_dir is None:
            print("Warning: frames_dir not set, cannot save frames")
            return None
            
        try:
            from PIL import Image as PILImage
            
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            fps = float(stream.average_rate) if stream.average_rate else 30.0
            sample_rate = max(1, int(fps / self.target_fps))

            # Create trajectory-specific directory
            traj_frames_dir = self.frames_dir / trajectory_id
            traj_frames_dir.mkdir(parents=True, exist_ok=True)

            image_paths = []
            frame_idx = 0
            saved_idx = 0

            for frame in container.decode(stream):
                if frame_idx % sample_rate == 0:
                    # Convert to RGB numpy array
                    img_array = frame.to_ndarray(format='rgb24')
                    
                    # Save image as PNG
                    img_path = traj_frames_dir / f"frame_{saved_idx:06d}.png"
                    if not img_path.exists():
                        pil_img = PILImage.fromarray(img_array)
                        pil_img.save(img_path)
                    
                    # Store relative path from frames_dir
                    rel_path = img_path.relative_to(self.frames_dir.parent)
                    image_paths.append(str(rel_path))
                    saved_idx += 1

                frame_idx += 1

            container.close()

            return image_paths if image_paths else None

        except Exception as e:
            print(f"Warning: Could not extract and save frames from {video_path}: {e}")
            return None


@ray.remote
def process_trajectory_batch(
    trajectories: List[Dict[str, Any]],
    processor_config: Dict[str, Any]
) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of trajectories without DINO (for CPU-only processing)."""
    results = []

    for traj_data in trajectories:
        try:
            # Simple processing without DINO
            trajectory_id = traj_data['trajectory_id']
            task_id = traj_data['task_id']

            # Create frame labels
            segments = traj_data.get('segments', [])
            if segments:
                max_frame = max(seg['end_frame'] for seg in segments)
                frame_labels = np.zeros(max_frame, dtype=int)
                for seg in segments:
                    frame_labels[seg['start_frame']:seg['end_frame']] = seg.get('segment_id', 0) + 1
            else:
                frame_labels = np.array([])

            result = {
                'trajectory_id': trajectory_id,
                'task_id': task_id,
                'episode_id': traj_data.get('episode_id', 0),
                'task_name': traj_data.get('task_name', ''),
                'segments': segments,
                'frame_labels': frame_labels.tolist(),
                'proprioception': None,
                'proprio_timestamps': None,
                'dino_features': None,
                'image_paths': None,
                'video_paths': traj_data.get('video_paths', {}),
                'total_frames': traj_data.get('total_frames', 0),
                'fps': processor_config.get('target_fps', 5.0),
                'metadata': traj_data.get('metadata', {})
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing trajectory {traj_data.get('trajectory_id', 'unknown')}: {e}")
            results.append(None)

    return results


class RayDatasetBuilder:
    """Dataset builder using Ray for parallel processing."""

    def __init__(
        self,
        num_workers: int = None,
        num_gpus_per_worker: float = 0.25,
        frames_dir: str = None,
        **processor_config
    ):
        """
        Initialize Ray dataset builder.

        Args:
            num_workers: Number of Ray workers (None = auto-detect)
            num_gpus_per_worker: GPUs allocated per worker
            frames_dir: Directory to save extracted frames (required if extract_frames=True)
            **processor_config: Configuration passed to TrajectoryProcessor
        """
        self.num_workers = num_workers
        self.num_gpus_per_worker = num_gpus_per_worker
        self.frames_dir = frames_dir
        self.processor_config = processor_config
        self.processor_config['frames_dir'] = frames_dir

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            # Determine resources
            num_cpus = os.cpu_count()
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

            if self.num_workers is None:
                # Auto-detect optimal number of workers
                if processor_config.get('extract_dino', False) and num_gpus > 0:
                    # For DINO extraction, use GPU workers
                    self.num_workers = min(int(num_gpus / self.num_gpus_per_worker), num_cpus)
                else:
                    # For non-DINO processing, use CPU workers
                    self.num_workers = num_cpus

            print(f"Initializing Ray with {num_cpus} CPUs and {num_gpus} GPUs")
            print(f"Using {self.num_workers} workers")

            ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    def process_dataset(
        self,
        trajectories: List[Any],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process trajectories in parallel using Ray.

        Args:
            trajectories: List of trajectory objects or dicts
            batch_size: Batch size for processing

        Returns:
            List of processed trajectory dictionaries
        """
        # Convert trajectories to serializable format
        trajectory_dicts = []
        for traj in trajectories:
            if hasattr(traj, '__dict__'):
                # Convert object to dict
                traj_dict = {
                    'trajectory_id': traj.trajectory_id,
                    'task_id': traj.task_id,
                    'episode_id': traj.metadata.get('episode_id', 0) if hasattr(traj, 'metadata') else 0,
                    'task_name': traj.task_name if hasattr(traj, 'task_name') else '',
                    'segments': [seg.to_dict() if hasattr(seg, 'to_dict') else seg
                                for seg in (traj.segments if hasattr(traj, 'segments') else [])],
                    'proprioception_path': str(traj.proprioception_path) if hasattr(traj, 'proprioception_path') and traj.proprioception_path else None,
                    'video_paths': {k: str(v) for k, v in traj.video_paths.items()} if hasattr(traj, 'video_paths') and traj.video_paths else {},
                    'total_frames': traj.total_frames if hasattr(traj, 'total_frames') else 0,
                    'metadata': traj.metadata if hasattr(traj, 'metadata') else {}
                }
            else:
                traj_dict = traj
            trajectory_dicts.append(traj_dict)

        # Process with Ray
        if self.processor_config.get('extract_dino', False):
            # Use GPU actors for DINO processing
            results = self._process_with_actors(trajectory_dicts)
        else:
            # Use simple remote functions for CPU processing
            results = self._process_with_remote_functions(trajectory_dicts, batch_size)

        return results

    def _process_with_actors(self, trajectory_dicts: List[Dict]) -> List[Dict]:
        """Process using Ray actors (for DINO extraction)."""
        # Create worker actors
        actors = []
        for i in range(self.num_workers):
            actor = TrajectoryProcessor.remote(**self.processor_config)
            actors.append(actor)

        # Distribute work among actors
        futures = []
        for i, traj_dict in enumerate(trajectory_dicts):
            actor = actors[i % self.num_workers]
            future = actor.process_trajectory.remote(traj_dict)
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        # Filter out None results
        results = [r for r in results if r is not None]

        return results

    def _process_with_remote_functions(self, trajectory_dicts: List[Dict], batch_size: int) -> List[Dict]:
        """Process using remote functions (for non-DINO processing)."""
        # Split into batches
        batches = [trajectory_dicts[i:i+batch_size]
                  for i in range(0, len(trajectory_dicts), batch_size)]

        # Process batches in parallel
        futures = []
        for batch in batches:
            future = process_trajectory_batch.remote(batch, self.processor_config)
            futures.append(future)

        # Collect results
        batch_results = ray.get(futures)

        # Flatten results
        results = []
        for batch in batch_results:
            results.extend([r for r in batch if r is not None])

        return results

    def shutdown(self):
        """Shutdown Ray."""
        if ray.is_initialized():
            ray.shutdown()