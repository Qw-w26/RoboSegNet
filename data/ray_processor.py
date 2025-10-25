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
        dino_model_name: str = 'dinov2_vitb14',
        target_fps: float = 5.0,
        device: str = None
    ):
        """Initialize the processor with optional DINO model."""
        self.extract_dino = extract_dino
        self.extract_frames = extract_frames
        self.target_fps = target_fps
        self.dino_model_name = dino_model_name

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize DINO model if needed
        self.dino_model = None
        self.dino_transform = None
        if extract_dino:
            self._init_dino_model()

    def _init_dino_model(self):
        """Initialize DINO model for feature extraction."""
        try:
            from torchvision import transforms

            # Load DINOv2 model
            self.dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_model_name)
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()

            # DINO preprocessing
            self.dino_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
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
            frames = None

            if 'video_paths' in trajectory_data and trajectory_data['video_paths']:
                # Use first available camera
                video_paths = trajectory_data['video_paths']
                if isinstance(video_paths, dict) and len(video_paths) > 0:
                    camera_name = list(video_paths.keys())[0]
                    video_path = Path(video_paths[camera_name])

                    if video_path.exists():
                        if self.extract_dino:
                            dino_features = self._extract_dino_features(video_path)
                        if self.extract_frames:
                            frames = self._extract_frames(video_path)

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
                'images': frames,  # Will be list of numpy arrays or None
                'video_paths': trajectory_data.get('video_paths', {}),
                'total_frames': trajectory_data.get('total_frames', 0),
                'metadata': trajectory_data.get('metadata', {})
            }

            return result

        except Exception as e:
            print(f"Error processing trajectory {trajectory_data.get('trajectory_id', 'unknown')}: {e}")
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

    def _extract_proprioception(self, proprio_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract proprioception data from H5 file."""
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

                return proprio_data, timestamps
        except Exception as e:
            print(f"Warning: Could not load proprioception from {proprio_path}: {e}")
            return None, None

    def _extract_dino_features(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract DINO features from video using PyAV."""
        if not self.dino_model:
            return None

        try:
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

                        # Apply DINO preprocessing
                        frame_tensor = self.dino_transform(img).unsqueeze(0).to(self.device)

                        # Extract features
                        feature = self.dino_model(frame_tensor).cpu().numpy()
                        features.append(feature.squeeze())

                    frame_idx += 1

            container.close()

            if features:
                return np.stack(features)
            return None

        except Exception as e:
            print(f"Warning: Could not extract DINO features from {video_path}: {e}")
            return None

    def _extract_frames(self, video_path: Path) -> Optional[List[np.ndarray]]:
        """Extract frames from video as numpy arrays using PyAV."""
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            fps = float(stream.average_rate) if stream.average_rate else 30.0
            sample_rate = max(1, int(fps / self.target_fps))

            frames = []
            frame_idx = 0

            for frame in container.decode(stream):
                if frame_idx % sample_rate == 0:
                    # Convert to RGB numpy array
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)

                frame_idx += 1

            container.close()

            return frames if frames else None

        except Exception as e:
            print(f"Warning: Could not extract frames from {video_path}: {e}")
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
                'images': None,
                'video_paths': traj_data.get('video_paths', {}),
                'total_frames': traj_data.get('total_frames', 0),
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
        **processor_config
    ):
        """
        Initialize Ray dataset builder.

        Args:
            num_workers: Number of Ray workers (None = auto-detect)
            num_gpus_per_worker: GPUs allocated per worker
            **processor_config: Configuration passed to TrajectoryProcessor
        """
        self.num_workers = num_workers
        self.num_gpus_per_worker = num_gpus_per_worker
        self.processor_config = processor_config

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