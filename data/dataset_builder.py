"""
Dataset builder for creating comprehensive, self-contained datasets.

This module creates structured datasets with:
- Proprioception data
- DINO features (extracted from videos)
- Image/video paths and frames
- Task segmentation labels
- Metadata

The output format is compatible with HuggingFace datasets and can be directly
used by downstream training modules.
"""

import json
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pickle
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
import os

# Import video processing library
try:
    import av
except ImportError:
    raise ImportError("PyAV is required for video processing. Install it with: pip install av")

# Import Ray for parallel processing
import ray

from .base_dataset import TrajectoryData, SegmentAnnotation
from .agibot_dataset import AgiBotWorldDataset


@dataclass
class ProcessedTrajectory:
    """Processed trajectory with all features extracted."""

    # Identifiers
    trajectory_id: str
    task_id: int
    episode_id: int

    # Segmentation labels
    segments: List[Dict[str, Any]]  # List of segment annotations
    frame_labels: np.ndarray  # Frame-level labels (N,)

    # Proprioception
    proprioception: Optional[np.ndarray] = None  # (T, D) proprioceptive features
    proprio_timestamps: Optional[np.ndarray] = None  # (T,) timestamps

    # Visual features
    dino_features: Optional[np.ndarray] = None  # (T, D) DINO features
    images: Optional[List[np.ndarray]] = None  # List of image arrays
    video_paths: Optional[Dict[str, str]] = None  # Camera -> video path

    # Metadata
    task_name: Optional[str] = None
    total_frames: int = 0
    fps: float = 30.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.proprioception is not None:
            data['proprioception'] = self.proprioception.tolist()
        if self.proprio_timestamps is not None:
            data['proprio_timestamps'] = self.proprio_timestamps.tolist()
        if self.dino_features is not None:
            data['dino_features'] = self.dino_features.tolist()
        if self.frame_labels is not None:
            data['frame_labels'] = self.frame_labels.tolist()
        # Note: images are not serialized to JSON due to size
        if self.images is not None:
            data['num_images'] = len(self.images)
            data['images'] = None  # Don't save raw images to JSON
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedTrajectory':
        """Create from dictionary."""
        # Convert lists back to numpy arrays
        if 'proprioception' in data and data['proprioception'] is not None:
            data['proprioception'] = np.array(data['proprioception'])
        if 'proprio_timestamps' in data and data['proprio_timestamps'] is not None:
            data['proprio_timestamps'] = np.array(data['proprio_timestamps'])
        if 'dino_features' in data and data['dino_features'] is not None:
            data['dino_features'] = np.array(data['dino_features'])
        if 'frame_labels' in data and data['frame_labels'] is not None:
            data['frame_labels'] = np.array(data['frame_labels'])
        return cls(**data)


class DatasetBuilder:
    """Builds comprehensive datasets from raw trajectory data using Ray parallel processing."""

    def __init__(
        self,
        output_dir: Path,
        extract_dino: bool = True,
        extract_frames: bool = False,
        dino_model: str = 'dinov2_vitb14',
        target_fps: float = 5.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_workers: int = None
    ):
        """
        Initialize dataset builder with Ray parallel processing.

        Args:
            output_dir: Directory to save processed dataset
            extract_dino: Whether to extract DINO features
            extract_frames: Whether to extract individual frames
            dino_model: DINO model variant to use
            target_fps: Target FPS for frame extraction (if different from video)
            device: Device for DINO extraction
            num_workers: Number of Ray workers (None = auto-detect)
        """
        self.output_dir = Path(output_dir)
        self.extract_dino = extract_dino
        self.extract_frames = extract_frames
        self.dino_model_name = dino_model
        self.target_fps = target_fps
        self.device = device
        self.num_workers = num_workers

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = self.output_dir / 'features'
        self.features_dir.mkdir(exist_ok=True)

        if extract_frames:
            self.frames_dir = self.output_dir / 'frames'
            self.frames_dir.mkdir(exist_ok=True)

        # Initialize Ray processor
        from .ray_processor import RayDatasetBuilder
        self.ray_processor = RayDatasetBuilder(
            num_workers=num_workers,
            extract_dino=extract_dino,
            extract_frames=extract_frames,
            dino_model_name=dino_model,
            target_fps=target_fps,
            device=device
        )

        print(f"Using Ray parallel processing with {self.ray_processor.num_workers} workers")
        print("Using PyAV for video processing (AV1 codec supported)")


    def build_dataset(
        self,
        trajectories: List[TrajectoryData],
        split_name: str = 'train',
        indices: Optional[List[int]] = None
    ) -> List[ProcessedTrajectory]:
        """
        Build processed dataset from trajectories using Ray parallel processing.

        Args:
            trajectories: List of raw trajectory data
            split_name: Name of the split (train/val/test)
            indices: Optional indices to process (if None, process all)

        Returns:
            List of processed trajectories
        """
        if indices is not None:
            trajectories = [trajectories[i] for i in indices]

        print(f"\nBuilding {split_name} dataset with {len(trajectories)} trajectories...")
        print(f"Processing with Ray ({self.ray_processor.num_workers} workers)...")

        # Process trajectories in parallel with Ray
        processed_dicts = self.ray_processor.process_dataset(trajectories)

        # Convert dicts back to ProcessedTrajectory objects
        processed_trajectories = []
        for p_dict in processed_dicts:
            if p_dict is not None:
                # Convert lists back to numpy arrays where needed
                if p_dict.get('proprioception') is not None:
                    p_dict['proprioception'] = np.array(p_dict['proprioception'])
                if p_dict.get('proprio_timestamps') is not None:
                    p_dict['proprio_timestamps'] = np.array(p_dict['proprio_timestamps'])
                if p_dict.get('dino_features') is not None:
                    p_dict['dino_features'] = np.array(p_dict['dino_features'])
                if p_dict.get('frame_labels') is not None:
                    p_dict['frame_labels'] = np.array(p_dict['frame_labels'])

                # Create ProcessedTrajectory object
                processed = ProcessedTrajectory(**p_dict)
                processed_trajectories.append(processed)

        # Save processed dataset
        self._save_split(processed_trajectories, split_name)

        print(f"Processed {len(processed_trajectories)} trajectories for {split_name}")

        return processed_trajectories


    def _save_split(self, trajectories: List[ProcessedTrajectory], split_name: str):
        """Save processed trajectories for a split."""
        # Save as JSON for easy loading
        json_path = self.output_dir / f"{split_name}_processed.json"

        data = {
            'split': split_name,
            'num_trajectories': len(trajectories),
            'trajectories': [t.to_dict() for t in trajectories]
        }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save as pickle for faster loading with numpy arrays intact
        pkl_path = self.output_dir / f"{split_name}_processed.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(trajectories, f)

        # Save as HDF5 for efficient numerical data access
        h5_path = self.output_dir / f"{split_name}_processed.h5"
        self._save_h5(trajectories, h5_path)

        print(f"Saved {split_name} split to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Pickle: {pkl_path}")
        print(f"  - HDF5: {h5_path}")

    def _save_h5(self, trajectories: List[ProcessedTrajectory], h5_path: Path):
        """Save trajectories in HDF5 format for efficient access."""
        with h5py.File(h5_path, 'w') as f:
            for i, traj in enumerate(trajectories):
                grp = f.create_group(f"trajectory_{i}")

                # Save metadata
                grp.attrs['trajectory_id'] = traj.trajectory_id
                grp.attrs['task_id'] = traj.task_id
                grp.attrs['episode_id'] = traj.episode_id
                grp.attrs['task_name'] = traj.task_name or ''
                grp.attrs['total_frames'] = traj.total_frames
                grp.attrs['fps'] = traj.fps

                # Save arrays
                if traj.frame_labels is not None:
                    grp.create_dataset('frame_labels', data=traj.frame_labels)

                if traj.proprioception is not None:
                    grp.create_dataset('proprioception', data=traj.proprioception)

                if traj.proprio_timestamps is not None:
                    grp.create_dataset('proprio_timestamps', data=traj.proprio_timestamps)

                if traj.dino_features is not None:
                    grp.create_dataset('dino_features', data=traj.dino_features)

                # Save segments
                if traj.segments:
                    seg_grp = grp.create_group('segments')
                    for j, seg in enumerate(traj.segments):
                        s = seg_grp.create_group(f"segment_{j}")
                        s.attrs.update(seg)

    def create_huggingface_dataset(self, splits: Dict[str, List[ProcessedTrajectory]]) -> None:
        """
        Create a HuggingFace-compatible dataset with all data included.

        Args:
            splits: Dictionary mapping split names to processed trajectories
        """
        try:
            from datasets import Dataset, DatasetDict, Features, Value, Array2D, Array3D, Sequence, Image
            from PIL import Image as PILImage
            import io

            print("\nCreating self-contained HuggingFace dataset...")

            dataset_dict = {}

            for split_name, trajectories in splits.items():
                # Convert to format suitable for HuggingFace datasets
                examples = []

                for traj in trajectories:
                    example = {
                        'trajectory_id': traj.trajectory_id,
                        'task_id': traj.task_id,
                        'episode_id': traj.episode_id,
                        'task_name': traj.task_name or '',
                        'frame_labels': traj.frame_labels.tolist() if traj.frame_labels is not None else [],
                        'segments': traj.segments,
                        'total_frames': traj.total_frames,
                        'fps': traj.fps
                    }

                    # Add proprioception data (actual data, not just metadata)
                    if traj.proprioception is not None:
                        example['proprioception'] = traj.proprioception.tolist()
                        example['proprio_shape'] = traj.proprioception.shape
                    else:
                        example['proprioception'] = []
                        example['proprio_shape'] = [0, 0]

                    if traj.proprio_timestamps is not None:
                        example['proprio_timestamps'] = traj.proprio_timestamps.tolist()
                    else:
                        example['proprio_timestamps'] = []

                    # Add DINO features (actual features)
                    if traj.dino_features is not None:
                        example['dino_features'] = traj.dino_features.tolist()
                        example['dino_shape'] = traj.dino_features.shape
                    else:
                        example['dino_features'] = []
                        example['dino_shape'] = [0, 0]

                    # Add images (actual image data)
                    if traj.images is not None and len(traj.images) > 0:
                        # Convert numpy arrays to PIL Images for HuggingFace
                        pil_images = []
                        for img_array in traj.images:
                            # Ensure uint8 format
                            if img_array.dtype != np.uint8:
                                img_array = (img_array * 255).astype(np.uint8)
                            pil_img = PILImage.fromarray(img_array)
                            pil_images.append(pil_img)
                        example['images'] = pil_images
                        example['num_images'] = len(pil_images)
                    else:
                        example['images'] = []
                        example['num_images'] = 0

                    # Add video paths for reference
                    if traj.video_paths:
                        example['video_paths'] = str(traj.video_paths)
                    else:
                        example['video_paths'] = ''

                    examples.append(example)

                # Create Dataset with proper feature types
                if examples:
                    # Define features explicitly to handle Image type
                    from datasets import Features, Value, Sequence, Array2D, Image as HFImage

                    # Check if we have images to determine feature structure
                    has_images = any(ex['num_images'] > 0 for ex in examples)

                    if has_images:
                        # Use Image feature for images
                        features = Features({
                            'trajectory_id': Value('string'),
                            'task_id': Value('int32'),
                            'episode_id': Value('int32'),
                            'task_name': Value('string'),
                            'frame_labels': Sequence(Value('int32')),
                            'segments': Sequence({
                                'start_frame': Value('int32'),
                                'end_frame': Value('int32'),
                                'action_text': Value('string'),
                                'skill': Value('string'),
                                'segment_id': Value('int32')
                            }),
                            'total_frames': Value('int32'),
                            'fps': Value('float32'),
                            'proprioception': Array2D(dtype='float32', shape=(None, None)),
                            'proprio_shape': Sequence(Value('int32')),
                            'proprio_timestamps': Sequence(Value('float32')),
                            'dino_features': Array2D(dtype='float32', shape=(None, None)),
                            'dino_shape': Sequence(Value('int32')),
                            'images': Sequence(HFImage()),
                            'num_images': Value('int32'),
                            'video_paths': Value('string')
                        })
                        dataset = Dataset.from_list(examples, features=features)
                    else:
                        # No images, use simpler structure
                        dataset = Dataset.from_list(examples)

                    dataset_dict[split_name] = dataset
                    print(f"  {split_name}: {len(examples)} trajectories")

            # Create DatasetDict
            dataset = DatasetDict(dataset_dict)

            # Save to disk
            hf_path = self.output_dir / 'huggingface_dataset'
            dataset.save_to_disk(str(hf_path))

            print(f"\nSaved self-contained HuggingFace dataset to {hf_path}")
            print(f"Dataset info: {dataset}")

            # Print size information
            total_size = 0
            for split_name, split_dataset in dataset.items():
                split_size = split_dataset._estimate_nbytes() / (1024 * 1024)  # Convert to MB
                print(f"  {split_name} size: ~{split_size:.2f} MB")
                total_size += split_size
            print(f"  Total size: ~{total_size:.2f} MB")

        except ImportError as e:
            print(f"Warning: Could not create HuggingFace dataset: {e}")
            print("Please install: pip install datasets pillow")