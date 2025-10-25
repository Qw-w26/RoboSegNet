"""
Dataset builder for creating comprehensive datasets.

This module creates structured datasets with:
- Proprioception data (with relative paths)
- DINO features (subsampled, correlated with images)
- Image paths (relative paths to PNG files)
- Video paths for reference
- Task segmentation labels
- Metadata

The output format is saved as Parquet files for efficient storage and loading.
Images are stored as PNG files with relative paths in the dataset.
"""

import numpy as np
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
import os
from PIL import Image as PILImage

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
    image_paths: Optional[List[str]] = None  # List of relative image paths
    video_paths: Optional[Dict[str, str]] = None  # Camera -> video path

    # Metadata
    task_name: Optional[str] = None
    total_frames: int = 0
    fps: float = 30.0
    metadata: Optional[Dict[str, Any]] = None

    def to_parquet_row(self, frames_base_dir: Path) -> Dict[str, Any]:
        """
        Convert to dictionary for Parquet row with relative paths.
        
        Args:
            frames_base_dir: Base directory for frames (to compute relative paths)
            
        Returns:
            Dictionary suitable for Parquet serialization
        """
        row = {
            'trajectory_id': self.trajectory_id,
            'task_id': self.task_id,
            'episode_id': self.episode_id,
            'task_name': self.task_name or '',
            'total_frames': self.total_frames,
            'fps': self.fps,
        }
        
        # Frame labels
        if self.frame_labels is not None:
            row['frame_labels'] = self.frame_labels.tolist()
        else:
            row['frame_labels'] = []
        
        # Segments
        if self.segments:
            row['segments'] = self.segments
        else:
            row['segments'] = []
        
        # Proprioception - store as list
        if self.proprioception is not None:
            row['proprioception'] = self.proprioception.tolist()
            row['proprio_shape'] = list(self.proprioception.shape)
        else:
            row['proprioception'] = []
            row['proprio_shape'] = [0, 0]
            
        if self.proprio_timestamps is not None:
            row['proprio_timestamps'] = self.proprio_timestamps.tolist()
        else:
            row['proprio_timestamps'] = []
        
        # DINO features - store as list (subsampled, correlated with images)
        if self.dino_features is not None:
            row['dino_features'] = self.dino_features.tolist()
            row['dino_shape'] = list(self.dino_features.shape)
        else:
            row['dino_features'] = []
            row['dino_shape'] = [0, 0]
        
        # Image paths - relative paths stored as list (already saved by Ray workers)
        if self.image_paths is not None and len(self.image_paths) > 0:
            row['image_paths'] = self.image_paths
            row['num_images'] = len(self.image_paths)
        else:
            row['image_paths'] = []
            row['num_images'] = 0
        
        # Video paths
        if self.video_paths:
            row['video_paths'] = str(self.video_paths)
        else:
            row['video_paths'] = ''
        
        # Metadata
        if self.metadata:
            row['metadata'] = str(self.metadata)
        else:
            row['metadata'] = ''
        
        return row


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
            frames_dir=str(self.frames_dir) if extract_frames else None,
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
        """
        Save processed trajectories for a split in Parquet format.
        
        Images are already saved by Ray workers, we just need to save the metadata.
        """
        print(f"\nSaving {split_name} split to Parquet format...")
        
        # Prepare data rows (images already saved by Ray workers)
        rows = []
        
        for traj in tqdm(trajectories, desc=f"Saving {split_name}"):
            # Convert trajectory to row (image_paths already set)
            row = traj.to_parquet_row(self.output_dir / 'frames')
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Save as Parquet
        parquet_path = self.output_dir / f"{split_name}_processed.parquet"
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        
        print(f"\nSaved {split_name} split to: {parquet_path}")
        print(f"  - Trajectories: {len(trajectories)}")
        print(f"  - Total images saved: {sum(len(row.get('image_paths', [])) for row in rows)}")
        print(f"  - Parquet file size: {parquet_path.stat().st_size / (1024**2):.2f} MB")
        
        # Print column info
        print(f"\nDataset columns:")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")
    
    def load_split(self, split_name: str) -> pd.DataFrame:
        """
        Load a processed split from Parquet file.
        
        Args:
            split_name: Name of the split (train/val/test)
            
        Returns:
            DataFrame with the split data
        """
        parquet_path = self.output_dir / f"{split_name}_processed.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Split file not found: {parquet_path}")
        
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        print(f"Loaded {split_name} split: {len(df)} trajectories")
        return df