"""
Training dataset generator for segment boundary prediction.

This module generates training data for learning to predict segment boundaries
using sliding windows over trajectories with DINO visual features and Fast-DROID
tokenized proprioception.
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pickle

from data import BaseSegmentationDataset, TrajectoryData
from embedding.fast_droid_embeddings import FastDROIDEmbedder
from benchmark.metrics import detect_boundaries


class SegmentBoundaryDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for segment boundary prediction.

    Generates windows from trajectories and labels them as boundary/non-boundary.
    Features include:
    - DINO features from begin, middle, end frames (frozen)
    - Fast-DROID tokenized proprio
    - Statistical proprio features
    """

    def __init__(
        self,
        dataset: BaseSegmentationDataset,
        window_size: int = 30,
        stride: int = 30,
        max_token_length: int = 600,
        boundary_tolerance: int = 2,
        cache_dir: Optional[Path] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        precompute_features: bool = True
    ):
        """
        Initialize boundary prediction dataset.

        Args:
            dataset: Segmentation dataset with trajectories
            window_size: Size of sliding window
            stride: Stride for sliding window
            max_token_length: Fixed maximum token length for Fast-DROID
            boundary_tolerance: Tolerance for labeling frames near boundaries as positive
            cache_dir: Directory to cache extracted features
            device: Device for DINO model
            precompute_features: If True, pre-compute and cache all DINO features
        """
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride
        self.max_token_length = max_token_length
        self.boundary_tolerance = boundary_tolerance
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.precompute_features = precompute_features

        # Initialize embedder
        self.embedder = FastDROIDEmbedder(
            window_size=window_size,
            stride=stride,
            use_dino=True,
            max_token_length=max_token_length,
            device=device
        )

        # Storage for samples and cached features
        self.samples = []
        self.cached_dino_features = {}  # Dict: sample_idx -> dino_features

        # Generate samples
        print("Generating training samples...")
        self._generate_samples()

        # Pre-compute DINO features if requested
        if self.precompute_features:
            print("Pre-computing DINO features for all samples...")
            self._precompute_dino_features()

    def _generate_samples(self):
        """Generate all training samples from the dataset."""
        for traj in tqdm(self.dataset.trajectories, desc="Processing trajectories"):
            traj_samples = self._process_trajectory(traj)
            self.samples.extend(traj_samples)

        print(f"Generated {len(self.samples)} samples")

        # Print class distribution
        n_positive = sum(1 for s in self.samples if s['label'] == 1)
        n_negative = len(self.samples) - n_positive
        print(f"Positive samples (boundaries): {n_positive}")
        print(f"Negative samples (non-boundaries): {n_negative}")
        print(f"Ratio: {n_positive / len(self.samples):.3f}")

    def _process_trajectory(self, traj: TrajectoryData) -> List[Dict[str, Any]]:
        """
        Process a single trajectory and generate samples.

        Args:
            traj: Trajectory data

        Returns:
            List of sample dictionaries
        """
        samples = []

        # Skip if no segments
        if traj.num_segments == 0:
            return samples

        # Load proprioception
        proprio_path = traj.proprioception_path
        if not proprio_path or not proprio_path.exists():
            return samples

        proprio = self.embedder.load_proprioception(proprio_path)
        if proprio is None or len(proprio) < self.window_size:
            return samples

        # Get ground truth segment labels
        segment_labels = traj.get_segment_labels()
        if len(segment_labels) == 0:
            return samples

        # Detect ground truth boundaries
        gt_boundaries = detect_boundaries(segment_labels)
        gt_boundaries_set = set(gt_boundaries)

        # Expand boundaries with tolerance
        boundary_frames = set()
        for b in gt_boundaries:
            for offset in range(-self.boundary_tolerance, self.boundary_tolerance + 1):
                frame_idx = b + offset
                if 0 <= frame_idx < len(segment_labels):
                    boundary_frames.add(frame_idx)

        # Get video path (prefer head camera)
        video_path = None
        if traj.video_paths:
            video_path = (
                traj.video_paths.get('head_color') or
                traj.video_paths.get('hand_left_color') or
                list(traj.video_paths.values())[0]
            )

        # Create windows
        windows = self.embedder.create_windows(proprio)

        for window, center_frame_idx in windows:
            # Check if center frame is valid
            if center_frame_idx >= len(segment_labels):
                continue

            # Determine label
            label = 1 if center_frame_idx in boundary_frames else 0

            # Compute frame indices for DINO extraction
            start_idx = center_frame_idx - self.window_size // 2
            end_idx = start_idx + self.window_size
            begin_frame = max(0, start_idx)
            middle_frame = center_frame_idx
            end_frame = min(len(proprio) - 1, end_idx - 1)

            sample = {
                'trajectory_id': traj.trajectory_id,
                'center_frame_idx': center_frame_idx,
                'window': window,  # (window_size, 14)
                'video_path': str(video_path) if video_path else None,
                'frame_indices': (begin_frame, middle_frame, end_frame),
                'label': label,
                'proprio_path': str(proprio_path)
            }

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - tokenized_actions: Tokenized actions from window
            - proprio_stats: Statistical features from window
            - dino_features: DINO features from begin, middle, end frames
            - label: Binary label (0 or 1)
        """
        sample = self.samples[idx]

        # Extract window features
        window = sample['window']

        # Tokenize actions
        if len(window) > 1:
            actions = np.diff(window, axis=0)
            action_str = " ".join([f"{x:.4f}" for x in actions.flatten()])
            tokens = self.embedder.tokenizer(action_str, truncation=False, return_tensors='np')
            token_ids = tokens['input_ids'].flatten()

            # Pad/truncate
            if len(token_ids) < self.max_token_length:
                pad_id = self.embedder.tokenizer.pad_token_id or 0
                token_embedding = np.pad(
                    token_ids,
                    (0, self.max_token_length - len(token_ids)),
                    constant_values=pad_id
                )
            else:
                token_embedding = token_ids[:self.max_token_length]
        else:
            token_embedding = np.zeros(self.max_token_length)

        # Statistical features
        proprio_stats = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            np.percentile(window, 25, axis=0),
            np.percentile(window, 75, axis=0)
        ])

        # Get DINO features (from cache if available, otherwise extract)
        if idx in self.cached_dino_features:
            dino_features = self.cached_dino_features[idx]
        else:
            dino_features = self._extract_dino_features(
                sample['video_path'],
                sample['frame_indices']
            )

        return {
            'tokenized_actions': torch.tensor(token_embedding, dtype=torch.long),
            'proprio_stats': torch.tensor(proprio_stats, dtype=torch.float32),
            'dino_features': torch.tensor(dino_features, dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'center_frame_idx': sample['center_frame_idx'],
            'trajectory_id': sample['trajectory_id']
        }

    def _extract_dino_features(
        self,
        video_path: Optional[str],
        frame_indices: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract DINO features from begin, middle, end frames.

        Args:
            video_path: Path to video file
            frame_indices: (begin, middle, end) frame indices

        Returns:
            Concatenated DINO features of shape (3 * 1024,)
        """
        if video_path is None or not Path(video_path).exists():
            # Return zeros if no video
            return np.zeros(3 * 1024)

        begin_idx, middle_idx, end_idx = frame_indices

        # Extract features for each frame
        features = []
        for frame_idx in [begin_idx, middle_idx, end_idx]:
            feat = self.embedder.extract_dino_features(Path(video_path), frame_idx)
            if feat is None:
                feat = np.zeros(1024)
            features.append(feat)

        return np.concatenate(features)

    def _precompute_dino_features(self):
        """Pre-compute DINO features for all samples to speed up training."""
        print(f"Pre-computing DINO features for {len(self.samples)} samples...")

        for idx, sample in enumerate(tqdm(self.samples, desc="Extracting DINO features")):
            dino_features = self._extract_dino_features(
                sample['video_path'],
                sample['frame_indices']
            )
            self.cached_dino_features[idx] = dino_features

        print(f"Pre-computed DINO features for {len(self.cached_dino_features)} samples")

    def save_to_disk(self, save_path: Path):
        """Save preprocessed dataset to disk including cached features."""
        print(f"Saving dataset to {save_path}...")

        data = {
            'samples': self.samples,
            'cached_dino_features': self.cached_dino_features,
            'config': {
                'window_size': self.window_size,
                'stride': self.stride,
                'max_token_length': self.max_token_length,
                'boundary_tolerance': self.boundary_tolerance
            }
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        cache_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"Dataset saved to {save_path} ({cache_size_mb:.2f} MB)")

    @classmethod
    def load_from_disk(cls, load_path: Path, dataset: BaseSegmentationDataset, device: str = 'cuda') -> 'SegmentBoundaryDataset':
        """Load preprocessed dataset from disk."""
        print(f"Loading cached dataset from {load_path}...")

        if not load_path.exists():
            raise FileNotFoundError(f"Cache file not found: {load_path}")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        # Create instance without regenerating samples
        instance = cls.__new__(cls)
        instance.dataset = dataset
        instance.samples = data['samples']
        instance.cached_dino_features = data.get('cached_dino_features', {})
        instance.window_size = data['config']['window_size']
        instance.stride = data['config']['stride']
        instance.max_token_length = data['config']['max_token_length']
        instance.boundary_tolerance = data['config']['boundary_tolerance']
        instance.device = device
        instance.precompute_features = True

        # Initialize embedder
        instance.embedder = FastDROIDEmbedder(
            window_size=instance.window_size,
            stride=instance.stride,
            use_dino=True,
            max_token_length=instance.max_token_length,
            device=device
        )

        cache_size_mb = load_path.stat().st_size / (1024 * 1024)
        print(f"Loaded {len(instance.samples)} samples with {len(instance.cached_dino_features)} cached DINO features ({cache_size_mb:.2f} MB)")

        return instance


def create_dataloaders(
    train_dataset: SegmentBoundaryDataset,
    val_dataset: Optional[SegmentBoundaryDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, val_loader) tuple
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    from data import AgiBotWorldDataset

    # Load dataset
    dataset = AgiBotWorldDataset(
        data_root=Path("./agibot_full_data/extracted"),
        task_ids=[327],
        max_trajectories=10
    )

    # Generate training data
    train_dataset = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=15,
        stride=1,
        max_token_length=600,
        boundary_tolerance=2
    )

    # Save to disk
    train_dataset.save_to_disk(Path("./training_data/boundary_dataset.pkl"))

    # Create dataloader
    train_loader, _ = create_dataloaders(train_dataset, batch_size=32)

    # Test loading a batch
    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
