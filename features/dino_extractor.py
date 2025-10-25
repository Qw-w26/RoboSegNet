"""DINO vision feature extraction with Ray parallelization support.

This module provides feature extraction using Facebook's DINOv3-ViT-L16 model,
with both single-process and Ray-parallelized implementations for efficient
batch processing across multiple GPUs.
"""

from typing import List, Dict, Optional, Set
from pathlib import Path
import numpy as np
import av
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import ray
from tqdm import tqdm


# Model configuration
DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BATCH_SIZE = 256
DEFAULT_FEATURE_DIM = 1024


class DinoFeatureExtractor:
    """
    Extract visual features from video frames using DINOv3-ViT-L16.

    This extractor loads a pre-trained DINO model and extracts features from
    specified frames in a video file. Features are extracted in batches to
    handle memory efficiently.

    Attributes:
        device: Device to run model on ('cuda' or 'cpu')
        batch_size: Number of frames to process in each batch
        model: DINOv3 vision transformer model
        processor: Image processor for model inputs
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Initialize DINO feature extractor.

        Args:
            device: Device to run model on ('cuda' or 'cpu')
            cache_dir: Directory to cache downloaded model weights
            batch_size: Number of frames to process per batch

        Example:
            >>> extractor = DinoFeatureExtractor(device='cuda')
            >>> features = extractor.extract_batch('video.mp4', [0, 10, 20])
        """
        self.device = device
        self.batch_size = batch_size

        # Load model and processor
        self.model = AutoModel.from_pretrained(
            DINO_MODEL_NAME,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device).eval()

        self.processor = AutoProcessor.from_pretrained(
            DINO_MODEL_NAME,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

    def extract_batch(
        self,
        video_path: str,
        frame_indices: List[int],
        show_progress: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Extract DINO features for specified frames from a video.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract features for
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping frame index to feature vector (1024-dim)

        Raises:
            ValueError: If video file doesn't exist or no frames found
            RuntimeError: If feature extraction fails

        Example:
            >>> features = extractor.extract_batch('video.mp4', [0, 10, 20])
            >>> print(features[0].shape)  # (1024,)
        """
        # Validate inputs
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        frame_indices = sorted(set(frame_indices))
        if not frame_indices:
            return {}

        # Extract frames from video
        imgs, found_idx = self._load_frames_from_video(
            str(video_path),
            frame_indices,
            show_progress=show_progress
        )

        if not imgs:
            raise ValueError(
                f"No images found for frames {frame_indices} in video {video_path}"
            )

        # Extract features in batches
        features = self._extract_features_batched(imgs, show_progress=show_progress)

        # Map features to frame indices
        return {idx: features[i] for i, idx in enumerate(found_idx)}

    def _load_frames_from_video(
        self,
        video_path: str,
        frame_indices: List[int],
        show_progress: bool = False
    ) -> tuple[List[Image.Image], List[int]]:
        """
        Load specific frames from video file.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to load
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (images, found_indices)
        """
        imgs = []
        found_idx = []
        current_idx = 0

        # Set up progress bar if requested
        pbar = tqdm(total=max(frame_indices) + 1, disable=not show_progress,
                   desc="Loading frames")

        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if current_idx in frame_indices:
                            # Convert frame to PIL Image
                            img_array = frame.to_ndarray(format="rgb24")
                            imgs.append(Image.fromarray(img_array))
                            found_idx.append(current_idx)

                            # Stop early if we found all frames
                            if len(found_idx) == len(frame_indices):
                                break

                        current_idx += 1
                        pbar.update(1)

                    if len(found_idx) == len(frame_indices):
                        break
        finally:
            pbar.close()

        return imgs, found_idx

    def _extract_features_batched(
        self,
        imgs: List[Image.Image],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Extract DINO features from images in batches.

        Args:
            imgs: List of PIL Images
            show_progress: Whether to show progress bar

        Returns:
            Feature array of shape (N, 1024)
        """
        feats_list = []

        # Process in batches to avoid OOM
        num_batches = (len(imgs) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(range(num_batches), disable=not show_progress,
                   desc="Extracting DINO features")

        for i in pbar:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(imgs))
            batch_imgs = imgs[start_idx:end_idx]

            # Process batch
            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token representation (first token)
                feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            feats_list.append(feats)

            # Clear GPU memory
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batches
        return np.concatenate(feats_list, axis=0)

    @staticmethod
    def get_feature_dim() -> int:
        """Get dimensionality of extracted features."""
        return DEFAULT_FEATURE_DIM


@ray.remote(num_gpus=0.25)
class RayDinoFeatureExtractor(DinoFeatureExtractor):
    """
    Ray-remote version of DinoFeatureExtractor for parallel processing.

    This class can be instantiated as a Ray actor to enable parallel
    feature extraction across multiple GPUs.

    Example:
        >>> import ray
        >>> ray.init(num_gpus=4)
        >>> # Create 4 actors, each using 0.25 GPU
        >>> actors = [RayDinoFeatureExtractor.remote() for _ in range(4)]
        >>> # Extract features in parallel
        >>> futures = [actor.extract_batch.remote(video, frames)
        ...            for actor, video, frames in zip(actors, videos, frames_list)]
        >>> results = ray.get(futures)
    """
    pass


def extract_features_for_windows(
    video_path: str,
    windows: List[tuple],
    extractor: DinoFeatureExtractor,
    frames_per_window: int = 3
) -> Dict[int, np.ndarray]:
    """
    Extract DINO features for a list of windows.

    For each window, extracts features at begin, middle, and end frames.

    Args:
        video_path: Path to video file
        windows: List of (start_frame, end_frame) tuples
        extractor: DinoFeatureExtractor instance
        frames_per_window: Number of frames to sample per window (default: 3)

    Returns:
        Dictionary mapping frame index to features

    Example:
        >>> windows = [(0, 30), (30, 60), (60, 90)]
        >>> features = extract_features_for_windows('video.mp4', windows, extractor)
    """
    # Collect all frames of interest
    interest_frames: Set[int] = set()

    for start_frame, end_frame in windows:
        if frames_per_window == 3:
            # Begin, middle, end
            middle_frame = (start_frame + end_frame) // 2
            interest_frames.update([start_frame, middle_frame, end_frame])
        elif frames_per_window == 1:
            # Only middle
            middle_frame = (start_frame + end_frame) // 2
            interest_frames.add(middle_frame)
        else:
            # Uniform sampling
            step = (end_frame - start_frame) // (frames_per_window - 1)
            for i in range(frames_per_window):
                interest_frames.add(start_frame + i * step)

    # Extract features for all frames
    return extractor.extract_batch(video_path, list(interest_frames))


# Utility functions for Ray parallel extraction
def create_ray_extractors(
    num_actors: int = 4,
    gpus_per_actor: float = 0.25,
    **kwargs
) -> List[ray.ObjectRef]:
    """
    Create multiple Ray actor instances for parallel extraction.

    Args:
        num_actors: Number of Ray actors to create
        gpus_per_actor: GPU fraction per actor
        **kwargs: Additional arguments for DinoFeatureExtractor

    Returns:
        List of Ray actor references

    Example:
        >>> actors = create_ray_extractors(num_actors=4)
        >>> # Now use actors for parallel extraction
    """
    # Modify the Ray remote decorator dynamically
    RemoteClass = ray.remote(num_gpus=gpus_per_actor)(DinoFeatureExtractor)

    return [RemoteClass.remote(**kwargs) for _ in range(num_actors)]


def batch_extract_parallel(
    video_paths: List[str],
    frame_indices_list: List[List[int]],
    num_workers: int = 4,
    gpus_per_worker: float = 0.25,
    **kwargs
) -> List[Dict[int, np.ndarray]]:
    """
    Extract features from multiple videos in parallel using Ray.

    Args:
        video_paths: List of video file paths
        frame_indices_list: List of frame indices for each video
        num_workers: Number of Ray workers
        gpus_per_worker: GPU fraction per worker
        **kwargs: Additional arguments for DinoFeatureExtractor

    Returns:
        List of feature dictionaries, one per video

    Example:
        >>> videos = ['video1.mp4', 'video2.mp4']
        >>> frames = [[0, 10, 20], [5, 15, 25]]
        >>> results = batch_extract_parallel(videos, frames, num_workers=2)
    """
    from utils.ray_utils import ensure_ray_initialized

    # Ensure Ray is initialized
    ensure_ray_initialized(num_gpus=num_workers * gpus_per_worker)

    # Create workers
    actors = create_ray_extractors(num_workers, gpus_per_worker, **kwargs)

    # Distribute work across actors
    futures = []
    for i, (video_path, frame_indices) in enumerate(zip(video_paths, frame_indices_list)):
        actor_idx = i % num_workers
        future = actors[actor_idx].extract_batch.remote(video_path, frame_indices)
        futures.append(future)

    # Wait for all results
    results = ray.get(futures)

    # Cleanup actors
    for actor in actors:
        ray.kill(actor)

    return results
