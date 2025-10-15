"""
Direct HuggingFace loader for AgiBotWorld dataset.

This loader downloads JSON annotations, proprioceptive data (H5), and videos
from HuggingFace for complete Fast-DROID + DINOv3 embeddings.
"""

import json
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .base_dataset import (
    BaseSegmentationDataset,
    TrajectoryData,
    SegmentAnnotation
)


class AgiBotHFDirectDataset(BaseSegmentationDataset):
    """
    AgiBotWorld dataset loader that directly downloads JSON annotations from HuggingFace.

    This is a lightweight loader that only downloads annotation files,
    making it perfect for quick testing and benchmarking without downloading
    the full dataset (videos, proprioception, etc.).
    """

    def __init__(
        self,
        task_ids: Optional[List[int]] = None,
        max_trajectories: Optional[int] = None,
        cache_dir: str = "./agibot_cache",
        split: str = 'train',
        download_videos: bool = True,
        download_proprio: bool = True
    ):
        """
        Initialize AgiBotWorld dataset with direct HuggingFace download.

        Args:
            task_ids: Optional list of task IDs to load (default: [327])
            max_trajectories: Maximum number of trajectories to load
            cache_dir: Directory to cache downloaded files
            split: Dataset split (currently all data is treated as 'train')
            download_videos: Whether to download video files (can be large)
            download_proprio: Whether to download proprioceptive H5 files
        """
        self.task_ids = task_ids or [327]  # Default to task 327
        self.max_trajectories = max_trajectories
        self.cache_dir = Path(cache_dir)
        self.repo_id = "agibot-world/AgiBotWorld-Alpha"
        self.download_videos = download_videos
        self.download_proprio = download_proprio

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for extracted TAR files
        self.extract_dir = self.cache_dir / "extracted"
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        # Cache mapping episode IDs to their data files
        self.episode_data_cache: Dict[int, Dict[str, Any]] = {}

        super().__init__(self.cache_dir, split)

    def _load_trajectories(self) -> List[TrajectoryData]:
        """Load trajectories by downloading JSON files from HuggingFace."""
        trajectories = []

        print(f"Downloading AgiBotWorld annotations from HuggingFace...")
        print(f"Task IDs: {self.task_ids}")

        for task_id in tqdm(self.task_ids, desc="Loading tasks"):
            task_trajectories = self._load_task_from_hf(task_id)
            trajectories.extend(task_trajectories)

            # Check if we've reached the maximum
            if self.max_trajectories and len(trajectories) >= self.max_trajectories:
                trajectories = trajectories[:self.max_trajectories]
                break

        print(f"Loaded {len(trajectories)} trajectories")
        return trajectories

    def _load_task_from_hf(self, task_id: int) -> List[TrajectoryData]:
        """Download and parse a task JSON file from HuggingFace."""
        try:
            # Download the JSON file
            json_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f'task_info/task_{task_id}.json',
                repo_type='dataset',
                cache_dir=str(self.cache_dir)
            )

            # Load and parse JSON
            with open(json_path, 'r') as f:
                episodes = json.load(f)

            # Limit episodes if max_trajectories is set
            if self.max_trajectories:
                episodes = episodes[:self.max_trajectories]

            # Get episode IDs to determine which TAR files to download
            episode_ids = [ep.get('episode_id') for ep in episodes]

            # Download and extract TAR files for this task
            if self.download_videos or self.download_proprio:
                self._download_and_extract_tars(task_id, episode_ids)

            trajectories = []
            for episode in episodes:
                trajectory = self._parse_episode(episode, task_id)
                if trajectory:
                    trajectories.append(trajectory)

            return trajectories

        except Exception as e:
            print(f"Warning: Failed to load task {task_id}: {e}")
            return []

    def _download_and_extract_tars(self, task_id: int, episode_ids: List[int]):
        """Download and extract TAR files containing videos and H5 data for episodes."""
        from huggingface_hub import list_repo_files

        print(f"\nDownloading data for task {task_id} ({len(episode_ids)} episodes)...")

        # Download proprioceptive stats TAR (contains all H5 files)
        if self.download_proprio:
            self._download_proprio_stats(episode_ids)

        # List all TAR files in the observations directory
        if self.download_videos:
            try:
                # Check if videos are already extracted for our episodes
                extract_task_dir = self.extract_dir / str(task_id)
                videos_already_extracted = 0
                if extract_task_dir.exists():
                    for episode_id in episode_ids:
                        episode_dir = extract_task_dir / str(episode_id)
                        if episode_dir.exists() and (episode_dir / "videos").exists():
                            videos_already_extracted += 1
                            # Index already extracted videos
                            self._index_extracted_files(extract_task_dir, task_id)

                if videos_already_extracted == len(episode_ids):
                    print(f"All {videos_already_extracted} video directories already extracted")
                    return

                all_files = list_repo_files(
                    repo_id=self.repo_id,
                    repo_type='dataset'
                )
                tar_files = [
                    f for f in all_files
                    if f.startswith(f'observations/{task_id}/') and f.endswith('.tar')
                ]

                if not tar_files:
                    print(f"No observation TAR files found for task {task_id}")
                    return

                print(f"Found {len(tar_files)} observation TAR files")

                # Download and extract each TAR file
                for tar_filename in tqdm(tar_files, desc=f"Extracting observation TARs"):
                    try:
                        # Download TAR file
                        tar_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=tar_filename,
                            repo_type='dataset',
                            cache_dir=str(self.cache_dir)
                        )

                        # Extract TAR file
                        extract_task_dir.mkdir(parents=True, exist_ok=True)

                        with tarfile.open(tar_path, 'r') as tar:
                            tar.extractall(path=extract_task_dir)

                        # Index extracted files by episode ID
                        self._index_extracted_files(extract_task_dir, task_id)

                    except Exception as e:
                        print(f"Warning: Failed to extract {tar_filename}: {e}")

            except Exception as e:
                print(f"Warning: Failed to list TAR files for task {task_id}: {e}")

    def _download_proprio_stats(self, episode_ids: List[int]):
        """Download and extract proprioceptive stats TAR file (only for requested episodes)."""
        print(f"\nDownloading proprioceptive H5 files for {len(episode_ids)} episodes...")

        try:
            # Download the main proprio_stats TAR file
            tar_path = hf_hub_download(
                repo_id=self.repo_id,
                filename='proprio_stats/648533-923022.tar',
                repo_type='dataset',
                cache_dir=str(self.cache_dir)
            )

            # Extract only the H5 files we need
            print("Extracting H5 files...")
            proprio_extract_dir = self.extract_dir / "proprio_stats"
            proprio_extract_dir.mkdir(parents=True, exist_ok=True)

            # First, check for already extracted files
            already_extracted = 0
            for episode_id in episode_ids:
                # Check both flat structure and nested structure
                flat_path = proprio_extract_dir / f"{episode_id}.h5"
                nested_paths = list(proprio_extract_dir.rglob(f"*/{episode_id}/proprio_stats.h5"))

                target_path = None
                if flat_path.exists():
                    target_path = flat_path
                    already_extracted += 1
                elif nested_paths:
                    target_path = nested_paths[0]
                    already_extracted += 1

                if target_path:
                    if episode_id not in self.episode_data_cache:
                        self.episode_data_cache[episode_id] = {
                            'videos': {},
                            'h5': None
                        }
                    self.episode_data_cache[episode_id]['h5'] = target_path

            if already_extracted == len(episode_ids):
                print(f"All {already_extracted} H5 files already extracted")
                return

            # Open TAR and extract only requested episodes
            # H5 files are structured as: {task_id}/{episode_id}/proprio_stats.h5
            extracted_count = 0
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    # Check if this is an H5 file
                    if member.name.endswith('proprio_stats.h5'):
                        try:
                            # Parse path: {task_id}/{episode_id}/proprio_stats.h5
                            parts = Path(member.name).parts
                            if len(parts) >= 3:
                                episode_id = int(parts[1])

                                if episode_id in episode_ids:
                                    # Check if already extracted or indexed
                                    if episode_id in self.episode_data_cache and self.episode_data_cache[episode_id]['h5']:
                                        continue

                                    # Extract to flat structure
                                    target_path = proprio_extract_dir / f"{episode_id}.h5"
                                    if not target_path.exists():
                                        # Extract to temp location
                                        tar.extract(member, path=proprio_extract_dir)
                                        extracted_path = proprio_extract_dir / member.name
                                        extracted_path.rename(target_path)
                                        # Clean up empty directories
                                        try:
                                            extracted_path.parent.rmdir()
                                            extracted_path.parent.parent.rmdir()
                                        except:
                                            pass
                                        extracted_count += 1

                                    # Index this file
                                    if episode_id not in self.episode_data_cache:
                                        self.episode_data_cache[episode_id] = {
                                            'videos': {},
                                            'h5': None
                                        }
                                    self.episode_data_cache[episode_id]['h5'] = target_path
                        except (ValueError, IndexError):
                            pass

            print(f"Extracted {extracted_count} new H5 files ({already_extracted} already existed)")

        except Exception as e:
            print(f"Warning: Failed to download proprioceptive stats: {e}")

    def _index_h5_files(self, proprio_dir: Path):
        """Index H5 files by episode ID."""
        for h5_file in proprio_dir.rglob('*.h5'):
            try:
                # H5 files are named like: 648642.h5
                episode_id = int(h5_file.stem)

                if episode_id not in self.episode_data_cache:
                    self.episode_data_cache[episode_id] = {
                        'videos': {},
                        'h5': None
                    }

                self.episode_data_cache[episode_id]['h5'] = h5_file

            except (ValueError, IndexError):
                pass

    def _index_extracted_files(self, extract_dir: Path, task_id: int):
        """Index extracted files by episode ID."""
        # Find all files in extracted directory
        for file_path in extract_dir.rglob('*'):
            if file_path.is_file():
                # Extract episode ID from filename
                # Files are typically named like: 648642_head_color.mp4, 648642.h5, etc.
                filename = file_path.stem.split('_')[0]  # Get episode ID part
                try:
                    episode_id = int(filename)

                    if episode_id not in self.episode_data_cache:
                        self.episode_data_cache[episode_id] = {
                            'videos': {},
                            'h5': None
                        }

                    # Categorize file
                    if file_path.suffix == '.h5':
                        self.episode_data_cache[episode_id]['h5'] = file_path
                    elif file_path.suffix == '.mp4':
                        # Determine video type from filename
                        video_type = '_'.join(file_path.stem.split('_')[1:])  # e.g., "head_color"
                        self.episode_data_cache[episode_id]['videos'][video_type] = file_path

                except (ValueError, IndexError):
                    # Not a valid episode file
                    pass

    def _get_h5_file(self, episode_id: int) -> Optional[Path]:
        """Get proprioceptive H5 file for an episode from cache."""
        if not self.download_proprio:
            return None

        episode_data = self.episode_data_cache.get(episode_id)
        if episode_data and episode_data.get('h5'):
            return episode_data['h5']

        return None

    def _get_videos(self, episode_id: int) -> Dict[str, Path]:
        """Get video files for an episode from cache."""
        if not self.download_videos:
            return {}

        episode_data = self.episode_data_cache.get(episode_id)
        if episode_data and episode_data.get('videos'):
            return episode_data['videos']

        return {}

    def _parse_episode(self, episode: Dict[str, Any], task_id: int) -> Optional[TrajectoryData]:
        """Parse an episode into TrajectoryData."""
        try:
            # Extract basic info
            episode_id = episode.get('episode_id')
            task_name = f"Task {task_id}"  # Task names not in JSON

            # Parse segmentation annotations
            segments = []
            label_info = episode.get('label_info') or episode.get('lable_info')
            if label_info and 'action_config' in label_info:
                action_configs = label_info['action_config']

                for seg_id, action in enumerate(action_configs):
                    segment = SegmentAnnotation(
                        start_frame=action['start_frame'],
                        end_frame=action['end_frame'],
                        action_text=action['action_text'],
                        skill=action.get('skill'),
                        segment_id=seg_id
                    )
                    segments.append(segment)

            # Get H5 proprioceptive data from extracted TAR
            h5_path = self._get_h5_file(episode_id)

            # Get videos from extracted TAR
            video_paths = self._get_videos(episode_id)

            # Create trajectory data
            trajectory = TrajectoryData(
                trajectory_id=f"{task_id}_{episode_id}",
                task_id=str(task_id),
                task_name=task_name,
                proprioception_path=h5_path,
                video_paths=video_paths if video_paths else None,
                segments=segments,
                metadata={
                    'episode_id': episode_id,
                    'source': 'huggingface_direct',
                    'has_proprio': h5_path is not None,
                    'has_videos': len(video_paths) > 0
                }
            )

            return trajectory

        except Exception as e:
            print(f"Warning: Failed to parse episode: {e}")
            return None


if __name__ == "__main__":
    # Test the loader
    dataset = AgiBotHFDirectDataset(
        task_ids=[327],
        max_trajectories=5
    )

    dataset.print_statistics()

    # Show a few examples
    print("\nSample trajectories:")
    for i in range(min(3, len(dataset))):
        traj = dataset[i]
        print(f"\nTrajectory {i}: {traj.trajectory_id}")
        print(f"  Task: {traj.task_name}")
        print(f"  Segments: {traj.num_segments}")
        print(f"  Total frames: {traj.total_frames}")
        print(f"  Segments:")
        for seg in traj.segments[:3]:  # Show first 3 segments
            print(f"    [{seg.start_frame}-{seg.end_frame}] {seg.action_text} ({seg.skill})")
