import subprocess
import os
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
from typing import List, Optional
from dataclasses import dataclass, field
import tyro
from pathlib import Path

api = HfApi()

@dataclass
class DownloadConfig:
    """Configuration for downloading AgiBot dataset."""
    
    download_dir: str
    """Local directory to download files to before moving to destination"""

    download_proprio: bool = True
    """Download proprio stats"""
    
    video_names: list[str] = field(default_factory=lambda: ["head_color"])
    """Video names to extract from tar files"""
    
    repo: str = "agibot-world/AgiBotWorld-Alpha"
    """HuggingFace repository name (can also be 'agibot-world/AgiBotWorld-Beta')"""
    
    max_num_trajectories: int = 2000
    """Maximum number of trajectories to extract per task"""
    
    task_ids: Optional[list[int]] = None
    """Task IDs to download. If None, downloads all available tasks"""
    
    max_workers: int = 6
    """Number of parallel workers for downloading tasks"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        os.makedirs(self.download_dir, exist_ok=True)

def get_all_task_ids(repo: str) -> List[int]:
    files = api.list_repo_files(repo, repo_type="dataset")
    prefix = "observations/"
    task_ids = set()
    for f in files:
        if f.startswith(prefix):
            task_ids.add(int(f.split("/")[-2]))
    return sorted(list(task_ids))

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def get_observation_tars(task_id: int, repo: str) -> List[str]:
    """Return the first .tar file in observations/{task_id}/."""
    files = api.list_repo_files(repo, repo_type="dataset")
    prefix = f"observations/{task_id}/"
    tars = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".tar"):
            tars.append(f)
    return tars


def download_and_extract(task_id: int, config: DownloadConfig):
    """Download task files, extract the observation tar, and cleanup."""
    obs_paths_full = get_observation_tars(task_id, config.repo)
    if not obs_paths_full:
        print(f"[Task {task_id}] No observation file found.")
        return
    
    num_trajectories = 0
    
    for obs_path in obs_paths_full:
        task_dir = os.path.join(config.download_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        print(f"[Task {task_id}] Downloading {obs_path}...")

        tar_filename = os.path.join(task_dir, obs_path)
        os.makedirs(os.path.dirname(tar_filename), exist_ok=True)

        # Download task_info.json and the observation tar
        if not os.path.exists(tar_filename):
            run_cmd([
                "hf", "download",
                "--repo-type", "dataset",
                "--local-dir", task_dir,
                config.repo,
                f"task_info/task_{task_id}.json",
                obs_path
            ])

        extract_dir = os.path.join(task_dir, "observations")
        os.makedirs(extract_dir, exist_ok=True)

        print(f"[Task {task_id}] Extracting {tar_filename}...")

        def should_extract(member: tarfile.TarInfo) -> bool:
            return any([x in member.name for x in config.video_names])

        with tarfile.open(tar_filename, "r") as tar:
            members = [m for m in tar.getmembers() if should_extract(m)][:config.max_num_trajectories - num_trajectories]
            num_trajectories += len(members)
            tar.extractall(path=extract_dir, members=members)

        print(f"[Task {task_id}] Extraction complete.")

        cleanup(task_dir, tar_filename)

        if num_trajectories >= config.max_num_trajectories:
            print(f"[Task {task_id}] Reached max_num_trajectories ({config.max_num_trajectories}). Stopping extraction for this task.")
            break

    cleanup_full(task_dir, task_id)

def download_and_extract_safe(task_id: int, config: DownloadConfig):
    try:
        download_and_extract(task_id, config)
    except Exception as e:
        print("-" * 80)
        print(f"[Task {task_id}] Error: {e}")
        print("-" * 80)
        return False
    return True


def cleanup(task_dir: str, tar_filename: str):
    """Delete the downloaded tar and all observations/*/depth folders."""
    if os.path.exists(tar_filename):
        os.remove(tar_filename)
    
    huggingface_cache_dir = os.path.join(task_dir, ".cache")
    if os.path.exists(huggingface_cache_dir):
        shutil.rmtree(huggingface_cache_dir)
    else:
        print(f"HuggingFace cache directory not found: {huggingface_cache_dir}, deleting the cache prevents filling disk space.")

def cleanup_full(task_dir: str, task_id: int):
    tar_dir = os.path.join(task_dir, "observations", str(task_id))
    if os.path.exists(tar_dir):
        shutil.rmtree(tar_dir)

def download_proprio_stats(config: DownloadConfig):
    """Download proprio stats for all tasks."""
    run_cmd([
        "hf", "download",
        config.repo,
        "--repo-type", "dataset",
        "--local-dir", config.download_dir,
        "--include", "proprio_stats/*"
    ])
    proprio_stats_dir = Path(os.path.join(config.download_dir, "proprio_stats"))
    tar_files = list(proprio_stats_dir.glob("*.tar"))

    print("Extracting proprio...")
    for tar_file in tar_files:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=proprio_stats_dir)
        os.remove(tar_file)

if __name__ == "__main__":
    config = tyro.cli(DownloadConfig)

    if config.download_proprio:
        print("Downloading proprio...")
        download_proprio_stats(config)
    
    if config.task_ids is None:
        all_task_ids = get_all_task_ids(config.repo)
        task_ids = all_task_ids
    else:
        task_ids = config.task_ids
    
    print("Downloading tasks: ", task_ids)
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        executor.map(lambda task_id: download_and_extract_safe(task_id, config), task_ids)

    print("All tasks processed, extracted, and cleaned up!")