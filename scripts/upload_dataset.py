import os
import tarfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from huggingface_hub import HfApi
import tyro

api = HfApi()

IGNORE = ["hf_cache"]

@dataclass
class UploadConfig:
    """Configuration for uploading dataset to HuggingFace."""
    
    dataset_dir: str
    """Directory containing the dataset to upload"""
    
    upload_folder: str
    """Directory where processed dataset (tars and files) will be stored before upload"""
    
    repo_id: str = "apgoldberg/RoboSegData"
    """HuggingFace repository ID"""
    
    repo_type: str = "dataset"
    """Repository type (dataset or model)"""

    cleanup: bool = False
    """Whether to cleanup the upload folder after upload"""

def main(config: UploadConfig):
    source_path = Path(config.dataset_dir)
    upload_path = Path(config.upload_folder)
    
    if not source_path.exists():
        raise ValueError(f"Dataset directory does not exist: {config.dataset_dir}")
    
    # Create the upload folder if it doesn't exist
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Process each item in the source directory
    for item in source_path.iterdir():
        if item.name in IGNORE:
            continue
        if item.is_dir():
            # Create a tar file for each directory
            tar_path = upload_path / f"{item.name}.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(item, arcname=item.name)
            print(f"Created tar: {tar_path}")
        elif item.is_file():
            shutil.copy2(item, upload_path / item.name)
            print(f"Copied file: {item.name}")
    
    print(f"Uploading processed dataset from {upload_path}...")
    api.upload_large_folder(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        folder_path=str(upload_path),
    )
    print("Upload complete!")

    if config.cleanup:
        shutil.rmtree(upload_path)
        print(f"Cleaned up upload folder: {upload_path}")

if __name__ == "__main__":
    config = tyro.cli(UploadConfig)
    main(config)