"""
Example script for training the boundary prediction model.

This script demonstrates how to:
1. Load the AgiBotWorld dataset
2. Generate training data with DINO features and tokenized proprio
3. Train an MLP model to predict segment boundaries
"""

import os

from pathlib import Path
import torch
import wandb
import numpy as np
import cv2
from tqdm import tqdm
import ray
from typing import List, Tuple, Dict, Any

from data import AgiBotHFDirectDataset
from training import SegmentBoundaryDataset, create_dataloaders, Trainer, create_smoother
from model import create_model, SoftFocalLoss, detect_boundaries as detect_boundaries_nms


@ray.remote(num_gpus=1)
class InferenceDinoExtractor:
    """Ray actor for extracting DINO features during inference."""

    def __init__(self, device='cuda'):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

        try:
            self.dino_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.dino_model.eval()
            self.dino_model = self.dino_model.to(self.device)
            self.dino_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load DINOv3 model: {e}")
            self.dino_model = None
            self.dino_processor = None

    def extract_features_batch(self, video_path: str, frame_indices_list: List[Tuple[int, int, int]]) -> List[np.ndarray]:
        """Extract DINO features for a batch of windows."""
        from pathlib import Path

        results = []
        for frame_indices in frame_indices_list:
            features = self._extract_dino_features(Path(video_path), frame_indices)
            results.append(features)
        return results

    def _extract_dino_features(self, video_path, frame_indices: Tuple[int, int, int]) -> np.ndarray:
        """Extract DINO features from begin, middle, end frames."""
        if not video_path.exists() or self.dino_model is None:
            return np.zeros(3 * 1024)

        begin_idx, middle_idx, end_idx = frame_indices
        features = []

        for frame_idx in [begin_idx, middle_idx, end_idx]:
            feat = self._extract_single_frame(video_path, frame_idx)
            if feat is None:
                feat = np.zeros(1024)
            features.append(feat)

        return np.concatenate(features)

    def _extract_single_frame(self, video_path, frame_idx: int):
        """Extract DINO features from a single frame."""
        try:
            import av
            from PIL import Image
        except ImportError:
            return None

        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            if frame_idx > 0:
                time_base = float(stream.time_base)
                avg_frame_rate = float(stream.average_rate)
                if avg_frame_rate > 0:
                    timestamp = int(frame_idx / avg_frame_rate / time_base)
                    container.seek(timestamp, stream=stream)

            current_idx = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if current_idx == frame_idx:
                        img = frame.to_ndarray(format='rgb24')
                        container.close()

                        image = Image.fromarray(img)
                        inputs = self.dino_processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.dino_model(**inputs)
                            features = outputs.last_hidden_state[:, 0, :]

                        return features.cpu().numpy().flatten()

                    current_idx += 1
                    if current_idx > frame_idx:
                        break

            container.close()
            return None

        except Exception as e:
            if 'container' in locals():
                container.close()
            return None


def predict_trajectory(model, dataset, traj_idx, window_size, stride, max_token_length, boundary_tolerance, device, use_ray=True, num_workers=3, batch_size=32):
    """
    Run inference on a single trajectory and return predictions.

    Args:
        model: Trained boundary prediction model
        dataset: AgiBotHFDirectDataset
        traj_idx: Index of trajectory to predict
        window_size: Window size used during training
        stride: Stride used during training
        max_token_length: Max token length
        boundary_tolerance: Boundary tolerance
        device: Device to run on
        use_ray: If True, use Ray for parallel DINO extraction
        num_workers: Number of Ray workers for DINO extraction
        batch_size: Batch size for model inference

    Returns:
        Dictionary with predictions, ground truth, and metadata
    """
    from embedding.fast_droid_embeddings import FastDROIDEmbedder
    from benchmark.metrics import detect_boundaries

    model.eval()

    # Get trajectory
    traj = dataset.trajectories[traj_idx]
    print(f"\nRunning inference on trajectory: {traj.trajectory_id}")
    print(f"  Task: {traj.task_name}")
    print(f"  Segments: {traj.num_segments}")

    # Load proprioception
    embedder = FastDROIDEmbedder(
        window_size=window_size,
        stride=stride,
        use_dino=True,
        max_token_length=max_token_length,
        device=device
    )

    proprio_path = traj.proprioception_path
    if not proprio_path or not proprio_path.exists():
        print("  Error: No proprioception data found")
        return None

    proprio = embedder.load_proprioception(proprio_path)
    if proprio is None or len(proprio) < window_size:
        print("  Error: Invalid proprioception data")
        return None

    # Get ground truth
    segment_labels = traj.get_segment_labels()
    gt_boundaries = detect_boundaries(segment_labels)

    # Get video path
    video_path = None
    if traj.video_paths:
        video_path = (
            traj.video_paths.get('head_color') or
            traj.video_paths.get('hand_left_color') or
            list(traj.video_paths.values())[0]
        )

    if not video_path or not video_path.exists():
        print("  Error: No video found")
        return None

    # Create windows
    windows = embedder.create_windows(proprio)
    print(f"  Processing {len(windows)} windows...")

    # Prepare all tokenized actions and proprio stats first
    all_tokenized = []
    all_proprio_stats = []
    center_frames = []
    frame_indices_list = []

    print("  Preparing features...")
    for window, center_frame_idx in tqdm(windows, desc="  Tokenizing", leave=False):
        center_frames.append(center_frame_idx)

        # Tokenize actions
        if len(window) > 1:
            actions = np.diff(window, axis=0)
            action_str = " ".join([f"{x:.4f}" for x in actions.flatten()])
            tokens = embedder.tokenizer(action_str, truncation=False, return_tensors='np')
            token_ids = tokens['input_ids'].flatten()

            if len(token_ids) < max_token_length:
                pad_id = embedder.tokenizer.pad_token_id or 0
                token_embedding = np.pad(
                    token_ids,
                    (0, max_token_length - len(token_ids)),
                    constant_values=pad_id
                )
            else:
                token_embedding = token_ids[:max_token_length]
        else:
            token_embedding = np.zeros(max_token_length)

        all_tokenized.append(token_embedding)

        # Statistical features
        proprio_stats = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            np.percentile(window, 25, axis=0),
            np.percentile(window, 75, axis=0)
        ])
        all_proprio_stats.append(proprio_stats)

        # Frame indices for DINO
        begin_idx = center_frame_idx - window_size // 2
        end_idx = begin_idx + window_size
        begin_frame = max(0, begin_idx)
        middle_frame = center_frame_idx
        end_frame = min(len(proprio) - 1, end_idx - 1)
        frame_indices_list.append((begin_frame, middle_frame, end_frame))

    # Extract DINO features in parallel using Ray
    if use_ray:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        print(f"  Extracting DINO features with Ray ({num_workers} workers)...")
        actors = [InferenceDinoExtractor.remote(device=device) for _ in range(num_workers)]

        # Split work across actors
        dino_batch_size = max(1, len(frame_indices_list) // num_workers)
        futures = []
        for i in range(0, len(frame_indices_list), dino_batch_size):
            batch = frame_indices_list[i:i + dino_batch_size]
            actor = actors[(i // dino_batch_size) % num_workers]
            future = actor.extract_features_batch.remote(str(video_path), batch)
            futures.append(future)

        # Collect DINO features
        all_dino_features = []
        for future in tqdm(futures, desc="  DINO extraction", leave=False):
            batch_features = ray.get(future)
            all_dino_features.extend(batch_features)

        # Cleanup
        for actor in actors:
            ray.kill(actor)
    else:
        print("  Extracting DINO features sequentially...")
        all_dino_features = []
        for frame_indices in tqdm(frame_indices_list, desc="  DINO extraction", leave=False):
            begin_idx, middle_idx, end_idx = frame_indices
            dino_features = []
            for frame_idx in [begin_idx, middle_idx, end_idx]:
                feat = embedder.extract_dino_features(video_path, frame_idx)
                if feat is None:
                    feat = np.zeros(1024)
                dino_features.append(feat)
            all_dino_features.append(np.concatenate(dino_features))

    # Batch model inference
    print(f"  Running model inference (batch_size={batch_size})...")
    predictions = []
    probabilities = []

    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="  Model inference", leave=False):
            batch_end = min(i + batch_size, len(windows))

            # Prepare batch
            batch_tokenized = torch.tensor(
                np.stack(all_tokenized[i:batch_end]),
                dtype=torch.long
            ).to(device)

            batch_proprio = torch.tensor(
                np.stack(all_proprio_stats[i:batch_end]),
                dtype=torch.float32
            ).to(device)

            batch_dino = torch.tensor(
                np.stack(all_dino_features[i:batch_end]),
                dtype=torch.float32
            ).to(device)

            # Batch prediction
            logits = model(batch_tokenized, batch_proprio, batch_dino)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs_boundary = probs[:, 1].cpu().numpy()

            predictions.extend(preds.tolist())
            probabilities.extend(probs_boundary.tolist())

    return {
        'trajectory_id': traj.trajectory_id,
        'task_name': traj.task_name,
        'video_path': video_path,
        'center_frames': center_frames,
        'predictions': predictions,
        'probabilities': probabilities,
        'gt_boundaries': gt_boundaries,
        'segment_labels': segment_labels,
        'total_frames': len(segment_labels)
    }


def create_visualization_video(result, output_path, window_size, boundary_tolerance):
    """
    Create a video with predictions and ground truth overlaid.

    Args:
        result: Result dictionary from predict_trajectory
        output_path: Path to save output video
        window_size: Window size used
        boundary_tolerance: Boundary tolerance used
    """
    import av

    video_path = result['video_path']
    center_frames = result['center_frames']
    predictions = result['predictions']
    probabilities = result['probabilities']
    gt_boundaries = result['gt_boundaries']
    segment_labels = result['segment_labels']

    print(f"\nCreating visualization video...")
    print(f"  Input video: {video_path}")
    print(f"  Output video: {output_path}")

    # Open input video with PyAV
    try:
        input_container = av.open(str(video_path))
        input_stream = input_container.streams.video[0]
        fps = float(input_stream.average_rate)
        width = input_stream.width
        height = input_stream.height
        total_frames = input_stream.frames
    except Exception as e:
        print(f"  Error: Could not open video: {e}")
        return False

    print(f"  Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create output writer with PyAV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_container = av.open(str(output_path), mode='w')
        # Convert fps to integer or Fraction for PyAV
        from fractions import Fraction
        fps_fraction = Fraction(fps).limit_denominator(1000)
        output_stream = output_container.add_stream('h264', rate=fps_fraction)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
    except Exception as e:
        print(f"  Error: Could not create output video: {e}")
        input_container.close()
        return False

    # Create lookup dictionaries
    pred_dict = {center_frames[i]: (predictions[i], probabilities[i])
                 for i in range(len(center_frames))}
    gt_boundaries_set = set(gt_boundaries)

    # Expand GT boundaries with tolerance
    gt_boundary_frames = set()
    for b in gt_boundaries:
        for offset in range(-boundary_tolerance, boundary_tolerance + 1):
            frame_idx = b + offset
            if 0 <= frame_idx < len(segment_labels):
                gt_boundary_frames.add(frame_idx)

    print(f"  Processing {total_frames} frames...")

    frame_idx = 0
    for packet in tqdm(input_container.demux(input_stream), total=total_frames, desc="  Rendering"):
        for in_frame in packet.decode():
            # Convert frame to numpy array
            img = in_frame.to_ndarray(format='rgb24')

            # Convert to BGR for OpenCV operations
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Get current segment and GT  probability
            if frame_idx < len(segment_labels):
                current_segment = segment_labels[frame_idx]
                is_gt_boundary = frame_idx in gt_boundaries_set
                gt_prob = 1.0 if is_gt_boundary else 0.0
            else:
                current_segment = -1
                is_gt_boundary = False
                gt_prob = 0.0

            # Get prediction probability if available
            pred_prob = None
            if frame_idx in pred_dict:
                _, pred_prob = pred_dict[frame_idx]

            # Draw segment info
            y_offset = 30

            # Current segment number
            text = f"Segment: #{current_segment}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y_offset += 35

            # Ground truth probability
            gt_text = f"GT Probability: {gt_prob:.3f}"
            # Color gradient based on GT probability
            gt_color_intensity = int(255 * gt_prob)
            gt_color = (0, gt_color_intensity, 255 - gt_color_intensity)
            cv2.putText(frame, gt_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, gt_color, 2)
            y_offset += 35

            # Prediction probability
            if pred_prob is not None:
                pred_text = f"Pred Probability: {pred_prob:.3f}"
                # Color gradient based on prediction probability
                pred_color_intensity = int(255 * pred_prob)
                pred_color_vis = (0, pred_color_intensity, 255 - pred_color_intensity)
                cv2.putText(frame, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, pred_color_vis, 2)
                y_offset += 35

                # Show probability difference
                prob_diff = abs(pred_prob - gt_prob)
                diff_text = f"Diff: {prob_diff:.3f}"
                # Color: green if close, red if far
                if prob_diff < 0.2:
                    diff_color = (0, 255, 0)  # Green - good match
                elif prob_diff < 0.5:
                    diff_color = (0, 165, 255)  # Orange - moderate
                else:
                    diff_color = (0, 0, 255)  # Red - bad match
                cv2.putText(frame, diff_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, diff_color, 2)

                # Draw indicator circle with color based on difference
                cv2.circle(frame, (width - 30, 30), 15, diff_color, -1)

                # Draw probability bars on the right side
                bar_width = 150
                bar_height = 20
                bar_x = width - bar_width - 10
                bar_y_gt = 70
                bar_y_pred = 100

                # GT probability bar
                cv2.rectangle(frame, (bar_x, bar_y_gt), (bar_x + bar_width, bar_y_gt + bar_height),
                            (100, 100, 100), -1)
                if gt_prob > 0:
                    cv2.rectangle(frame, (bar_x, bar_y_gt),
                                (bar_x + int(bar_width * gt_prob), bar_y_gt + bar_height),
                                gt_color, -1)
                cv2.putText(frame, "GT", (bar_x - 30, bar_y_gt + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Pred probability bar
                cv2.rectangle(frame, (bar_x, bar_y_pred), (bar_x + bar_width, bar_y_pred + bar_height),
                            (100, 100, 100), -1)
                if pred_prob > 0:
                    cv2.rectangle(frame, (bar_x, bar_y_pred),
                                (bar_x + int(bar_width * pred_prob), bar_y_pred + bar_height),
                                pred_color_vis, -1)
                cv2.putText(frame, "Pred", (bar_x - 40, bar_y_pred + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert back to RGB for encoding
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create AV frame and encode
            out_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)

            frame_idx += 1

    # Flush encoder
    for packet in output_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()

    print(f"  Video saved to {output_path}")
    return True


def main():
    # Configuration
    cache_dir = Path("/home/syx/ucsf/task_seg/agibot_full_data")
    task_ids = [327]
    max_trajectories = 50  # Start with fewer for testing

    window_size = 30
    stride = 30
    max_token_length = 600
    boundary_tolerance = 2

    # Label smoothing configuration
    smoother_type = 'gaussian'  # Options: 'gaussian', 'triangular', 'uniform', 'exponential', 'none'
    smoother_sigma = 2.0  # For Gaussian smoothing

    # Post-processing configuration
    nms_method = 'nms'  # Options: 'nms', 'threshold', 'adaptive', 'topk'
    nms_threshold = 0.3
    nms_min_distance = 10

    # Inference configuration
    run_inference = True  # Set to False to skip slow inference step
    inference_num_workers = 1  # Reduce workers if running into issues
    inference_batch_size = 64  # Larger batch for faster inference

    batch_size = 8  # Smaller batch size for testing
    num_epochs = 2  # Just 2 epochs for initial testing
    learning_rate = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path("./training_output")
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Weights & Biases in offline mode
    wandb.init(
        project="task-segmentation",
        name="boundary-prediction-training",
        # mode="offline",  # Run in offline mode
        # dir=str(wandb_dir),  # Save logs locally
        config={
            "task_ids": task_ids,
            "max_trajectories": max_trajectories,
            "window_size": window_size,
            "stride": stride,
            "max_token_length": max_token_length,
            "boundary_tolerance": boundary_tolerance,
            "smoother_type": smoother_type,
            "smoother_sigma": smoother_sigma,
            "nms_method": nms_method,
            "nms_threshold": nms_threshold,
            "nms_min_distance": nms_min_distance,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "device": device,
            "hidden_dims": [512, 256, 128],
            "dropout": 0.3,
            "loss_type": "SoftFocalLoss",
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0
        }
    )

    print("="*70)
    print("Boundary Prediction Model Training")
    print("="*70)

    # Check if cached training dataset exists
    dataset_cache_path = output_dir / 'cached_dataset.pkl'
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = None  # Will load only if needed

    # Step 1: Load or generate training data (with caching)
    if dataset_cache_path.exists():
        print(f"\nStep 1: Found cached training dataset at {dataset_cache_path}")
        print("Loading from cache (skipping dataset loading)...")
        # Create a minimal dataset object just for load_from_disk
        # We don't need the full dataset since everything is cached
        dataset = AgiBotHFDirectDataset(
            task_ids=task_ids,
            max_trajectories=max_trajectories,
            cache_dir=str(cache_dir),
            download_videos=False,  # Don't download videos when using cache
            download_proprio=False  # Don't download proprio when using cache
        )
        train_dataset = SegmentBoundaryDataset.load_from_disk(
            dataset_cache_path,
            dataset=dataset,
            device=device
        )
    else:
        print(f"\nStep 1: Loading AgiBotWorld dataset from HuggingFace...")
        print("(No cached training dataset found, will generate from scratch)")
        dataset = AgiBotHFDirectDataset(
            task_ids=task_ids,
            max_trajectories=max_trajectories,
            cache_dir=str(cache_dir),
            download_videos=True,  # Need videos for DINO
            download_proprio=True  # Need H5 for proprio
        )
        dataset.print_statistics()

        print("\nStep 2: Generating training samples...")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride}")
        print(f"  Boundary tolerance: {boundary_tolerance}")
        print(f"  Label smoother: {smoother_type} (sigma={smoother_sigma})")
        print("(This will pre-compute DINO features and may take some time)")

        # Create label smoother
        if smoother_type == 'gaussian':
            label_smoother = create_smoother('gaussian', sigma=smoother_sigma, peak_value=1.0)
        elif smoother_type == 'triangular':
            label_smoother = create_smoother('triangular', width=int(smoother_sigma), peak_value=1.0)
        elif smoother_type == 'uniform':
            label_smoother = create_smoother('uniform', window_size=int(smoother_sigma), value=1.0)
        elif smoother_type == 'exponential':
            label_smoother = create_smoother('exponential', decay_rate=smoother_sigma, peak_value=1.0)
        else:
            label_smoother = create_smoother('none')

        train_dataset = SegmentBoundaryDataset(
            dataset=dataset,
            window_size=window_size,
            stride=stride,
            max_token_length=max_token_length,
            boundary_tolerance=boundary_tolerance,
            device=device,
            precompute_features=True,  # Pre-compute DINO features
            use_ray=True,  # Enable Ray parallelization
            num_ray_workers=4,  # Use 4 parallel workers for DINO extraction
            label_smoother=label_smoother  # Use configured smoother
        )
        # Save for next time
        print(f"\nSaving dataset cache to {dataset_cache_path}...")
        train_dataset.save_to_disk(dataset_cache_path)

    # Step 2: Create data loader
    print("\nStep 2: Creating data loaders...")
    train_loader, _ = create_dataloaders(
        train_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader)}")

    # Step 3: Create model
    print("\nStep 3: Creating model...")
    model = create_model(
        vocab_size=32000,  # Fast-DROID tokenizer
        max_token_length=max_token_length,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    # Step 4: Setup training
    print("\nStep 4: Setup training...")

    # Use Soft Focal Loss to handle smooth probability labels and class imbalance
    criterion = SoftFocalLoss(alpha=0.25, gamma=2.0)
    print("  Loss function: Soft Focal Loss (alpha=0.25, gamma=2.0)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    print(f"  Optimizer: AdamW (lr={learning_rate})")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    print("  Scheduler: ReduceLROnPlateau")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Add validation loader if you have val data
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir / 'checkpoints',
        log_dir=output_dir / 'logs'
    )

    # Step 5: Train
    print(f"\nStep 5: Training for {num_epochs} epochs...")
    print("="*70)

    trainer.train(num_epochs=num_epochs, save_every=5)

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Logs saved to: {output_dir / 'logs'}")
    print(f"Wandb logs saved to: {wandb_dir}")
    print("="*70)

    # Step 6: Run inference on one trajectory and create visualization
    if run_inference:
        print(f"\nStep 6: Running inference on a sample trajectory...")
        print("="*70)

        # Load best model
        best_model_path = output_dir / 'checkpoints' / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print("Warning: Best model not found, using final model state")

        # Load full dataset if not already loaded with videos (needed for inference)
        # Only load 1 trajectory for inference to save time
        needs_full_dataset = (
            dataset is None or
            len(dataset.trajectories) == 0 or
            not dataset.trajectories[0].video_paths or
            len(dataset.trajectories[0].video_paths) == 0
        )

        if needs_full_dataset:
            print("\nLoading 1 trajectory for inference (with video and proprio)...")
            dataset = AgiBotHFDirectDataset(
                task_ids=task_ids,
                max_trajectories=1,  # Only need 1 trajectory for inference
                cache_dir=str(cache_dir),
                download_videos=True,  # Need videos for visualization
                download_proprio=True  # Need H5 for inference
            )

        # Select first trajectory
        traj_idx = 0
        result = predict_trajectory(
            model=model,
            dataset=dataset,
            traj_idx=traj_idx,
            window_size=window_size,
            stride=stride,
            max_token_length=max_token_length,
            boundary_tolerance=boundary_tolerance,
            device=device,
            use_ray=True,
            num_workers=inference_num_workers,
            batch_size=inference_batch_size
        )

        if result:
            # Get probabilities and apply NMS for boundary detection
            probabilities = np.array(result['probabilities'])
            center_frames = np.array(result['center_frames'])
            gt_boundaries = result['gt_boundaries']

            # Create full probability array for the trajectory
            full_probs = np.zeros(result['total_frames'])
            for i, frame_idx in enumerate(center_frames):
                if frame_idx < len(full_probs):
                    full_probs[frame_idx] = probabilities[i]

            # Apply NMS to detect boundaries
            print(f"\nApplying NMS for boundary detection...")
            print(f"  Method: {nms_method}")
            print(f"  Threshold: {nms_threshold}")
            print(f"  Min distance: {nms_min_distance}")

            detected_boundaries, detected_probs = detect_boundaries_nms(
                full_probs,
                method=nms_method,
                threshold=nms_threshold,
                min_distance=nms_min_distance
            )

            # Calculate boundary detection metrics
            n_detected = len(detected_boundaries)
            n_gt = len(gt_boundaries)

            # Calculate precision and recall with tolerance
            tolerance = boundary_tolerance
            true_positives = 0
            for det_b in detected_boundaries:
                # Check if this detected boundary is within tolerance of any GT boundary
                if any(abs(det_b - gt_b) <= tolerance for gt_b in gt_boundaries):
                    true_positives += 1

            precision = true_positives / n_detected if n_detected > 0 else 0
            recall = true_positives / n_gt if n_gt > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\nInference Results:")
            print(f"  Trajectory: {result['trajectory_id']}")
            print(f"  Task: {result['task_name']}")
            print(f"  Total frames: {result['total_frames']}")
            print(f"  Total windows: {len(center_frames)}")
            print(f"\nProbability Statistics:")
            print(f"  Mean probability: {probabilities.mean():.4f}")
            print(f"  Max probability: {probabilities.max():.4f}")
            print(f"  Min probability: {probabilities.min():.4f}")
            print(f"  Std probability: {probabilities.std():.4f}")
            print(f"\nBoundary Detection (with NMS):")
            print(f"  Ground truth boundaries: {n_gt}")
            print(f"  Detected boundaries: {n_detected}")
            print(f"  True positives: {true_positives}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Show detected boundaries vs GT
            if n_detected > 0:
                print(f"\nDetected Boundaries (frame: probability):")
                for i, (frame, prob) in enumerate(zip(detected_boundaries, detected_probs)):
                    # Find closest GT boundary
                    if len(gt_boundaries) > 0:
                        closest_gt = min(gt_boundaries, key=lambda x: abs(x - frame))
                        dist = abs(frame - closest_gt)
                        print(f"    {i+1}. Frame {frame}: {prob:.4f} (closest GT: {closest_gt}, dist: {dist})")
                    else:
                        print(f"    {i+1}. Frame {frame}: {prob:.4f}")

            # Create visualization video
            viz_output = output_dir / 'visualization' / f"{result['trajectory_id']}_predictions.mp4"
            success = create_visualization_video(
                result,
                viz_output,
                window_size,
                boundary_tolerance
            )

            if success:
                print(f"\nVisualization complete!")
                print(f"  Video saved to: {viz_output}")
                print(f"  You can watch it to see predictions vs ground truth")

                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'inference/precision': precision,
                        'inference/recall': recall,
                        'inference/f1_score': f1,
                        'inference/n_detected_boundaries': n_detected,
                        'inference/n_gt_boundaries': n_gt,
                        'inference/true_positives': true_positives,
                        'inference/mean_probability': probabilities.mean(),
                        'inference/max_probability': probabilities.max(),
                        'inference/video': wandb.Video(str(viz_output))
                    })
    else:
        print("\nSkipping inference (set run_inference=True to enable)")

    print("\n" + "="*70)
    print("All done!")
    print(f"\nView training progress with:")
    print(f"  - Tensorboard: tensorboard --logdir {output_dir / 'logs'}")
    print(f"  - Wandb (offline): Logs saved in {wandb_dir}")
    print(f"    To sync to cloud later: wandb sync {wandb_dir}")
    print("="*70)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
