"""
Example script for training the boundary prediction model.

This script demonstrates how to:
1. Load the AgiBotWorld dataset
2. Generate training data with DINO features and tokenized proprio
3. Train an MLP model to predict segment boundaries
"""

from pathlib import Path
import torch
import wandb
import numpy as np
import cv2
from tqdm import tqdm

from data import AgiBotHFDirectDataset
from training import SegmentBoundaryDataset, create_dataloaders, Trainer
from model import create_model, FocalLoss


def predict_trajectory(model, dataset, traj_idx, window_size, stride, max_token_length, boundary_tolerance, device):
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

    predictions = []
    probabilities = []
    center_frames = []

    print(f"  Processing {len(windows)} windows...")

    with torch.no_grad():
        for window, center_frame_idx in tqdm(windows, desc="  Predicting"):
            center_frames.append(center_frame_idx)

            # Tokenize actions
            if len(window) > 1:
                actions = np.diff(window, axis=0)
                action_str = " ".join([f"{x:.4f}" for x in actions.flatten()])
                tokens = embedder.tokenizer(action_str, truncation=False, return_tensors='np')
                token_ids = tokens['input_ids'].flatten()

                # Pad/truncate
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

            # Statistical features
            proprio_stats = np.concatenate([
                window.mean(axis=0),
                window.std(axis=0),
                window.min(axis=0),
                window.max(axis=0),
                np.percentile(window, 25, axis=0),
                np.percentile(window, 75, axis=0)
            ])

            # Extract DINO features
            begin_idx = center_frame_idx - window_size // 2
            end_idx = begin_idx + window_size
            begin_frame = max(0, begin_idx)
            middle_frame = center_frame_idx
            end_frame = min(len(proprio) - 1, end_idx - 1)

            dino_features = []
            for frame_idx in [begin_frame, middle_frame, end_frame]:
                feat = embedder.extract_dino_features(video_path, frame_idx)
                if feat is None:
                    feat = np.zeros(1024)
                dino_features.append(feat)
            dino_features = np.concatenate(dino_features)

            # Prepare inputs
            tokenized_actions = torch.tensor(token_embedding, dtype=torch.long).unsqueeze(0).to(device)
            proprio_stats_t = torch.tensor(proprio_stats, dtype=torch.float32).unsqueeze(0).to(device)
            dino_features_t = torch.tensor(dino_features, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict
            logits = model(tokenized_actions, proprio_stats_t, dino_features_t)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            prob = probs[0, 1].item()

            predictions.append(pred)
            probabilities.append(prob)

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
        output_stream = output_container.add_stream('h264', rate=fps)
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

            # Get current segment
            if frame_idx < len(segment_labels):
                current_segment = segment_labels[frame_idx]
                gt_label = 1 if frame_idx in gt_boundary_frames else 0
            else:
                current_segment = -1
                gt_label = 0

            # Get prediction if available
            pred_label = None
            pred_prob = None
            if frame_idx in pred_dict:
                pred_label, pred_prob = pred_dict[frame_idx]

            # Draw segment info
            y_offset = 30

            # Current segment number
            text = f"Segment: #{current_segment}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y_offset += 35

            # Ground truth
            gt_text = f"GT: {'Boundary' if gt_label else 'Non-boundary'}"
            gt_color = (0, 255, 255) if gt_label else (200, 200, 200)
            cv2.putText(frame, gt_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, gt_color, 2)
            y_offset += 35

            # Prediction
            if pred_label is not None:
                pred_text = f"Pred: {'Boundary' if pred_label else 'Non-boundary'} ({pred_prob:.3f})"
                # Color: green if correct, red if incorrect
                if pred_label == gt_label:
                    pred_color = (0, 255, 0)  # Green
                else:
                    pred_color = (0, 0, 255)  # Red
                cv2.putText(frame, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, pred_color, 2)

                # Draw indicator circle
                cv2.circle(frame, (width - 30, 30), 15, pred_color, -1)

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
    max_trajectories = 5  # Start with fewer for testing

    window_size = 15
    stride = 1
    max_token_length = 600
    boundary_tolerance = 2

    batch_size = 8  # Smaller batch size for testing
    num_epochs = 2  # Just 2 epochs for initial testing
    learning_rate = 1e-3

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
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "device": device,
            "hidden_dims": [512, 256, 128],
            "dropout": 0.3,
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
        print("(This will pre-compute DINO features and may take some time)")

        train_dataset = SegmentBoundaryDataset(
            dataset=dataset,
            window_size=window_size,
            stride=stride,
            max_token_length=max_token_length,
            boundary_tolerance=boundary_tolerance,
            device=device,
            precompute_features=True,  # Pre-compute DINO features
            use_ray=True,  # Enable Ray parallelization
            num_ray_workers=4  # Use 4 parallel workers for DINO extraction
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
    print("\nStep 4: Setting up training...")

    # Use Focal Loss to handle class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("  Loss function: Focal Loss (alpha=0.25, gamma=2.0)")

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
        device=device
    )

    if result:
        # Calculate metrics
        predictions = np.array(result['predictions'])
        center_frames = np.array(result['center_frames'])
        gt_boundaries_set = set(result['gt_boundaries'])

        # Expand boundaries with tolerance for evaluation
        gt_boundary_frames = set()
        for b in result['gt_boundaries']:
            for offset in range(-boundary_tolerance, boundary_tolerance + 1):
                frame_idx = b + offset
                if 0 <= frame_idx < result['total_frames']:
                    gt_boundary_frames.add(frame_idx)

        gt_labels = [1 if f in gt_boundary_frames else 0 for f in center_frames]

        # Calculate accuracy
        accuracy = (predictions == np.array(gt_labels)).mean()
        n_pred_boundaries = predictions.sum()
        n_gt_boundaries = len(result['gt_boundaries'])

        print(f"\nInference Results:")
        print(f"  Trajectory: {result['trajectory_id']}")
        print(f"  Task: {result['task_name']}")
        print(f"  Total windows: {len(predictions)}")
        print(f"  Predicted boundaries: {n_pred_boundaries}")
        print(f"  Ground truth boundaries: {n_gt_boundaries}")
        print(f"  Accuracy: {accuracy:.4f}")

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
                    'inference/accuracy': accuracy,
                    'inference/n_pred_boundaries': n_pred_boundaries,
                    'inference/n_gt_boundaries': n_gt_boundaries,
                    'inference/video': wandb.Video(str(viz_output))
                })

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
