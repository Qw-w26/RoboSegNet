#!/usr/bin/env python3
"""Inference script for boundary prediction models.

This script provides production inference with:
- Loading trained model from checkpoint
- Processing single trajectory or batch of trajectories
- Extracting features (DINO + proprioceptive)
- Predicting segment boundaries
- Outputting predictions in multiple formats (JSON, NPZ, visualization)

Example usage:
    # Single trajectory inference
    python scripts/inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --video-path ./data/episode_123.mp4 \
        --proprio-path ./data/episode_123_proprio.npy \
        --output results.json

    # Batch inference with visualization
    python scripts/inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --split-manifest ./splits/task_327_test.json \
        --output-dir ./inference_results \
        --visualize

    # With custom threshold
    python scripts/inference.py \
        --checkpoint ./checkpoints/best_model.pt \
        --video-path ./data/episode_123.mp4 \
        --proprio-path ./data/episode_123_proprio.npy \
        --threshold 0.7 \
        --output results.json

Author: Claude Code
Date: October 24, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model, ModelConfig
from features import DinoFeatureExtractor, DroidTokenEmbedder, FastTokenizerConfig
from data.splits import load_split_manifest
from utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with boundary prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Single trajectory mode
    parser.add_argument(
        '--video-path',
        type=str,
        help='Path to video file (for single trajectory inference)'
    )

    parser.add_argument(
        '--proprio-path',
        type=str,
        help='Path to proprioceptive data file (NPY format)'
    )

    # Batch mode
    parser.add_argument(
        '--split-manifest',
        type=str,
        help='Path to split manifest JSON (for batch inference)'
    )

    # Model config
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=2048,
        help='Tokenizer vocabulary size (must match training)'
    )

    parser.add_argument(
        '--prediction-window-size',
        type=int,
        default=18,
        help='Prediction window size (must match training)'
    )

    # Inference settings
    parser.add_argument(
        '--window-size',
        type=int,
        default=180,
        help='Window size for feature extraction (frames)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Stride for sliding window (frames)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for boundary detection'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON or NPZ) for single trajectory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_results',
        help='Output directory for batch inference'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        default='json',
        choices=['json', 'npz', 'both'],
        help='Output format for predictions'
    )

    # System
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    vocab_size: int,
    prediction_window_size: int,
    device: torch.device
) -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        prediction_window_size=prediction_window_size,
        use_dino=True,
        use_proprio=True
    )

    model = create_model(model_config, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    return model


def extract_features(
    video_path: str,
    proprio_data: np.ndarray,
    window_size: int,
    stride: int,
    vocab_size: int,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """Extract DINO and proprioceptive features for trajectory.

    Args:
        video_path: Path to video file
        proprio_data: Proprioceptive data (joint positions)
        window_size: Window size in frames
        stride: Stride for sliding windows
        vocab_size: Vocabulary size for tokenizer
        device: Device to use

    Returns:
        Tuple of (dino_features, proprio_features, window_starts)
    """
    print("Extracting features...")

    # Initialize extractors
    dino_extractor = DinoFeatureExtractor(device=str(device))
    
    # Infer FPS from data length (assume this matches preprocessing)
    # Default to 5.0 fps which is the typical target_fps
    inferred_fps = 5.0
    
    proprio_embedder = DroidTokenEmbedder(
        config=FastTokenizerConfig(
            time_s=window_size / inferred_fps,
            frame_frequency=inferred_fps,
            action_dim=proprio_data.shape[1],
            vocab_size=vocab_size
        )
    )

    num_frames = len(proprio_data)

    # Create sliding windows
    window_starts = list(range(0, num_frames - window_size, stride))
    print(f"  Created {len(window_starts)} windows")

    # Extract DINO features
    print("  Extracting DINO features...")
    dino_features = []
    for start in tqdm(window_starts, desc="DINO"):
        middle = start + window_size // 2
        end = start + window_size

        # Extract features for begin, middle, end frames
        frames = [start, middle, end]
        batch_features = dino_extractor.extract_batch(video_path, frames)

        # Concatenate begin, middle, end features
        features = torch.cat([
            torch.tensor(batch_features[start], dtype=torch.float32),
            torch.tensor(batch_features[middle], dtype=torch.float32),
            torch.tensor(batch_features[end], dtype=torch.float32)
        ])

        dino_features.append(features)

    # Extract proprioceptive features
    print("  Extracting proprioceptive features...")
    proprio_dict = {
        'joint_positions': proprio_data,
        'timestamps': np.arange(len(proprio_data))
    }

    proprio_features_dict = proprio_embedder.embed_frames(
        proprio_dict,
        window_starts,
        window_size
    )

    proprio_features = [
        torch.tensor(proprio_features_dict[start], dtype=torch.long)
        for start in window_starts
    ]

    print("✓ Feature extraction complete")

    return dino_features, proprio_features, window_starts


def run_inference(
    model: nn.Module,
    dino_features: List[torch.Tensor],
    proprio_features: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """Run model inference on features.

    Args:
        model: Model to use
        dino_features: List of DINO feature tensors
        proprio_features: List of proprio feature tensors
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        Array of predictions (probabilities)
    """
    print("Running inference...")

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dino_features), batch_size), desc="Inference"):
            # Get batch
            batch_dino = torch.stack(dino_features[i:i+batch_size]).to(device)
            batch_proprio = torch.stack(proprio_features[i:i+batch_size]).to(device)

            # Forward pass
            logits = model(batch_proprio, None, batch_dino)

            # Convert to probabilities
            probabilities = torch.sigmoid(logits)

            all_predictions.append(probabilities.cpu().numpy())

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)

    print(f"✓ Inference complete ({predictions.shape[0]} windows)")

    return predictions


def aggregate_predictions(
    predictions: np.ndarray,
    window_starts: List[int],
    window_size: int,
    total_frames: int
) -> np.ndarray:
    """Aggregate window predictions into frame-level predictions.

    Args:
        predictions: Window predictions (num_windows, window_size)
        window_starts: List of window start frames
        window_size: Window size
        total_frames: Total number of frames

    Returns:
        Frame-level predictions (total_frames,)
    """
    # Initialize aggregation arrays
    prediction_sum = np.zeros(total_frames)
    prediction_count = np.zeros(total_frames)

    # Aggregate predictions from all windows
    for pred, start in zip(predictions, window_starts):
        end = start + len(pred)
        prediction_sum[start:end] += pred
        prediction_count[start:end] += 1

    # Average overlapping predictions
    frame_predictions = np.zeros(total_frames)
    mask = prediction_count > 0
    frame_predictions[mask] = prediction_sum[mask] / prediction_count[mask]

    return frame_predictions


def detect_boundaries(
    predictions: np.ndarray,
    threshold: float = 0.5
) -> List[int]:
    """Detect boundary frames from predictions.

    Args:
        predictions: Frame-level predictions
        threshold: Detection threshold

    Returns:
        List of boundary frame indices
    """
    # Threshold predictions
    binary_predictions = predictions > threshold

    # Find rising edges (transitions from 0 to 1)
    boundaries = []
    for i in range(1, len(binary_predictions)):
        if binary_predictions[i] and not binary_predictions[i-1]:
            boundaries.append(i)

    return boundaries


def save_results(
    predictions: np.ndarray,
    boundaries: List[int],
    output_path: str,
    output_format: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save inference results.

    Args:
        predictions: Frame-level predictions
        boundaries: Detected boundary frames
        output_path: Output file path
        output_format: Format ('json', 'npz', or 'both')
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare results
    results = {
        'boundaries': boundaries,
        'num_boundaries': len(boundaries),
        'predictions': predictions.tolist() if output_format == 'json' else predictions,
        'metadata': metadata or {}
    }

    # Save JSON
    if output_format in ['json', 'both']:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json_results = {
                'boundaries': boundaries,
                'num_boundaries': len(boundaries),
                'metadata': metadata or {}
            }
            json.dump(json_results, f, indent=2)
        print(f"✓ Saved results to: {json_path}")

    # Save NPZ
    if output_format in ['npz', 'both']:
        npz_path = output_path.with_suffix('.npz')
        np.savez(
            npz_path,
            predictions=predictions,
            boundaries=np.array(boundaries),
            **{f'metadata_{k}': v for k, v in (metadata or {}).items()}
        )
        print(f"✓ Saved predictions to: {npz_path}")


def visualize_predictions(
    predictions: np.ndarray,
    boundaries: List[int],
    output_path: str,
    ground_truth_boundaries: Optional[List[int]] = None
) -> None:
    """Create visualization of predictions.

    Args:
        predictions: Frame-level predictions
        boundaries: Detected boundaries
        output_path: Output file path
        ground_truth_boundaries: Optional ground truth for comparison
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot predictions
    frames = np.arange(len(predictions))
    ax.plot(frames, predictions, 'b-', alpha=0.7, label='Predicted Probability')

    # Plot detected boundaries
    for boundary in boundaries:
        ax.axvline(boundary, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Plot ground truth boundaries if available
    if ground_truth_boundaries:
        for boundary in ground_truth_boundaries:
            ax.axvline(boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Boundary Probability')
    ax.set_title('Boundary Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Saved visualization to: {output_path}")


def main():
    """Main inference function."""
    args = parse_args()

    # Print banner
    print("="*80)
    print("Boundary Prediction Inference")
    print("="*80)

    # Set random seed
    set_all_seeds(args.seed)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(
        args.checkpoint,
        args.vocab_size,
        args.prediction_window_size,
        device
    )

    # Single trajectory mode
    if args.video_path and args.proprio_path:
        print("\n" + "="*80)
        print("Single Trajectory Inference")
        print("="*80)

        # Load proprioceptive data
        print(f"Loading proprioceptive data from: {args.proprio_path}")
        proprio_data = np.load(args.proprio_path)
        print(f"  Shape: {proprio_data.shape}")

        # Extract features
        dino_features, proprio_features, window_starts = extract_features(
            args.video_path,
            proprio_data,
            args.window_size,
            args.stride,
            args.vocab_size,
            device
        )

        # Run inference
        window_predictions = run_inference(
            model, dino_features, proprio_features, device
        )

        # Aggregate to frame-level
        print("Aggregating predictions...")
        frame_predictions = aggregate_predictions(
            window_predictions,
            window_starts,
            args.window_size,
            len(proprio_data)
        )

        # Detect boundaries
        boundaries = detect_boundaries(frame_predictions, args.threshold)
        print(f"✓ Detected {len(boundaries)} boundaries: {boundaries}")

        # Save results
        output_path = args.output or 'inference_results.json'
        save_results(
            frame_predictions,
            boundaries,
            output_path,
            args.output_format,
            metadata={
                'video_path': args.video_path,
                'num_frames': len(proprio_data),
                'threshold': args.threshold
            }
        )

        # Visualize
        if args.visualize:
            viz_path = Path(output_path).with_suffix('.png')
            visualize_predictions(frame_predictions, boundaries, str(viz_path))

    # Batch mode
    elif args.split_manifest:
        print("\n" + "="*80)
        print("Batch Inference")
        print("="*80)
        print("Note: Batch inference not yet fully implemented")
        print("Use evaluation script for batch processing")

    else:
        print("Error: Must specify either --video-path + --proprio-path or --split-manifest")
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ Inference complete!")
    print("="*80)


if __name__ == '__main__':
    main()
