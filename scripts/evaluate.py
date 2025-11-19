#!/usr/bin/env python3
"""Evaluation script for boundary prediction models.

DEPRECATED: This script uses legacy data loading (data.splits).
For multi-task evaluation, use scripts/evaluate_ablation.py instead.

This script provides comprehensive evaluation with:
- Loading trained model from checkpoint
- Computing all metrics (F1@IoU, classification, temporal, confidence)
- Generating visualizations (confusion matrix, PR/ROC curves, predictions)
- Saving results to structured output directory

Example usage:
    # Evaluate on test set
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --test-split ./splits/task_327_test.json \
        --output-dir ./evaluation_results

    # With custom IoU thresholds
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --test-split ./splits/task_327_test.json \
        --iou-thresholds 0.3 0.5 0.7 0.9

    # Save predictions for analysis
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --test-split ./splits/task_327_test.json \
        --save-predictions


Date: October 24, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model, ModelConfig
from training import BoundaryDataset, TrainingDatasetConfig, collate_dict
from evaluation import ModelEvaluator, EvaluationConfig
from data.splits import load_split_manifest
from utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate boundary prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Data
    parser.add_argument(
        '--test-split',
        type=str,
        required=True,
        help='Path to test split manifest JSON'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Cache directory for features'
    )

    # Model config (override if needed)
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

    # Evaluation settings
    parser.add_argument(
        '--iou-thresholds',
        type=float,
        nargs='+',
        default=[0.3, 0.5, 0.7],
        help='IoU thresholds for F1@IoU metric'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for binary predictions'
    )

    parser.add_argument(
        '--num-visualizations',
        type=int,
        default=10,
        help='Number of prediction visualizations to save'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save raw predictions to NPZ file'
    )

    parser.add_argument(
        '--save-metrics-csv',
        action='store_true',
        default=True,
        help='Save metrics to CSV file'
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
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    prediction_window_size: int,
    device: torch.device
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Vocabulary size
        prediction_window_size: Prediction window size
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"\nLoading model from checkpoint: {checkpoint_path}")

    # Create model with same config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        prediction_window_size=prediction_window_size,
        use_dino=True,
        use_proprio=True
    )

    model = create_model(model_config, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_metric' in checkpoint:
            print(f"Best metric: {checkpoint['best_metric']:.4f}")
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def create_test_dataset(
    args: argparse.Namespace,
    device: torch.device
) -> BoundaryDataset:
    """Create test dataset.

    Args:
        args: Command line arguments
        device: Device for feature extraction

    Returns:
        Test dataset
    """
    print("\n" + "="*80)
    print("Creating test dataset...")
    print("="*80)

    # Load test split
    print(f"Loading test split from {args.test_split}")
    test_manifest = load_split_manifest(args.test_split)
    test_trajectories = test_manifest.trajectories

    print(f"Found {len(test_trajectories)} test trajectories")

    # Dataset configuration
    dataset_config = TrainingDatasetConfig(
        window_size=180,
        stride=10,
        vocab_size=args.vocab_size,
        num_workers=args.num_workers,
        device=str(device),
        ray_worker_device=str(device),
        cache_path=args.cache_dir
    )

    print(f"Creating test dataset...")
    test_dataset = BoundaryDataset(test_trajectories, dataset_config)

    print(f"Test dataset: {len(test_dataset)} samples")

    return test_dataset


def create_test_loader(
    test_dataset: BoundaryDataset,
    args: argparse.Namespace
) -> DataLoader:
    """Create test data loader.

    Args:
        test_dataset: Test dataset
        args: Command line arguments

    Returns:
        Test data loader
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dict,
        pin_memory=True
    )

    print(f"Test batches: {len(test_loader)}")

    return test_loader


def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    args: argparse.Namespace
) -> dict:
    """Run comprehensive evaluation.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        args: Command line arguments

    Returns:
        Dictionary of evaluation results
    """
    print("\n" + "="*80)
    print("Running evaluation...")
    print("="*80)

    # Create evaluation config
    eval_config = EvaluationConfig(
        iou_thresholds=args.iou_thresholds,
        confidence_threshold=args.confidence_threshold,
        save_predictions=args.save_predictions,
        save_visualizations=True,
        save_metrics_csv=args.save_metrics_csv,
        num_visualizations=args.num_visualizations,
        device=str(args.device)
    )

    # Create evaluator
    evaluator = ModelEvaluator(model, eval_config)

    # Run evaluation
    output_dir = Path(args.output_dir)
    results = evaluator.evaluate(
        test_loader,
        save_dir=str(output_dir),
        visualize=True
    )

    return results


def print_results_summary(results: dict) -> None:
    """Print evaluation results summary.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Classification metrics
    print("\n📊 Classification Metrics:")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall:    {results.get('recall', 0):.4f}")
    print(f"  F1 Score:  {results.get('f1_score', 0):.4f}")
    print(f"  Accuracy:  {results.get('accuracy', 0):.4f}")

    # F1@IoU metrics
    print("\n🎯 F1@IoU Metrics:")
    for key in sorted(results.keys()):
        if key.startswith('f1_iou_'):
            threshold = key.split('_')[-1]
            f1 = results[key].get('f1', 0)
            precision = results[key].get('precision', 0)
            recall = results[key].get('recall', 0)
            print(f"  IoU={threshold}: F1={f1:.4f} (P={precision:.4f}, R={recall:.4f})")

    # Confidence metrics
    if 'auc_roc' in results:
        print("\n📈 Confidence Metrics:")
        print(f"  AUC-ROC: {results['auc_roc']:.4f}")
        print(f"  AUC-PR:  {results['auc_pr']:.4f}")

    # Temporal metrics
    if 'temporal_precision' in results:
        print("\n⏱️  Temporal Metrics:")
        print(f"  Temporal Precision: {results['temporal_precision']:.4f}")
        print(f"  Temporal Recall:    {results['temporal_recall']:.4f}")
        print(f"  Temporal F1:        {results['temporal_f1']:.4f}")

    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Print banner
    print("="*80)
    print("Boundary Prediction Model Evaluation")
    print("="*80)

    # Set random seed
    set_all_seeds(args.seed)
    print(f"Random seed: {args.seed}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        args.vocab_size,
        args.prediction_window_size,
        device
    )

    # Create test dataset
    test_dataset = create_test_dataset(args, device)

    # Create test loader
    test_loader = create_test_loader(test_dataset, args)

    # Run evaluation
    results = run_evaluation(model, test_loader, args)

    # Print summary
    print_results_summary(results)

    # Print output location
    print(f"\n✅ Evaluation complete!")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - metrics.json (all metrics)")
    print(f"  - metrics_summary.txt (formatted summary)")
    if args.save_metrics_csv:
        print(f"  - metrics.csv (for analysis)")
    if args.save_predictions:
        print(f"  - predictions.npz (raw predictions)")
    print(f"  - visualizations/ (plots)")


if __name__ == '__main__':
    main()
