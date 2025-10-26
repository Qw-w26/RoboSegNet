"""Demo: Peak detection for boundary prediction with distribution-based losses.

This script demonstrates how to extract discrete boundary predictions from
probability distributions, handling overlapping boundaries and multiple peaks.

Run: python examples/peak_detection_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from models.postprocessing import detect_boundaries, non_maximum_suppression


def create_multi_boundary_distribution(window_size=120, boundaries=[30, 32, 80], sigma=5.0):
    """Create synthetic distribution with multiple overlapping boundaries."""
    dist = np.zeros(window_size)

    for boundary in boundaries:
        distances = np.abs(np.arange(window_size) - boundary)
        dist += np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # Normalize to sum to 1 (like dist_ce output)
    dist = dist / dist.sum()

    return dist


def demo_overlapping_boundaries():
    """Demo 1: Handling overlapping boundaries."""
    print("=" * 80)
    print("Demo 1: Overlapping Boundaries (2 frames apart)")
    print("=" * 80)

    # Create distribution with boundaries very close together (30 and 32)
    dist = create_multi_boundary_distribution(boundaries=[30, 32, 80])

    print(f"\nGround truth: boundaries at frames 30, 32, 80")
    print(f"Challenge: Frames 30 and 32 are only 2 frames apart!")
    print(f"Distribution sum: {dist.sum():.4f}")
    print(f"Max probability: {dist.max():.4f}\n")

    # Try different configurations
    configs = [
        ("No prominence", {
            'threshold': 0.01,
            'min_distance': 3,
            'prominence': None
        }),
        ("With prominence", {
            'threshold': 0.01,
            'min_distance': 3,
            'prominence': 0.005
        }),
        ("Higher min_distance", {
            'threshold': 0.01,
            'min_distance': 5,
            'prominence': 0.005
        }),
    ]

    fig, axes = plt.subplots(len(configs) + 1, 1, figsize=(12, 10))

    # Plot distribution
    axes[0].plot(dist, linewidth=2, label='Probability Distribution')
    for b in [30, 32, 80]:
        axes[0].axvline(b, color='green', linestyle='--', alpha=0.7, label='Ground Truth' if b == 30 else None)
    axes[0].set_title('Predicted Distribution (3 ground truth boundaries, 2 very close)')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test each config
    for i, (name, params) in enumerate(configs, 1):
        boundaries, probs = detect_boundaries(dist, method='nms', **params)

        axes[i].plot(dist, linewidth=2, alpha=0.5, label='Distribution')
        axes[i].scatter(boundaries, probs, color='red', s=100, zorder=5, marker='v',
                       label=f'Detected ({len(boundaries)} peaks)')

        for b in [30, 32, 80]:
            axes[i].axvline(b, color='green', linestyle='--', alpha=0.7)

        param_str = f"threshold={params['threshold']}, min_distance={params['min_distance']}, prominence={params['prominence']}"
        axes[i].set_title(f'{name}: {param_str}')
        axes[i].set_ylabel('Probability')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        print(f"{name}:")
        print(f"  Parameters: {param_str}")
        print(f"  Detected {len(boundaries)} boundaries at frames: {list(boundaries)}")
        print(f"  Probabilities: {probs}\n")

    axes[-1].set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig('peak_detection_overlapping.png', dpi=150)
    print("✓ Saved visualization to: peak_detection_overlapping.png\n")
    plt.close()


def demo_parameter_tuning():
    """Demo 2: Parameter tuning for different scenarios."""
    print("=" * 80)
    print("Demo 2: Parameter Tuning")
    print("=" * 80)

    # Create distribution with 3 well-separated boundaries
    dist = create_multi_boundary_distribution(boundaries=[20, 60, 100])

    scenarios = [
        ("Conservative (few false positives)", {
            'threshold': 0.015,
            'prominence': 0.01,
            'min_distance': 10
        }),
        ("Balanced (recommended)", {
            'threshold': 0.01,
            'prominence': 0.005,
            'min_distance': 5
        }),
        ("Sensitive (catch all boundaries)", {
            'threshold': 0.005,
            'prominence': 0.002,
            'min_distance': 3
        }),
    ]

    print("\nGround truth: 3 boundaries at frames 20, 60, 100\n")

    for name, params in scenarios:
        boundaries, probs = detect_boundaries(dist, method='nms', **params)
        print(f"{name}:")
        print(f"  threshold={params['threshold']}, prominence={params['prominence']}, min_distance={params['min_distance']}")
        print(f"  Detected {len(boundaries)} boundaries at: {list(boundaries)}")
        print(f"  Confidences: {[f'{p:.4f}' for p in probs]}\n")


def demo_inference_workflow():
    """Demo 3: Complete inference workflow."""
    print("=" * 80)
    print("Demo 3: Complete Inference Workflow")
    print("=" * 80)

    # Simulate model inference
    print("\n1. Model outputs distribution (after softmax):")
    dist = create_multi_boundary_distribution(boundaries=[25, 75], sigma=6.0)
    print(f"   Distribution shape: {dist.shape}")
    print(f"   Distribution sums to: {dist.sum():.4f}")

    print("\n2. Apply peak detection:")
    boundaries, confidences = detect_boundaries(
        dist,
        method='nms',
        threshold=0.01,
        prominence=0.005,
        min_distance=5
    )

    print(f"   Detected {len(boundaries)} boundaries")
    for i, (frame, conf) in enumerate(zip(boundaries, confidences)):
        print(f"     Boundary {i+1}: frame {frame}, confidence {conf:.4f}")

    print("\n3. Convert to timestamps (@ 5 fps):")
    timestamps = boundaries / 5.0  # Convert frames to seconds
    for i, (frame, ts, conf) in enumerate(zip(boundaries, timestamps, confidences)):
        print(f"     Boundary {i+1}: {ts:.2f}s (frame {frame}), confidence {conf:.4f}")

    return boundaries, confidences


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Peak Detection for Boundary Prediction" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    # Run demos
    demo_overlapping_boundaries()
    print("\n")
    demo_parameter_tuning()
    print("\n")
    demo_inference_workflow()

    print("\n" + "=" * 80)
    print("Summary & Recommendations")
    print("=" * 80)
    print("""
For distribution-based models (dist_ce, kl_div, js_div):

Recommended starting parameters:
  - threshold: 0.01 (1% of probability mass)
  - prominence: 0.005 (0.5% - filters overlapping peaks)
  - min_distance: 5 frames (1 second @ 5fps)

Tuning guidelines:
  ↑ threshold → Fewer, higher-confidence boundaries
  ↑ prominence → Only sharp, distinct peaks (reduces false positives)
  ↑ min_distance → Fewer closely-spaced boundaries

Usage in code:
    from models.postprocessing import detect_boundaries

    # Get distribution from model
    dist = model.predict_distribution(tokens, None, dino)  # (batch, 120)

    # Detect boundaries
    boundaries, confidences = detect_boundaries(
        dist[0],                 # Single sample
        method='nms',
        threshold=0.01,
        prominence=0.005,
        min_distance=5
    )

    # Convert to timestamps
    timestamps = boundaries / fps
    """)
