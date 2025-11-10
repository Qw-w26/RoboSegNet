#!/usr/bin/env python3
"""Plot frequency sensitivity analysis results.

Creates 4 plots showing the relationship between frequency and model performance:
1. Visual frequency (in-distribution)
2. Visual frequency (out-of-distribution)
3. Proprio frequency (in-distribution)
4. Proprio frequency (out-of-distribution)

Each plot compares multi-modal (baseline) and pure (single-modality) models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# Set seaborn style for presentation
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.4)


def extract_frequency(ablation_name: str) -> float:
    """Extract frequency value from ablation name.

    Args:
        ablation_name: Name like "Visual 6 Frames" or "Proprio 5Hz"

    Returns:
        Frequency value (frames for visual, Hz for proprio)
    """
    # Try to extract visual frames
    match = re.search(r'(\d+)\s+Frames?', ablation_name)
    if match:
        return float(match.group(1))

    # Try to extract proprio Hz
    match = re.search(r'(\d+)\s*Hz', ablation_name)
    if match:
        return float(match.group(1))

    return None


def categorize_ablation(ablation_name: str) -> tuple:
    """Categorize ablation by modality and model type.

    Args:
        ablation_name: Name like "Visual 6 Frames" or "Pure Proprio 5Hz"

    Returns:
        Tuple of (modality, model_type) where:
        - modality: 'visual' or 'proprio'
        - model_type: 'multi-modal' or 'pure'
    """
    is_pure = ablation_name.startswith('Pure')
    model_type = 'Pure' if is_pure else 'Multi-modal'

    if 'Visual' in ablation_name or 'Frames' in ablation_name:
        modality = 'visual'
    elif 'Proprio' in ablation_name or 'Hz' in ablation_name:
        modality = 'proprio'
    else:
        modality = 'unknown'

    return modality, model_type


def plot_frequency_relationship(
    df: pd.DataFrame,
    modality: str,
    eval_type: str,
    metric: str = 'F1@2.0',
    output_path: Path = None
):
    """Plot frequency vs performance for a specific modality and evaluation type.

    Args:
        df: DataFrame with ablation results
        modality: 'visual' or 'proprio'
        eval_type: 'indist' or 'ood'
        metric: Metric to plot on y-axis
        output_path: Path to save the plot
    """
    # Filter by modality
    data = []
    for _, row in df.iterrows():
        ablation_name = row['ablation']
        mod, model_type = categorize_ablation(ablation_name)

        if mod != modality:
            continue

        freq = extract_frequency(ablation_name)
        if freq is None:
            continue

        data.append({
            'frequency': freq,
            'metric': row[metric],
            'model_type': model_type,
            'ablation': ablation_name
        })

    if not data:
        print(f"No data for {modality} {eval_type}")
        return

    plot_df = pd.DataFrame(data)

    # Create plot with larger figure size for presentations
    plt.figure(figsize=(12, 8))

    # Plot multi-modal and pure separately
    for model_type in ['Multi-modal', 'Pure']:
        subset = plot_df[plot_df['model_type'] == model_type]
        if len(subset) > 0:
            # Sort by frequency
            subset = subset.sort_values('frequency')

            # Plot with markers and lines (larger for visibility)
            plt.plot(
                subset['frequency'],
                subset['metric'],
                marker='o',
                markersize=14,
                label=model_type,
                linewidth=3
            )

    # Labels (no title for cleaner presentation)
    freq_unit = 'Frames' if modality == 'visual' else 'Hz'
    eval_name = 'In-Distribution' if eval_type == 'indist' else 'Out-of-Distribution'

    plt.xlabel(f'{modality.capitalize()} Frequency ({freq_unit})', fontsize=22, fontweight='bold')
    plt.ylabel(metric, fontsize=22, fontweight='bold')

    # Increase tick label sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Generate all 4 plots."""
    results_dir = Path('./frequency_sensitivity_results')

    # Load data
    indist_df = pd.read_csv(results_dir / 'frequency_sensitivity_aggregate_indist.csv')
    ood_df = pd.read_csv(results_dir / 'frequency_sensitivity_aggregate_ood.csv')

    # Create output directory for plots
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Main metric to plot
    metric = 'F1@2.0'

    # Generate 4 plots
    print("\nGenerating frequency sensitivity plots...")
    print("=" * 80)

    # 1. Visual in-distribution
    plot_frequency_relationship(
        df=indist_df,
        modality='visual',
        eval_type='indist',
        metric=metric,
        output_path=plots_dir / 'visual_frequency_indist.png'
    )

    # 2. Visual out-of-distribution
    plot_frequency_relationship(
        df=ood_df,
        modality='visual',
        eval_type='ood',
        metric=metric,
        output_path=plots_dir / 'visual_frequency_ood.png'
    )

    # 3. Proprio in-distribution
    plot_frequency_relationship(
        df=indist_df,
        modality='proprio',
        eval_type='indist',
        metric=metric,
        output_path=plots_dir / 'proprio_frequency_indist.png'
    )

    # 4. Proprio out-of-distribution
    plot_frequency_relationship(
        df=ood_df,
        modality='proprio',
        eval_type='ood',
        metric=metric,
        output_path=plots_dir / 'proprio_frequency_ood.png'
    )

    print("=" * 80)
    print(f"All plots saved to: {plots_dir}")
    print("\nGenerated plots:")
    print(f"  1. Visual frequency (in-distribution): {plots_dir / 'visual_frequency_indist.png'}")
    print(f"  2. Visual frequency (out-of-distribution): {plots_dir / 'visual_frequency_ood.png'}")
    print(f"  3. Proprio frequency (in-distribution): {plots_dir / 'proprio_frequency_indist.png'}")
    print(f"  4. Proprio frequency (out-of-distribution): {plots_dir / 'proprio_frequency_ood.png'}")


if __name__ == '__main__':
    main()
