#!/usr/bin/env python3
"""Compare TSC methods with ablation study results.

This script loads results from both TSC evaluation and ablation study,
then generates comparison tables and plots.

Usage:
    # Compare TSC results with ablation results
    python scripts/compare_tsc_with_ablation.py \
        --tsc-results ./tsc_evaluation_results/tsc_evaluation_summary.csv \
        --ablation-results ./ablation_results/ablation_summary.csv \
        --output-dir ./comparison_results

Author: Claude Code
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ResultsComparator:
    """Compare TSC and ablation study results."""

    def __init__(
        self,
        tsc_results_path: str,
        ablation_results_path: str,
        output_dir: str
    ):
        """Initialize comparator.

        Args:
            tsc_results_path: Path to TSC results CSV
            ablation_results_path: Path to ablation results CSV
            output_dir: Output directory for comparison
        """
        self.tsc_results_path = Path(tsc_results_path)
        self.ablation_results_path = Path(ablation_results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.tsc_df = None
        self.ablation_df = None

    def load_results(self):
        """Load TSC and ablation results."""
        print("Loading results...")

        # Load TSC results
        if self.tsc_results_path.exists():
            self.tsc_df = pd.read_csv(self.tsc_results_path)
            self.tsc_df['source'] = 'TSC'
            print(f"  Loaded TSC results: {len(self.tsc_df)} rows")
        else:
            print(f"  WARNING: TSC results not found at {self.tsc_results_path}")
            self.tsc_df = pd.DataFrame()

        # Load ablation results
        if self.ablation_results_path.exists():
            self.ablation_df = pd.read_csv(self.ablation_results_path)
            self.ablation_df['source'] = 'Ablation'
            # Rename 'ablation' column to 'method' for consistency
            if 'ablation' in self.ablation_df.columns:
                self.ablation_df = self.ablation_df.rename(columns={'ablation': 'method'})
            print(f"  Loaded ablation results: {len(self.ablation_df)} rows")
        else:
            print(f"  WARNING: Ablation results not found at {self.ablation_results_path}")
            self.ablation_df = pd.DataFrame()

    def generate_comparison_table(self):
        """Generate comparison table combining both result sets."""
        print("\nGenerating comparison table...")

        if self.tsc_df.empty and self.ablation_df.empty:
            print("  No results to compare!")
            return

        # Combine dataframes
        combined_df = pd.concat([self.tsc_df, self.ablation_df], ignore_index=True)

        # Key metrics to compare
        key_metrics = [
            'F1@1.0', 'F1@2.0', 'F1@3.0',
            'Seg-F1@10%', 'Seg-F1@25%', 'Seg-F1@50%',
            'Edit-Dist', 'Framewise-Acc'
        ]

        # Filter to available metrics
        available_metrics = [m for m in key_metrics if m in combined_df.columns]

        # Create comparison table per task
        for task_id in combined_df['task_id'].unique():
            task_df = combined_df[combined_df['task_id'] == task_id]

            print(f"\n{'='*80}")
            print(f"Task {task_id} Comparison")
            print(f"{'='*80}")

            # Create pivot table
            comparison_cols = ['method', 'source'] + available_metrics
            display_df = task_df[comparison_cols].copy()

            # Sort by F1@2.0 (or first available F1 metric)
            f1_cols = [c for c in available_metrics if 'F1@' in c]
            if f1_cols:
                display_df = display_df.sort_values(by=f1_cols[0], ascending=False)

            print(display_df.to_string(index=False))

            # Save per-task comparison
            task_output_file = self.output_dir / f'task_{task_id}_comparison.csv'
            display_df.to_csv(task_output_file, index=False)
            print(f"\nSaved to: {task_output_file}")

        # Create overall comparison (averaged across tasks)
        print(f"\n{'='*80}")
        print(f"Overall Comparison (Average Across Tasks)")
        print(f"{'='*80}")

        overall_df = combined_df.groupby(['method', 'source'])[available_metrics].mean().reset_index()

        # Sort by F1@2.0
        f1_cols = [c for c in available_metrics if 'F1@' in c]
        if f1_cols:
            overall_df = overall_df.sort_values(by=f1_cols[0], ascending=False)

        print(overall_df.to_string(index=False))

        # Save overall comparison
        overall_output_file = self.output_dir / 'overall_comparison.csv'
        overall_df.to_csv(overall_output_file, index=False)
        print(f"\nSaved to: {overall_output_file}")

    def generate_ranking_table(self):
        """Generate ranking table showing best methods."""
        print(f"\n{'='*80}")
        print(f"Method Rankings by Metric")
        print(f"{'='*80}")

        if self.tsc_df.empty and self.ablation_df.empty:
            print("  No results to rank!")
            return

        # Combine dataframes
        combined_df = pd.concat([self.tsc_df, self.ablation_df], ignore_index=True)

        # Average across tasks
        key_metrics = [
            'F1@2.0', 'Seg-F1@50%', 'Edit-Dist', 'Framewise-Acc'
        ]
        available_metrics = [m for m in key_metrics if m in combined_df.columns]

        overall_df = combined_df.groupby(['method', 'source'])[available_metrics].mean().reset_index()

        # Rank by each metric
        rankings = []
        for metric in available_metrics:
            # Lower is better for Edit-Dist, higher is better for others
            ascending = (metric == 'Edit-Dist')

            ranked = overall_df[['method', 'source', metric]].sort_values(
                by=metric, ascending=ascending
            ).reset_index(drop=True)

            ranked['rank'] = range(1, len(ranked) + 1)

            print(f"\nTop 5 by {metric}:")
            print(ranked.head(5).to_string(index=False))

            rankings.append({
                'metric': metric,
                'top_method': ranked.iloc[0]['method'],
                'top_source': ranked.iloc[0]['source'],
                'top_value': ranked.iloc[0][metric]
            })

        # Save rankings
        rankings_df = pd.DataFrame(rankings)
        rankings_file = self.output_dir / 'rankings.csv'
        rankings_df.to_csv(rankings_file, index=False)
        print(f"\nRankings saved to: {rankings_file}")

    def generate_report(self):
        """Generate comprehensive comparison report."""
        print(f"\n{'='*80}")
        print(f"COMPARISON REPORT")
        print(f"{'='*80}\n")

        report_lines = []
        report_lines.append("# TSC vs Ablation Study Comparison Report\n")
        report_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")

        # Summary statistics
        report_lines.append("## Summary\n")
        if not self.tsc_df.empty:
            report_lines.append(f"- TSC Methods Evaluated: {self.tsc_df['method'].nunique()}")
            report_lines.append(f"- TSC Tasks: {sorted(self.tsc_df['task_id'].unique())}")
        if not self.ablation_df.empty:
            report_lines.append(f"- Ablation Methods Evaluated: {self.ablation_df['method'].nunique()}")
            report_lines.append(f"- Ablation Tasks: {sorted(self.ablation_df['task_id'].unique())}")

        report_lines.append("\n## Key Findings\n")

        if not self.tsc_df.empty and not self.ablation_df.empty:
            combined_df = pd.concat([self.tsc_df, self.ablation_df], ignore_index=True)

            # Find best overall method
            if 'F1@2.0' in combined_df.columns:
                overall = combined_df.groupby(['method', 'source'])['F1@2.0'].mean().reset_index()
                best = overall.loc[overall['F1@2.0'].idxmax()]
                report_lines.append(f"- Best Method by F1@2.0: {best['method']} ({best['source']}) = {best['F1@2.0']:.4f}")

            if 'Seg-F1@50%' in combined_df.columns:
                overall = combined_df.groupby(['method', 'source'])['Seg-F1@50%'].mean().reset_index()
                best = overall.loc[overall['Seg-F1@50%'].idxmax()]
                report_lines.append(f"- Best Method by Seg-F1@50%: {best['method']} ({best['source']}) = {best['Seg-F1@50%']:.4f}")

        # Save report
        report_file = self.output_dir / 'comparison_report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        print('\n'.join(report_lines))
        print(f"\nReport saved to: {report_file}")

    def run_comparison(self):
        """Run full comparison pipeline."""
        self.load_results()
        self.generate_comparison_table()
        self.generate_ranking_table()
        self.generate_report()

        print(f"\n{'='*80}")
        print(f"Comparison complete!")
        print(f"Results directory: {self.output_dir}")
        print(f"{'='*80}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare TSC methods with ablation study results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--tsc-results',
        type=str,
        default='./tsc_evaluation_results/tsc_evaluation_summary.csv',
        help='Path to TSC results CSV'
    )

    parser.add_argument(
        '--ablation-results',
        type=str,
        default='./ablation_results/ablation_summary.csv',
        help='Path to ablation results CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./comparison_results',
        help='Output directory for comparison'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    comparator = ResultsComparator(
        tsc_results_path=args.tsc_results,
        ablation_results_path=args.ablation_results,
        output_dir=args.output_dir
    )

    comparator.run_comparison()


if __name__ == '__main__':
    main()
