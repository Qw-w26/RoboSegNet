#!/usr/bin/env python3
"""Compute aggregate metrics for ASFormer baseline results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def compute_aggregate_metrics(results_file):
    """Compute aggregate metrics from evaluation results."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    metrics = {
        'F1@1.0': [],
        'F1@2.0': [],
        'F1@3.0': [],
        'Seg-F1@10%': [],
        'Seg-F1@25%': [],
        'Seg-F1@50%': [],
        'Edit-Dist': [],
        'Norm-Edit-Dist': [],
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': []
    }

    for task_id, task_results in results.items():
        metrics['F1@1.0'].append(task_results.get('f1@1.0', 0.0))
        metrics['F1@2.0'].append(task_results.get('f1@2.0', 0.0))
        metrics['F1@3.0'].append(task_results.get('f1@3.0', 0.0))
        metrics['Seg-F1@10%'].append(task_results.get('seg_f1@10', 0.0))
        metrics['Seg-F1@25%'].append(task_results.get('seg_f1@25', 0.0))
        metrics['Seg-F1@50%'].append(task_results.get('seg_f1@50', 0.0))
        metrics['Edit-Dist'].append(task_results.get('edit_distance', 0.0))
        metrics['Norm-Edit-Dist'].append(task_results.get('normalized_edit_distance', 1.0))
        metrics['accuracy'].append(task_results.get('accuracy', 0.0))
        metrics['f1_score'].append(task_results.get('f1_score', 0.0))
        metrics['precision'].append(task_results.get('precision', 0.0))
        metrics['recall'].append(task_results.get('recall', 0.0))

    # Compute means
    aggregated = {
        'ablation': 'ASFormer',
        'F1@1.0': np.mean(metrics['F1@1.0']),
        'F1@2.0': np.mean(metrics['F1@2.0']),
        'F1@3.0': np.mean(metrics['F1@3.0']),
        'Seg-F1@10%': np.mean(metrics['Seg-F1@10%']),
        'Seg-F1@25%': np.mean(metrics['Seg-F1@25%']),
        'Seg-F1@50%': np.mean(metrics['Seg-F1@50%']),
        'AP@1.0': '',  # Not computed
        'AP@2.0': '',  # Not computed
        'AP@3.0': '',  # Not computed
        'Acc': np.mean(metrics['accuracy']),
        'Edit-Dist': np.mean(metrics['Edit-Dist']),
        'Norm-Edit-Dist': np.mean(metrics['Norm-Edit-Dist']),
        'accuracy': np.mean(metrics['accuracy']),
        'f1_score': np.mean(metrics['f1_score']),
        'mAP': '',  # Not computed
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall'])
    }

    return aggregated

def main():
    # Compute for in-distribution (training tasks)
    results_file = Path('multi_task_asformer_results/evaluation_best/evaluation_results.json')

    if results_file.exists():
        print("Computing aggregate metrics for ASFormer (in-distribution)...")
        aggregated = compute_aggregate_metrics(results_file)

        # Create DataFrame
        df = pd.DataFrame([aggregated])

        # Save to CSV
        output_file = Path('multi_task_asformer_results/asformer_aggregate_indist.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        print(f"\nSaved to: {output_file}")
        print("\nAggregate Metrics:")
        print(df.to_string(index=False))

        # Print formatted table
        print("\n" + "="*80)
        print("ASFormer In-Distribution Results")
        print("="*80)
        print(f"F1@1.0:       {aggregated['F1@1.0']:.4f}")
        print(f"F1@2.0:       {aggregated['F1@2.0']:.4f}")
        print(f"F1@3.0:       {aggregated['F1@3.0']:.4f}")
        print(f"Seg-F1@10%:   {aggregated['Seg-F1@10%']:.4f}")
        print(f"Seg-F1@25%:   {aggregated['Seg-F1@25%']:.4f}")
        print(f"Seg-F1@50%:   {aggregated['Seg-F1@50%']:.4f}")
        print(f"Edit-Dist:    {aggregated['Edit-Dist']:.4f}")
        print(f"Norm-Edit:    {aggregated['Norm-Edit-Dist']:.4f}")
        print(f"Accuracy:     {aggregated['accuracy']:.4f}")
        print(f"Precision:    {aggregated['precision']:.4f}")
        print(f"Recall:       {aggregated['recall']:.4f}")
        print("="*80)
    else:
        print(f"Results file not found: {results_file}")

if __name__ == '__main__':
    main()
