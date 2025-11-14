#!/usr/bin/env python3
"""Generate aggregate tables for ASFormer results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def compute_aggregate_metrics(results_file, label):
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
        'ablation': label,
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
    # In-distribution results
    indist_file = Path('multi_task_asformer_results/evaluation_best/evaluation_results.json')
    ood_file = Path('multi_task_asformer_results/evaluation_best_ood/evaluation_results.json')

    results = []

    if indist_file.exists():
        print("Computing in-distribution aggregate metrics...")
        indist_agg = compute_aggregate_metrics(indist_file, 'ASFormer')
        results.append(indist_agg)

        # Save in-dist table
        df_indist = pd.DataFrame([indist_agg])
        output_indist = Path('multi_task_asformer_results/asformer_aggregate_indist.csv')
        df_indist.to_csv(output_indist, index=False)
        print(f"✓ Saved: {output_indist}")

    if ood_file.exists():
        print("Computing OOD aggregate metrics...")
        ood_agg = compute_aggregate_metrics(ood_file, 'ASFormer')
        results.append(ood_agg)

        # Save OOD table
        df_ood = pd.DataFrame([ood_agg])
        output_ood = Path('multi_task_asformer_results/asformer_aggregate_ood.csv')
        df_ood.to_csv(output_ood, index=False)
        print(f"✓ Saved: {output_ood}")

    # Display results
    print("\n" + "="*100)
    print("ASFormer IN-DISTRIBUTION Results")
    print("="*100)
    if indist_file.exists():
        print(f"ablation: ASFormer")
        print(f"F1@1.0:       {indist_agg['F1@1.0']:.4f}")
        print(f"F1@2.0:       {indist_agg['F1@2.0']:.4f}")
        print(f"F1@3.0:       {indist_agg['F1@3.0']:.4f}")
        print(f"Seg-F1@10%:   {indist_agg['Seg-F1@10%']:.4f}")
        print(f"Seg-F1@25%:   {indist_agg['Seg-F1@25%']:.4f}")
        print(f"Seg-F1@50%:   {indist_agg['Seg-F1@50%']:.4f}")
        print(f"Edit-Dist:    {indist_agg['Edit-Dist']:.4f}")
        print(f"Norm-Edit:    {indist_agg['Norm-Edit-Dist']:.4f}")
        print(f"Accuracy:     {indist_agg['accuracy']:.4f}")
        print(f"Precision:    {indist_agg['precision']:.4f}")
        print(f"Recall:       {indist_agg['recall']:.4f}")
    print("="*100)

    print("\n" + "="*100)
    print("ASFormer OUT-OF-DISTRIBUTION Results")
    print("="*100)
    if ood_file.exists():
        print(f"ablation: ASFormer")
        print(f"F1@1.0:       {ood_agg['F1@1.0']:.4f}")
        print(f"F1@2.0:       {ood_agg['F1@2.0']:.4f}")
        print(f"F1@3.0:       {ood_agg['F1@3.0']:.4f}")
        print(f"Seg-F1@10%:   {ood_agg['Seg-F1@10%']:.4f}")
        print(f"Seg-F1@25%:   {ood_agg['Seg-F1@25%']:.4f}")
        print(f"Seg-F1@50%:   {ood_agg['Seg-F1@50%']:.4f}")
        print(f"Edit-Dist:    {ood_agg['Edit-Dist']:.4f}")
        print(f"Norm-Edit:    {ood_agg['Norm-Edit-Dist']:.4f}")
        print(f"Accuracy:     {ood_agg['accuracy']:.4f}")
        print(f"Precision:    {ood_agg['precision']:.4f}")
        print(f"Recall:       {ood_agg['recall']:.4f}")
    print("="*100)

if __name__ == '__main__':
    main()
