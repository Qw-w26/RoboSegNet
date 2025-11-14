#!/usr/bin/env python3
"""Generate ASFormer summary CSV for LaTeX table generation."""

import json
import pandas as pd
from pathlib import Path

def generate_summary_csv(results_file, output_file, method_name='ASFormer'):
    """Generate summary CSV from evaluation results."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    rows = []
    for task_key, task_results in results.items():
        # Extract task ID
        task_id = int(task_key.replace('task_', ''))

        row = {
            'method': method_name,
            'task': task_id,
            'F1@1.0': task_results.get('f1@1.0', 0.0),
            'F1@2.0': task_results.get('f1@2.0', 0.0),
            'F1@3.0': task_results.get('f1@3.0', 0.0),
            'Seg-F1@10%': task_results.get('seg_f1@10', 0.0),
            'Seg-F1@25%': task_results.get('seg_f1@25', 0.0),
            'Seg-F1@50%': task_results.get('seg_f1@50', 0.0),
            'Acc': task_results.get('accuracy', 0.0),
            'Edit-Dist': task_results.get('edit_distance', 0.0),
            'Norm-Edit-Dist': task_results.get('normalized_edit_distance', 1.0),
            'precision': task_results.get('precision', 0.0),
            'recall': task_results.get('recall', 0.0),
            'f1_score': task_results.get('f1_score', 0.0)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('task')

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Tasks: {len(df)}")

    return df

def main():
    # Generate summary for in-distribution results
    results_file = Path('multi_task_asformer_results/evaluation_best/evaluation_results.json')
    output_file = Path('multi_task_asformer_results/asformer_summary.csv')

    if results_file.exists():
        print("Generating ASFormer summary CSV...")
        df = generate_summary_csv(results_file, output_file, method_name='ASFormer')
        print("\nFirst few rows:")
        print(df.head().to_string(index=False))
    else:
        print(f"Error: Results file not found: {results_file}")

if __name__ == '__main__':
    main()
