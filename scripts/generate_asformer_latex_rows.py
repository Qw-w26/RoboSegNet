#!/usr/bin/env python3
"""Generate LaTeX table rows for ASFormer results."""

import pandas as pd
from pathlib import Path

def format_value(value, is_edit_dist=False):
    """Format value for table display."""
    if pd.isna(value) or value == '':
        return "--"

    try:
        value = float(value)
    except:
        return "--"

    if is_edit_dist:
        return f"{value:.2f}"
    else:
        return f"{value * 100:.2f}"

def generate_latex_row(method_name, csv_path):
    """Generate LaTeX row from aggregate CSV."""
    df = pd.read_csv(csv_path)

    # Get the first (and only) row
    row = df.iloc[0]

    # Extract metrics in the order needed for the table
    f1_1s = format_value(row.get('F1@1.0', ''))
    f1_2s = format_value(row.get('F1@2.0', ''))
    f1_3s = format_value(row.get('F1@3.0', ''))
    seg_f1_50 = format_value(row.get('Seg-F1@50%', ''))
    acc = format_value(row.get('Acc', ''))
    edit_dist = format_value(row.get('Edit-Dist', ''), is_edit_dist=True)

    # Build LaTeX row
    latex_row = f"{method_name} & {f1_1s} & {f1_2s} & {f1_3s} & {seg_f1_50} & {acc} & {edit_dist} \\\\"

    return latex_row

def main():
    print("\n" + "="*80)
    print("ASFormer LaTeX Table Rows")
    print("="*80 + "\n")

    # In-distribution
    indist_file = Path('multi_task_asformer_results/asformer_aggregate_indist.csv')
    if indist_file.exists():
        indist_row = generate_latex_row('ASFormer (In-Dist)', indist_file)
        print("In-Distribution:")
        print(indist_row)
    else:
        print(f"Error: {indist_file} not found")

    print()

    # Out-of-distribution
    ood_file = Path('multi_task_asformer_results/asformer_aggregate_ood.csv')
    if ood_file.exists():
        ood_row = generate_latex_row('ASFormer (OOD)', ood_file)
        print("Out-of-Distribution:")
        print(ood_row)
    else:
        print(f"Error: {ood_file} not found")

    print("\n" + "="*80)
    print("Copy these rows into your LaTeX table:")
    print("="*80)
    if indist_file.exists():
        print(generate_latex_row('ASFormer (In-Dist)', indist_file))
    if ood_file.exists():
        print(generate_latex_row('ASFormer (OOD)', ood_file))
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
