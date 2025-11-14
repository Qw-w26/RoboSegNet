#!/usr/bin/env python3
"""
Calculate a single row for the comparison table by aggregating results across tasks.
"""

import pandas as pd
import sys

# Task weights based on number of trajectories
TASK_WEIGHTS = {
    374: 91,
    398: 173,
    327: 209,
    385: 214,
    389: 312,
    376: 403,
    375: 422,
    357: 428,
    358: 470,
    356: 475,
    354: 516,
    360: 517,
    372: 537,
    359: 644,
    378: 739,
    390: 750,
    377: 794,
    384: 808,
}

def calculate_weighted_average(df, method_name, metric_col, task_col='task', method_col='method'):
    """Calculate weighted average for a specific method and metric."""
    method_data = df[df[method_col] == method_name]

    weighted_sum = 0.0
    total_weight = 0.0

    for _, row in method_data.iterrows():
        task_name = row[task_col]
        metric_value = row[metric_col]

        # Extract numeric task ID
        if isinstance(task_name, str):
            if task_name.startswith('task_'):
                task_id = int(task_name.replace('task_', ''))
            else:
                try:
                    task_id = int(task_name)
                except:
                    continue
        else:
            task_id = int(task_name)

        # Skip if invalid
        if pd.isna(metric_value) or task_id not in TASK_WEIGHTS:
            continue

        weight = TASK_WEIGHTS[task_id]
        weighted_sum += float(metric_value) * weight
        total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight
    return None

def format_value(value, is_edit_dist=False):
    """Format value for table display."""
    if value is None or pd.isna(value):
        return "--"

    if is_edit_dist:
        return f"{value:.2f}"
    else:
        return f"{value * 100:.2f}"

def calculate_table_row(method_name, csv_path, method_col='method', task_col='task'):
    """Calculate a table row for a specific method."""
    df = pd.read_csv(csv_path)

    # Rename columns if needed
    if 'task_id' in df.columns:
        df = df.rename(columns={'task_id': 'task'})
        task_col = 'task'  # Update task_col to use the renamed column
    if 'Framewise-Acc' in df.columns:
        df = df.rename(columns={'Framewise-Acc': 'Acc'})

    # Calculate weighted averages for each metric
    metrics = {
        'F1@1.0': 'F1@1s',
        'F1@2.0': 'F1@2s',
        'F1@3.0': 'F1@3s',
        'Seg-F1@50%': 'Seg-F1@50%',
        'Acc': 'Acc',
        'Edit-Dist': 'Edit-Dist'
    }

    values = {}
    for metric_key, display_name in metrics.items():
        if metric_key in df.columns:
            values[display_name] = calculate_weighted_average(
                df, method_name, metric_key, task_col=task_col, method_col=method_col
            )
        else:
            values[display_name] = None

    return values

def print_latex_row(method_name, values):
    """Print a LaTeX table row."""
    row_parts = [method_name]

    for metric in ['F1@1s', 'F1@2s', 'F1@3s', 'Seg-F1@50%', 'Acc', 'Edit-Dist']:
        value = values.get(metric)
        is_edit_dist = (metric == 'Edit-Dist')
        row_parts.append(format_value(value, is_edit_dist=is_edit_dist))

    latex_row = " & ".join(row_parts) + r" \\"
    print(latex_row)
    return latex_row

def main():
    print("\n=== Calculating Table Rows ===\n")

    # Calculate for different methods

    # 1. GMM-HMM (Time-Varying GMM from TSC baselines)
    print("GMM-HMM:")
    tsc_df = pd.read_csv('tsc_baselines_full/tsc_evaluation_summary.csv')
    gmm_values = calculate_table_row(
        'Time-Varying GMM',
        'tsc_baselines_full/tsc_evaluation_summary.csv',
        method_col='method'
    )
    gmm_row = print_latex_row('GMM-HMM', gmm_values)

    # 2. TSC (Change Point Detection from TSC baselines)
    print("\nTSC:")
    tsc_values = calculate_table_row(
        'Change Point Detection',
        'tsc_baselines_full/tsc_evaluation_summary.csv',
        method_col='method'
    )
    tsc_row = print_latex_row('TSC', tsc_values)

    # 3. MS-TCN
    print("\nMS-TCN:")
    mstcn_values = calculate_table_row(
        'MS-TCN (2 stages, 5 layers)',
        'ms_tcn_results/ms_tcn_summary.csv',
        method_col='method',
        task_col='task'
    )
    mstcn_row = print_latex_row('MS-TCN', mstcn_values)

    # 4. ASFormer
    print("\nASFormer:")
    asformer_values = calculate_table_row(
        'ASFormer',
        'asformer_results/asformer_summary.csv',
        method_col='method',
        task_col='task'
    )
    asformer_row = print_latex_row('ASFormer', asformer_values)

    # 5. Single Task Baseline (our method)
    print("\n\\algname (Single Task Baseline):")
    baseline_values = calculate_table_row(
        'Single Task Baseline',
        'ablation_results/ablation_summary.csv',
        method_col='ablation',
        task_col='task'
    )
    baseline_row = print_latex_row(r'\algname', baseline_values)

    # Print summary
    print("\n=== Complete LaTeX Table ===\n")
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \captionsetup{font=scriptsize, justification=justified}")
    print()
    print(r"    \caption{\textbf{Single Task Performance of \algname}. We compare \algname with alternative methods. For proprioceptive data, we include Gaussian Mixture Model-Hidden Markov Model (GMM-HMM) and Transition State Clustering (TSC). For visual data, we include Multi-Stage Temporal Convolutional Network (MS-TCN). }")
    print(r"    \label{tab:ablation-results}")
    print(r"    \footnotesize")
    print(r"    \setlength{\tabcolsep}{2.5pt}")
    print()
    print(r"    \resizebox{\linewidth}{!}{")
    print(r"    \begin{tabular}{@{}lrrrrrr@{}}")
    print(r"        \toprule")
    print(r"        \textbf{Method} & {\textbf{F1@1s}} & {\textbf{F1@2s}} & {\textbf{F1@3s}} & {\textbf{Seg-F1@50\%}} & {\textbf{Acc}} & {\textbf{Edit-Dist}} \\")
    print(r"         & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} &  \\")
    print(r"        \midrule")
    print("        " + gmm_row)
    print("        " + tsc_row)
    print(r"        \midrule")
    print("        " + mstcn_row)
    print("        " + asformer_row)
    print(r"        \midrule")
    print("        " + baseline_row)
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    }")
    print(r"\end{table}")

if __name__ == '__main__':
    main()
