#!/usr/bin/env python3
"""
Unified script to generate all LaTeX tables for the paper.

This script generates:
1. Single Task Performance Comparison (with baselines)
2. Single Task Ablation Studies
3. Multi-Task In-Distribution Performance
4. Multi-Task Out-of-Distribution Performance

Usage:
    python scripts/generate_all_latex_tables.py --output results/all_tables.tex
    python scripts/generate_all_latex_tables.py --single-task-only
    python scripts/generate_all_latex_tables.py --multi-task-only
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


# Task weights based on number of trajectories
TASK_WEIGHTS = {
    374: 91, 440: 165, 398: 173, 327: 209, 385: 214, 389: 312,
    376: 403, 375: 422, 357: 428, 358: 470, 356: 475, 354: 516,
    360: 517, 372: 537, 359: 644, 378: 739, 390: 750, 377: 794,
    384: 808, 380: 815, 424: 916, 392: 977,
}


def format_value(value, decimal_places=2, is_best=False):
    """Format numerical value for LaTeX table."""
    if pd.isna(value):
        return "--"
    # For very small values close to zero, show as --
    if abs(value) < 1e-6:
        return "--"

    # All metrics are percentages
    formatted = f"{value * 100:.{decimal_places}f}"

    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def compute_weighted_average(df: pd.DataFrame, metric: str, method_col: str = 'method',
                             task_col: str = 'task', task_weights: Dict = None) -> Dict:
    """
    Compute task-weighted average for each method across tasks.

    Args:
        df: DataFrame with results
        metric: Name of the metric column
        method_col: Name of the method/ablation column
        task_col: Name of the task column
        task_weights: Dictionary mapping task IDs to weights (number of trajectories)

    Returns:
        Dictionary mapping method name to weighted average
    """
    if task_weights is None:
        task_weights = TASK_WEIGHTS

    methods = df[method_col].unique()
    results = {}

    for method in methods:
        method_data = df[df[method_col] == method]
        weighted_sum = 0.0
        total_weight = 0.0

        for _, row in method_data.iterrows():
            task_name = row[task_col]
            metric_value = row[metric]

            # Extract numeric task ID
            if isinstance(task_name, str):
                try:
                    task_id = int(task_name.replace('task_', ''))
                except (ValueError, AttributeError):
                    continue
            else:
                task_id = int(task_name)

            # Skip if invalid
            if task_id not in task_weights or pd.isna(metric_value):
                continue

            weight = task_weights[task_id]
            weighted_sum += float(metric_value) * weight
            total_weight += weight

        if total_weight > 0:
            results[method] = weighted_sum / total_weight
        else:
            results[method] = None

    return results


def generate_single_task_comparison_table(ablation_path: str, asformer_path: str,
                                          ms_tcn_path: str, tsc_path: str,
                                          threshold_sensitivity_path: str = None) -> str:
    """
    Generate Table 1: Single Task Performance Comparison with baselines.

    Rows: TSC, GMM-HMM, MS-TCN, ASFormer, Our Method
    Columns: F1@1s, F1@2s, F1@3s, Seg-F1@50%, Acc, Edit-Dist

    Args:
        threshold_sensitivity_path: Optional path to threshold sensitivity results CSV
                                   If provided, will show results for both threshold 0.8 and 0.9
    """
    # Read all data
    df_ablation = pd.read_csv(ablation_path)
    df_asformer = pd.read_csv(asformer_path)
    df_ms_tcn = pd.read_csv(ms_tcn_path)
    df_tsc = pd.read_csv(tsc_path)

    # Read threshold sensitivity data if provided
    if threshold_sensitivity_path:
        df_threshold = pd.read_csv(threshold_sensitivity_path)
    else:
        df_threshold = None

    # Rename columns to standardize
    df_tsc = df_tsc.rename(columns={'task_id': 'task', 'Framewise-Acc': 'Acc'})

    # Define metrics
    metrics = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@10%', 'Seg-F1@25%', 'Seg-F1@50%']

    # Compute weighted averages for each method
    results = {}

    # TSC: Use "Change Point Detection" or aggregate across all TSC methods
    tsc_methods = df_tsc['method'].unique()
    if 'Time-Varying GMM' in tsc_methods:
        # Use Time-Varying GMM as GMM-HMM baseline
        df_gmm_hmm = df_tsc[df_tsc['method'] == 'Time-Varying GMM']
        results['GMM-HMM'] = {m: compute_weighted_average(df_gmm_hmm, m, 'method', 'task')['Time-Varying GMM']
                              for m in metrics if m in df_gmm_hmm.columns}

    # For TSC, we'll use Change Point Detection as the TSC baseline
    if 'Change Point Detection' in tsc_methods:
        df_tsc_method = df_tsc[df_tsc['method'] == 'Change Point Detection']
        results['TSC'] = {m: compute_weighted_average(df_tsc_method, m, 'method', 'task')['Change Point Detection']
                         for m in metrics if m in df_tsc_method.columns}

    # MS-TCN
    results['MS-TCN'] = {m: compute_weighted_average(df_ms_tcn, m, 'method', 'task').get('MS-TCN (2 stages, 5 layers)')
                        for m in metrics if m in df_ms_tcn.columns}

    # ASFormer
    results['ASFormer'] = {m: compute_weighted_average(df_asformer, m, 'method', 'task').get('ASFormer')
                          for m in metrics if m in df_asformer.columns}

    # Our method (Single Task Baseline from ablation results or threshold sensitivity)
    if df_threshold is not None:
        # Use threshold sensitivity results - include both 0.8 and 0.9
        # Map column names from threshold sensitivity CSV to expected metric names
        threshold_metric_map = {
            'F1@1.0': 'f1@1.0',
            'F1@2.0': 'f1@2.0',
            'F1@3.0': 'f1@3.0',
            'Seg-F1@10%': 'seg_f1@10',
            'Seg-F1@25%': 'seg_f1@25',
            'Seg-F1@50%': 'seg_f1@50'
        }

        for threshold_val in [0.8, 0.9]:
            df_threshold_subset = df_threshold[
                (df_threshold['ablation'] == 'Single Task Baseline') &
                (df_threshold['threshold'] == threshold_val)
            ]

            if len(df_threshold_subset) > 0:
                method_name = f'\\algname (threshold={threshold_val})'
                results[method_name] = {}

                for display_metric, csv_metric in threshold_metric_map.items():
                    if csv_metric in df_threshold_subset.columns:
                        weighted_sum = 0.0
                        total_weight = 0.0

                        for _, row in df_threshold_subset.iterrows():
                            task_id = int(row['task_id'])
                            if task_id in TASK_WEIGHTS and not pd.isna(row[csv_metric]):
                                weighted_sum += float(row[csv_metric]) * TASK_WEIGHTS[task_id]
                                total_weight += TASK_WEIGHTS[task_id]

                        if total_weight > 0:
                            results[method_name][display_metric] = weighted_sum / total_weight
                        else:
                            results[method_name][display_metric] = None
                    else:
                        results[method_name][display_metric] = None
    else:
        # Use default ablation results
        df_ours = df_ablation[df_ablation['ablation'] == 'Single Task Baseline']
        results['\\algname'] = {m: compute_weighted_average(df_ours, m, 'ablation', 'task').get('Single Task Baseline')
                               for m in metrics if m in df_ours.columns}

    # Find best values (higher is better for all metrics)
    best_values = {}
    for metric in metrics:
        values = [results[method].get(metric) for method in results.keys() if results[method].get(metric) is not None]
        if values:
            best_values[metric] = max(values)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append("")
    latex_lines.append(r"  \caption{\textbf{Single Task Performance of \algname}. We compare \algname with alternative methods. For proprioceptive data, we include Gaussian Mixture Model-Hidden Markov Model (GMM-HMM) and Transition State Clustering (TSC); both of the methods are unsupervised. For visual data, we include Multi-Stage Temporal Convolutional Network (MS-TCN) ~\cite{farha2019ms} and ASFormer~\cite{chinayi_ASformer}. }")
    latex_lines.append(r"  \label{tab:ablation-results}")
    latex_lines.append(r"  \footnotesize")
    latex_lines.append(r"  \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")
    latex_lines.append(r"  \resizebox{\linewidth}{!}{")
    latex_lines.append(r"  \begin{tabular}{@{}lrrrrrr@{}}")
    latex_lines.append(r"      \toprule")
    latex_lines.append(r"      \textbf{Method} & {\textbf{F1@1s}} & {\textbf{F1@2s}} & {\textbf{F1@3s}} & {\textbf{Seg-F1@10\%}} & {\textbf{Seg-F1@25\%}} & {\textbf{Seg-F1@50\%}} \\")
    latex_lines.append(r"       & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} \\")
    latex_lines.append(r"      \midrule")

    # Add rows for each method
    if df_threshold is not None:
        method_order = ['TSC', 'GMM-HMM', 'MS-TCN', 'ASFormer', '\\algname (threshold=0.8)', '\\algname (threshold=0.9)']
    else:
        method_order = ['TSC', 'GMM-HMM', 'MS-TCN', 'ASFormer', '\\algname']

    for method in method_order:
        if method not in results:
            continue

        # Format method name for display
        if 'threshold=' in method:
            threshold_val = method.split('=')[1].rstrip(')')
            display_name = f'\\algname ($\\tau$={threshold_val})'
        else:
            display_name = method.replace('\\algname', r'\algname')

        row_parts = [display_name]
        for metric in metrics:
            value = results[method].get(metric)
            is_best = False
            if value is not None and metric in best_values:
                if metric == 'Edit-Dist':
                    is_best = abs(value - best_values[metric]) < 1e-6
                else:
                    is_best = abs(value - best_values[metric]) < 1e-6
            row_parts.append(format_value(value, is_best=is_best))

        latex_lines.append("      " + " & ".join(row_parts) + r" \\")

        # Add midrule before our method
        if method == 'ASFormer':
            latex_lines.append(r"      \midrule")

    latex_lines.append(r"      \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"  }")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def generate_single_task_ablation_table(ablation_path: str, threshold_sensitivity_path: str = None) -> str:
    """
    Generate Table 2: Single Task Ablation Studies.

    Rows: No Visual, No Kinematics, MLP + MSE, Our Method
    Columns: F1@1s, F1@2s, F1@3s, Seg-F1@50%, Acc, Edit-Dist

    Args:
        threshold_sensitivity_path: Optional path to threshold sensitivity results CSV
                                   If provided, will show results for both threshold 0.8 and 0.9
    """
    df = pd.read_csv(ablation_path)

    # Read threshold sensitivity data if provided
    if threshold_sensitivity_path:
        df_threshold = pd.read_csv(threshold_sensitivity_path)
    else:
        df_threshold = None

    # Define metrics
    metrics = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@10%', 'Seg-F1@25%', 'Seg-F1@50%']

    # Define ablation methods to include
    ablation_methods = {
        'Pure Proprioceptive': '\\algname (No Visual)',
        'Pure Visual': '\\algname (No Kinematics)',
        'MLP + MSE': 'MLP + MSE',
    }

    # Compute weighted averages
    results = {}
    for method_key, method_display in ablation_methods.items():
        df_method = df[df['ablation'] == method_key]
        if len(df_method) > 0:
            results[method_display] = {m: compute_weighted_average(df_method, m, 'ablation', 'task').get(method_key)
                                      for m in metrics if m in df_method.columns}

    # Add single task baseline - either from threshold sensitivity or default
    if df_threshold is not None:
        # Map column names from threshold sensitivity CSV to expected metric names
        threshold_metric_map = {
            'F1@1.0': 'f1@1.0',
            'F1@2.0': 'f1@2.0',
            'F1@3.0': 'f1@3.0',
            'Seg-F1@10%': 'seg_f1@10',
            'Seg-F1@25%': 'seg_f1@25',
            'Seg-F1@50%': 'seg_f1@50'
        }

        for threshold_val in [0.8, 0.9]:
            df_threshold_subset = df_threshold[
                (df_threshold['ablation'] == 'Single Task Baseline') &
                (df_threshold['threshold'] == threshold_val)
            ]

            if len(df_threshold_subset) > 0:
                method_name = f'\\algname (threshold={threshold_val})'
                results[method_name] = {}

                for display_metric, csv_metric in threshold_metric_map.items():
                    if csv_metric in df_threshold_subset.columns:
                        weighted_sum = 0.0
                        total_weight = 0.0

                        for _, row in df_threshold_subset.iterrows():
                            task_id = int(row['task_id'])
                            if task_id in TASK_WEIGHTS and not pd.isna(row[csv_metric]):
                                weighted_sum += float(row[csv_metric]) * TASK_WEIGHTS[task_id]
                                total_weight += TASK_WEIGHTS[task_id]

                        if total_weight > 0:
                            results[method_name][display_metric] = weighted_sum / total_weight
                        else:
                            results[method_name][display_metric] = None
                    else:
                        results[method_name][display_metric] = None
    else:
        # Use default ablation results
        df_ours = df[df['ablation'] == 'Single Task Baseline']
        if len(df_ours) > 0:
            results['\\algname'] = {m: compute_weighted_average(df_ours, m, 'ablation', 'task').get('Single Task Baseline')
                                   for m in metrics if m in df_ours.columns}

    # Find best values (higher is better for all metrics)
    best_values = {}
    for metric in metrics:
        values = [results[method].get(metric) for method in results.keys() if results[method].get(metric) is not None]
        if values:
            best_values[metric] = max(values)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append(r"    \caption{\textbf{Single-Task \algname Ablation Studies}. Using only visual and kinematics information does not produce good predictions. \algname design can effectively use both visual semantic and kinematic information. }")
    latex_lines.append(r"    \label{tab:single_task_results}")
    latex_lines.append(r"    \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")
    latex_lines.append(r"    \resizebox{\linewidth}{!}{")
    latex_lines.append(r"    \begin{tabular}{@{}lrrrrrr@{}}")
    latex_lines.append(r"        \toprule")
    latex_lines.append(r"        \textbf{Method} & {\textbf{F1@1s}} & {\textbf{F1@2s}} & {\textbf{F1@3s}} & {\textbf{Seg-F1@10\%}} & {\textbf{Seg-F1@25\%}} & {\textbf{Seg-F1@50\%}} \\")
    latex_lines.append(r"         & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} \\")
    latex_lines.append(r"        \midrule")

    # Add rows
    if df_threshold is not None:
        method_order = ['\\algname (No Visual)', '\\algname (No Kinematics)', 'MLP + MSE',
                       '\\algname (threshold=0.8)', '\\algname (threshold=0.9)']
    else:
        method_order = ['\\algname (No Visual)', '\\algname (No Kinematics)', 'MLP + MSE', '\\algname']

    for method in method_order:
        if method not in results:
            continue

        # Format method name for display
        if 'threshold=' in method:
            threshold_val = method.split('=')[1].rstrip(')')
            display_name = f'\\algname ($\\tau$={threshold_val})'
        else:
            display_name = method.replace('\\algname', r'\algname')

        row_parts = [display_name]
        for metric in metrics:
            value = results[method].get(metric)
            is_best = False
            if value is not None and metric in best_values:
                if metric == 'Edit-Dist':
                    is_best = abs(value - best_values[metric]) < 1e-6
                else:
                    is_best = abs(value - best_values[metric]) < 1e-6
            row_parts.append(format_value(value, is_best=is_best))

        latex_lines.append("        " + " & ".join(row_parts) + r" \\")

    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def generate_merged_single_task_table(ablation_path: str, asformer_path: str,
                                      ms_tcn_path: str, tsc_path: str,
                                      threshold_sensitivity_path: str = None) -> str:
    """
    Generate merged single-task table combining comparison and ablation studies.

    This table includes:
    - Baselines: TSC, GMM-HMM, MS-TCN, ASFormer
    - Ablations: No Visual, No Kinematics, MLP + MSE
    - Our method: Single task baseline with threshold 0.8

    Args:
        threshold_sensitivity_path: Path to threshold sensitivity results CSV
                                   If provided, will show results for threshold 0.8 only
    """
    # Read all data
    df_ablation = pd.read_csv(ablation_path)
    df_asformer = pd.read_csv(asformer_path)
    df_ms_tcn = pd.read_csv(ms_tcn_path)
    df_tsc = pd.read_csv(tsc_path)

    # Read threshold sensitivity data if provided
    if threshold_sensitivity_path:
        df_threshold = pd.read_csv(threshold_sensitivity_path)
    else:
        df_threshold = None

    # Rename columns to standardize
    df_tsc = df_tsc.rename(columns={'task_id': 'task', 'Framewise-Acc': 'Acc'})

    # Define metrics
    metrics = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@10%', 'Seg-F1@25%', 'Seg-F1@50%']

    # Compute weighted averages for each method
    results = {}

    # ===== BASELINES =====
    # TSC: Use "Change Point Detection" or aggregate across all TSC methods
    tsc_methods = df_tsc['method'].unique()
    if 'Time-Varying GMM' in tsc_methods:
        # Use Time-Varying GMM as GMM-HMM baseline
        df_gmm_hmm = df_tsc[df_tsc['method'] == 'Time-Varying GMM']
        results['GMM-HMM'] = {m: compute_weighted_average(df_gmm_hmm, m, 'method', 'task')['Time-Varying GMM']
                              for m in metrics if m in df_gmm_hmm.columns}

    # For TSC, we'll use Change Point Detection as the TSC baseline
    if 'Change Point Detection' in tsc_methods:
        df_tsc_method = df_tsc[df_tsc['method'] == 'Change Point Detection']
        results['TSC'] = {m: compute_weighted_average(df_tsc_method, m, 'method', 'task')['Change Point Detection']
                         for m in metrics if m in df_tsc_method.columns}

    # MS-TCN
    results['MS-TCN'] = {m: compute_weighted_average(df_ms_tcn, m, 'method', 'task').get('MS-TCN (2 stages, 5 layers)')
                        for m in metrics if m in df_ms_tcn.columns}

    # ASFormer
    results['ASFormer'] = {m: compute_weighted_average(df_asformer, m, 'method', 'task').get('ASFormer')
                          for m in metrics if m in df_asformer.columns}

    # ===== ABLATIONS =====
    ablation_methods = {
        'Pure Proprioceptive': '\\algname (No Visual)',
        'Pure Visual': '\\algname (No Kinematics)',
        'MLP + MSE': 'MLP + MSE',
    }

    for method_key, method_display in ablation_methods.items():
        df_method = df_ablation[df_ablation['ablation'] == method_key]
        if len(df_method) > 0:
            results[method_display] = {m: compute_weighted_average(df_method, m, 'ablation', 'task').get(method_key)
                                      for m in metrics if m in df_method.columns}

    # ===== OUR METHOD (Single Task Baseline) =====
    if df_threshold is not None:
        # Use threshold 0.8 only
        threshold_metric_map = {
            'F1@1.0': 'f1@1.0',
            'F1@2.0': 'f1@2.0',
            'F1@3.0': 'f1@3.0',
            'Seg-F1@10%': 'seg_f1@10',
            'Seg-F1@25%': 'seg_f1@25',
            'Seg-F1@50%': 'seg_f1@50'
        }

        threshold_val = 0.8
        df_threshold_subset = df_threshold[
            (df_threshold['ablation'] == 'Single Task Baseline') &
            (df_threshold['threshold'] == threshold_val)
        ]

        if len(df_threshold_subset) > 0:
            method_name = '\\algname (threshold=0.8)'
            results[method_name] = {}

            for display_metric, csv_metric in threshold_metric_map.items():
                if csv_metric in df_threshold_subset.columns:
                    weighted_sum = 0.0
                    total_weight = 0.0

                    for _, row in df_threshold_subset.iterrows():
                        task_id = int(row['task_id'])
                        if task_id in TASK_WEIGHTS and not pd.isna(row[csv_metric]):
                            weighted_sum += float(row[csv_metric]) * TASK_WEIGHTS[task_id]
                            total_weight += TASK_WEIGHTS[task_id]

                    if total_weight > 0:
                        results[method_name][display_metric] = weighted_sum / total_weight
                    else:
                        results[method_name][display_metric] = None
                else:
                    results[method_name][display_metric] = None
    else:
        # Use default ablation results
        df_ours = df_ablation[df_ablation['ablation'] == 'Single Task Baseline']
        if len(df_ours) > 0:
            results['\\algname'] = {m: compute_weighted_average(df_ours, m, 'ablation', 'task').get('Single Task Baseline')
                                   for m in metrics if m in df_ours.columns}

    # Find best values (higher is better for all metrics)
    best_values = {}
    for metric in metrics:
        values = [results[method].get(metric) for method in results.keys() if results[method].get(metric) is not None]
        if values:
            best_values[metric] = max(values)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append("")
    latex_lines.append(r"  \caption{\textbf{Single Task Performance and Ablation Studies}. We compare \algname with alternative methods and ablations. For proprioceptive data, we include Gaussian Mixture Model-Hidden Markov Model (GMM-HMM) and Transition State Clustering (TSC); both of the methods are unsupervised. For visual data, we include Multi-Stage Temporal Convolutional Network (MS-TCN) ~\cite{farha2019ms} and ASFormer~\cite{chinayi_ASformer}. Ablation studies show that using only visual or kinematics information does not produce good predictions, while \algname effectively uses both.}")
    latex_lines.append(r"  \label{tab:single-task-merged}")
    latex_lines.append(r"  \footnotesize")
    latex_lines.append(r"  \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")
    latex_lines.append(r"  \resizebox{\linewidth}{!}{")
    latex_lines.append(r"  \begin{tabular}{@{}lrrrrrr@{}}")
    latex_lines.append(r"      \toprule")
    latex_lines.append(r"      \textbf{Method} & {\textbf{F1@1s}} & {\textbf{F1@2s}} & {\textbf{F1@3s}} & {\textbf{Seg-F1@10\%}} & {\textbf{Seg-F1@25\%}} & {\textbf{Seg-F1@50\%}} \\")
    latex_lines.append(r"       & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} \\")
    latex_lines.append(r"      \midrule")

    # Add rows for baselines
    baseline_order = ['TSC', 'GMM-HMM', 'MS-TCN', 'ASFormer']
    for method in baseline_order:
        if method not in results:
            continue

        row_parts = [method]
        for metric in metrics:
            value = results[method].get(metric)
            is_best = False
            if value is not None and metric in best_values:
                is_best = abs(value - best_values[metric]) < 1e-6
            row_parts.append(format_value(value, is_best=is_best))

        latex_lines.append("      " + " & ".join(row_parts) + r" \\")

    # Add midrule before ablations
    latex_lines.append(r"      \midrule")

    # Add rows for ablations and our method
    ablation_order = ['\\algname (No Visual)', '\\algname (No Kinematics)', 'MLP + MSE', '\\algname (threshold=0.8)']
    for method in ablation_order:
        if method not in results:
            continue

        # Format method name for display
        if 'threshold=' in method:
            display_name = r'\algname ($\tau$=0.8)'
        else:
            display_name = method.replace('\\algname', r'\algname')

        row_parts = [display_name]
        for metric in metrics:
            value = results[method].get(metric)
            is_best = False
            if value is not None and metric in best_values:
                is_best = abs(value - best_values[metric]) < 1e-6
            row_parts.append(format_value(value, is_best=is_best))

        latex_lines.append("      " + " & ".join(row_parts) + r" \\")

    latex_lines.append(r"      \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"  }")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def generate_multi_task_table(asformer_path: str, ms_tcn_path: str, ablation_path: str,
                               table_type: str = 'indist', threshold_sensitivity_path: str = None) -> str:
    """
    Generate Multi-Task tables (both in-distribution and OOD).

    Args:
        asformer_path: Path to ASFormer aggregate CSV
        ms_tcn_path: Path to MS-TCN summary CSV
        ablation_path: Path to ablation aggregate CSV
        table_type: 'indist' or 'ood'
        threshold_sensitivity_path: Optional path to threshold sensitivity results CSV
                                   If provided, will show results for threshold 0.8

    Returns:
        LaTeX table string
    """
    # Read aggregate results
    df_asformer = pd.read_csv(asformer_path)
    df_ms_tcn = pd.read_csv(ms_tcn_path)
    df_ablation = pd.read_csv(ablation_path)

    # Read threshold sensitivity data if provided
    if threshold_sensitivity_path:
        df_threshold = pd.read_csv(threshold_sensitivity_path)
    else:
        df_threshold = None

    # Define metrics
    metrics = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@10%', 'Seg-F1@25%', 'Seg-F1@50%']

    # Extract values (these are already aggregated)
    results = {}

    # ASFormer
    asformer_row = df_asformer[df_asformer['ablation'] == 'ASFormer']
    if len(asformer_row) > 0:
        results['ASFormer'] = {m: asformer_row.iloc[0][m] if m in asformer_row.columns else None
                              for m in metrics}

    # MS-TCN (compute weighted average across tasks)
    # MS-TCN has per-task results, not aggregated
    results['MS-TCN'] = {}
    for metric in metrics:
        if metric in df_ms_tcn.columns:
            weighted_sum = 0.0
            total_weight = 0.0
            for _, row in df_ms_tcn.iterrows():
                task_name = row['task']
                if isinstance(task_name, str):
                    try:
                        task_id = int(task_name.replace('task_', ''))
                    except:
                        continue
                else:
                    task_id = int(task_name)

                if task_id in TASK_WEIGHTS and not pd.isna(row[metric]):
                    weighted_sum += float(row[metric]) * TASK_WEIGHTS[task_id]
                    total_weight += TASK_WEIGHTS[task_id]

            if total_weight > 0:
                results['MS-TCN'][metric] = weighted_sum / total_weight
            else:
                results['MS-TCN'][metric] = None

    # Ablation methods
    ablation_mapping = {
        'Pure Visual': '\\algname (No Kinematics)',
        'Pure Proprioceptive': '\\algname (No Visual)',
    }

    for method_key, method_display in ablation_mapping.items():
        method_row = df_ablation[df_ablation['ablation'] == method_key]
        if len(method_row) > 0:
            results[method_display] = {m: method_row.iloc[0][m] if m in method_row.columns else None
                                      for m in metrics}

    # Multi-Task Baseline - use threshold sensitivity if available
    if df_threshold is not None:
        # Use threshold 0.8 from threshold sensitivity results
        # This CSV has aggregated results with columns like avg_f1@1.0, avg_seg_f1@10, etc.
        threshold_metric_map = {
            'F1@1.0': 'avg_f1@1.0',
            'F1@2.0': 'avg_f1@2.0',
            'F1@3.0': 'avg_f1@3.0',
            'Seg-F1@10%': 'avg_seg_f1@10',
            'Seg-F1@25%': 'avg_seg_f1@25',
            'Seg-F1@50%': 'avg_seg_f1@50'
        }

        threshold_val = 0.8
        df_threshold_subset = df_threshold[
            (df_threshold['threshold'] == threshold_val) &
            (df_threshold['eval_type'] == table_type)
        ]

        if len(df_threshold_subset) > 0:
            results['\\algname'] = {}
            row = df_threshold_subset.iloc[0]

            for display_metric, csv_metric in threshold_metric_map.items():
                if csv_metric in df_threshold_subset.columns:
                    results['\\algname'][display_metric] = row[csv_metric]
                else:
                    results['\\algname'][display_metric] = None
        else:
            # Fallback to ablation results
            method_row = df_ablation[df_ablation['ablation'] == 'Multi-Task Baseline']
            if len(method_row) > 0:
                results['\\algname'] = {m: method_row.iloc[0][m] if m in method_row.columns else None
                                       for m in metrics}
    else:
        # Use default ablation results
        method_row = df_ablation[df_ablation['ablation'] == 'Multi-Task Baseline']
        if len(method_row) > 0:
            results['\\algname'] = {m: method_row.iloc[0][m] if m in method_row.columns else None
                                   for m in metrics}

    # Find best values
    best_values = {}
    for metric in metrics:
        values = [results[method].get(metric) for method in results.keys()
                 if results[method].get(metric) is not None and not pd.isna(results[method].get(metric))]
        if values:
            if metric == 'Edit-Dist':
                best_values[metric] = min(values)
            else:
                best_values[metric] = max(values)

    # Generate LaTeX table
    latex_lines = []

    if table_type == 'indist':
        caption = r"\textbf{\algname Trained on 18 Tasks Performance on In-Distribution Tasks}. The task distribution covers a variety of scenarios and skills, such as sorting laundry, brewing tea, packing in supermarket. "
        label = "tab:multitask-indist"
    else:
        caption = r"\textbf{Multi-Task \algname  Trained on 18 Tasks on 3 Out-of-Distribution Tasks}. The out-of-distribution tasks are packing in e-commerce, clear the countertop waste, and brush water bottle."
        label = "tab:multitask-ood"

    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append("")
    latex_lines.append(f"    \\caption{{{caption}}}")
    latex_lines.append(f"    \\label{{{label}}}")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")
    latex_lines.append(r"    \resizebox{\linewidth}{!}{")
    latex_lines.append(r"    \begin{tabular}{@{}lrrrrrr@{}}")
    latex_lines.append(r"        \toprule")
    latex_lines.append(r"        \textbf{Method} & {\textbf{F1@1s}} & {\textbf{F1@2s}} & {\textbf{F1@3s}} & {\textbf{Seg-F1@10\%}} & {\textbf{Seg-F1@25\%}} & {\textbf{Seg-F1@50\%}} \\")
    latex_lines.append(r"         & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} & {(\%)} \\")
    latex_lines.append(r"        \midrule")

    # Add rows in specified order
    method_order = ['ASFormer', 'MS-TCN', '\\algname (No Kinematics)', '\\algname (No Visual)', '\\algname']
    for method in method_order:
        if method not in results:
            continue

        row_parts = [method.replace('\\algname', r'\algname')]
        for metric in metrics:
            value = results[method].get(metric)
            is_best = False
            if value is not None and not pd.isna(value) and metric in best_values:
                if metric == 'Edit-Dist':
                    is_best = abs(value - best_values[metric]) < 1e-6
                else:
                    is_best = abs(value - best_values[metric]) < 1e-6
            row_parts.append(format_value(value, is_best=is_best))

        latex_lines.append("        " + " & ".join(row_parts) + r" \\")

    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all LaTeX tables for the paper"
    )

    # Input paths with defaults
    parser.add_argument(
        '--ablation-results',
        type=str,
        default='ablation_results/ablation_summary.csv',
        help='Path to single-task ablation results CSV'
    )
    parser.add_argument(
        '--asformer-results',
        type=str,
        default='asformer_results/asformer_summary.csv',
        help='Path to ASFormer results CSV'
    )
    parser.add_argument(
        '--ms-tcn-results',
        type=str,
        default='ms_tcn_results/ms_tcn_summary.csv',
        help='Path to MS-TCN results CSV'
    )
    parser.add_argument(
        '--tsc-results',
        type=str,
        default='tsc_baselines_full/tsc_evaluation_summary.csv',
        help='Path to TSC results CSV'
    )
    parser.add_argument(
        '--multi-task-asformer-indist',
        type=str,
        default='multi_task_asformer_results/asformer_aggregate_indist.csv',
        help='Path to multi-task ASFormer in-dist results'
    )
    parser.add_argument(
        '--multi-task-asformer-ood',
        type=str,
        default='multi_task_asformer_results/asformer_aggregate_ood.csv',
        help='Path to multi-task ASFormer OOD results'
    )
    parser.add_argument(
        '--multi-task-ms-tcn-indist',
        type=str,
        default='multi_task_ms_tcn_results/multi_task_ms_tcn_summary_indist.csv',
        help='Path to multi-task MS-TCN in-dist results'
    )
    parser.add_argument(
        '--multi-task-ms-tcn-ood',
        type=str,
        default='multi_task_ms_tcn_results/multi_task_ms_tcn_summary_ood.csv',
        help='Path to multi-task MS-TCN OOD results'
    )
    parser.add_argument(
        '--multi-task-ablation-indist',
        type=str,
        default='multi_task_ablation_results/multi_task_ablation_aggregate_indist.csv',
        help='Path to multi-task ablation in-dist results'
    )
    parser.add_argument(
        '--multi-task-ablation-ood',
        type=str,
        default='multi_task_ablation_results/multi_task_ablation_aggregate_ood.csv',
        help='Path to multi-task ablation OOD results'
    )
    parser.add_argument(
        '--threshold-sensitivity',
        type=str,
        default=None,
        help='Path to ablation threshold sensitivity results CSV for single-task (optional, to show results at threshold 0.8)'
    )
    parser.add_argument(
        '--multi-task-threshold-sensitivity',
        type=str,
        default=None,
        help='Path to threshold sensitivity results CSV for multi-task (optional, to show results at threshold 0.8)'
    )
    parser.add_argument(
        '--merged-single-task',
        action='store_true',
        help='Generate a single merged table for single-task (combines comparison and ablation)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output .tex file (optional, prints to stdout by default)'
    )
    parser.add_argument(
        '--single-task-only',
        action='store_true',
        help='Generate only single-task tables'
    )
    parser.add_argument(
        '--multi-task-only',
        action='store_true',
        help='Generate only multi-task tables'
    )

    args = parser.parse_args()

    # Convert paths to absolute if relative
    base_dir = Path('/home/syx/ucsf/task_seg')

    all_tables = []

    # Generate single-task tables
    if not args.multi_task_only:
        # Prepare threshold sensitivity path
        threshold_path = None
        if args.threshold_sensitivity:
            threshold_path = str(base_dir / args.threshold_sensitivity)

        if args.merged_single_task:
            # Generate merged single-task table
            print("Generating merged single-task table...")
            table_merged = generate_merged_single_task_table(
                str(base_dir / args.ablation_results),
                str(base_dir / args.asformer_results),
                str(base_dir / args.ms_tcn_results),
                str(base_dir / args.tsc_results),
                threshold_sensitivity_path=threshold_path
            )
            all_tables.append(table_merged)
        else:
            # Generate separate tables
            print("Generating single-task comparison table...")
            table1 = generate_single_task_comparison_table(
                str(base_dir / args.ablation_results),
                str(base_dir / args.asformer_results),
                str(base_dir / args.ms_tcn_results),
                str(base_dir / args.tsc_results),
                threshold_sensitivity_path=threshold_path
            )
            all_tables.append(table1)

            print("Generating single-task ablation table...")
            table2 = generate_single_task_ablation_table(
                str(base_dir / args.ablation_results),
                threshold_sensitivity_path=threshold_path
            )
            all_tables.append(table2)

    # Generate multi-task tables
    if not args.single_task_only:
        # Prepare multi-task threshold sensitivity path
        multi_task_threshold_path = None
        if args.multi_task_threshold_sensitivity:
            multi_task_threshold_path = str(base_dir / args.multi_task_threshold_sensitivity)

        print("Generating multi-task in-distribution table...")
        table3 = generate_multi_task_table(
            str(base_dir / args.multi_task_asformer_indist),
            str(base_dir / args.multi_task_ms_tcn_indist),
            str(base_dir / args.multi_task_ablation_indist),
            table_type='indist',
            threshold_sensitivity_path=multi_task_threshold_path
        )
        all_tables.append(table3)

        print("Generating multi-task OOD table...")
        table4 = generate_multi_task_table(
            str(base_dir / args.multi_task_asformer_ood),
            str(base_dir / args.multi_task_ms_tcn_ood),
            str(base_dir / args.multi_task_ablation_ood),
            table_type='ood',
            threshold_sensitivity_path=multi_task_threshold_path
        )
        all_tables.append(table4)

    # Combine all tables
    output = "\n\n".join(all_tables)

    # Print to stdout
    print("\n" + "="*80)
    print("GENERATED LATEX TABLES")
    print("="*80 + "\n")
    print(output)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"\n\nLaTeX tables saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
