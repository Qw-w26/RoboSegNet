#!/usr/bin/env python3
"""
Generate LaTeX tables from multi-task ablation summary CSVs.
By default, creates aggregated tables with task-weighted metrics.

python scripts/generate_multitask_latex_tables.py --output ablation_results/multitask_tables.tex
python scripts/generate_multitask_latex_tables.py --per-task --output ablation_results/multitask_tables_per_task.tex
"""

import pandas as pd
import argparse
from pathlib import Path


# Task weights based on number of trajectories
TASK_WEIGHTS = {
    374: 91,
    440: 165,
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
    380: 815,
    424: 916,
    392: 977,
}


def format_value(value, decimal_places=2, is_best=False, is_edit_dist=False):
    """Format numerical value for LaTeX table."""
    if pd.isna(value) or value == 0:
        return "--"
    # Edit distance is not a percentage, so don't multiply by 100
    if is_edit_dist:
        formatted = f"{value:.{decimal_places}f}"
    else:
        formatted = f"{value * 100:.{decimal_places}f}"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def compute_weighted_aggregates(df, task_weights):
    """
    Compute weighted aggregate metrics across tasks.

    Args:
        df: DataFrame with columns ['task', 'ablation', metric1, metric2, ...]
        task_weights: Dictionary mapping task_id to number of trajectories

    Returns:
        DataFrame with aggregated metrics per ablation method
    """
    methods = df['ablation'].unique()

    # Identify metric columns (everything except 'task' and 'ablation')
    metric_columns = [col for col in df.columns if col not in ['task', 'ablation', 'status']]

    aggregated_data = []

    for method in methods:
        method_data = df[df['ablation'] == method]
        method_row = {'ablation': method}

        for metric in metric_columns:
            # Compute weighted average for this metric
            weighted_sum = 0.0
            total_weight = 0.0

            for _, row in method_data.iterrows():
                task_name = row['task']
                metric_value = row[metric]

                # Extract numeric task ID from task name (e.g., "task_374" -> 374)
                if isinstance(task_name, str):
                    try:
                        # Remove "task_" prefix if present
                        if task_name.startswith('task_'):
                            task_id = int(task_name.replace('task_', ''))
                        else:
                            task_id = int(task_name)
                    except ValueError:
                        continue
                else:
                    task_id = int(task_name)

                # Convert metric value to float and skip if invalid
                try:
                    metric_value = float(metric_value)
                except (ValueError, TypeError):
                    continue

                # Skip if no weight available or metric is NaN
                if task_id not in task_weights or pd.isna(metric_value):
                    continue

                weight = task_weights[task_id]
                weighted_sum += metric_value * weight
                total_weight += weight

            # Compute weighted average
            if total_weight > 0:
                method_row[metric] = weighted_sum / total_weight
            else:
                method_row[metric] = None

        aggregated_data.append(method_row)

    return pd.DataFrame(aggregated_data)


def generate_single_table(df, title, label_suffix, bold_best=True, aggregate=True):
    """
    Generate a single LaTeX table from a dataframe.

    Args:
        df: DataFrame with the data
        title: Table title/caption
        label_suffix: Suffix for the table label
        bold_best: Whether to bold the best values
        aggregate: Whether to generate aggregated table (default) or per-task table

    Returns:
        String containing the LaTeX table
    """
    if aggregate:
        # Compute weighted aggregates across tasks
        df = compute_weighted_aggregates(df, TASK_WEIGHTS)
        tasks = ['Aggregated']
        methods = df['ablation'].unique()
    else:
        # Get unique tasks and methods
        tasks = sorted(df['task'].unique())
        methods = df['ablation'].unique()

    # Metrics to include
    metrics_to_include = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@50%', 'Acc', 'Edit-Dist']
    metrics = [m for m in metrics_to_include if m in df.columns]

    # Find best values for each metric and task
    best_values = {}
    if bold_best:
        if aggregate:
            # For aggregated table, find best across all methods
            for metric in metrics:
                # For Edit-Dist, lower is better; for others, higher is better
                if metric == 'Edit-Dist':
                    best_val = df[metric].min()
                else:
                    best_val = df[metric].max()
                if pd.notna(best_val) and best_val > 0:
                    best_values[('Aggregated', metric)] = best_val
        else:
            # For per-task table, find best per task
            for task in tasks:
                for metric in metrics:
                    task_data = df[df['task'] == task]
                    # For Edit-Dist, lower is better; for others, higher is better
                    if metric == 'Edit-Dist':
                        best_val = task_data[metric].min()
                    else:
                        best_val = task_data[metric].max()
                    if pd.notna(best_val) and best_val > 0:
                        best_values[(task, metric)] = best_val

    # Start building LaTeX table
    latex_lines = []

    # Table header
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(f"    \\caption{{{title}}}")
    latex_lines.append(f"    \\label{{tab:multitask-{label_suffix}}}")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")

    # Column format
    col_format = "l" + "r" * (len(metrics) * len(tasks))

    latex_lines.append(r"    \resizebox{\textwidth}{!}{")
    latex_lines.append(f"    \\begin{{tabular}}{{@{{}}{col_format}@{{}}}}")
    latex_lines.append(r"        \toprule")

    # Generate task group headers
    if len(tasks) > 1:
        header_line = r"        \multicolumn{1}{c}{} "
        for i, task in enumerate(tasks):
            task_name = task.replace('_', ' ').title()
            header_line += f"& \\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{task_name}}}}}"
            if i < len(tasks) - 1:
                header_line += " "
        header_line += r" \\"
        latex_lines.append(header_line)

        # Add cmidrules for task groups
        cmidrule_parts = []
        for i, task in enumerate(tasks):
            start_col = 2 + i * len(metrics)
            end_col = start_col + len(metrics) - 1
            if i == 0:
                cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
            elif i == len(tasks) - 1:
                cmidrule_parts.append(f"\\cmidrule(l){{{start_col}-{end_col}}}")
            else:
                cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        latex_lines.append("        " + "".join(cmidrule_parts))

    # Generate metric headers
    header_parts = [r"\textbf{Method}"]
    for task in tasks:
        for metric in metrics:
            # Clean up metric names
            if metric == 'F1@1.0':
                metric_name = "F1@1s"
            elif metric == 'F1@2.0':
                metric_name = "F1@2s"
            elif metric == 'F1@3.0':
                metric_name = "F1@3s"
            elif metric == 'Seg-F1@50%':
                metric_name = r"Seg-F1@50\%"
            elif metric == 'Acc':
                metric_name = "Acc"
            elif metric == 'Edit-Dist':
                metric_name = "Edit-Dist"
            else:
                metric_name = metric.capitalize()
            header_parts.append(f"{{\\textbf{{{metric_name}}}}}")

    latex_lines.append("        " + " & ".join(header_parts) + r" \\")

    # Add units row
    units_parts = [""]
    for task in tasks:
        for metric in metrics:
            if metric == 'Edit-Dist':
                units_parts.append("")
            else:
                units_parts.append("{(\\%)}")
    latex_lines.append("        " + " & ".join(units_parts) + r" \\")

    latex_lines.append(r"        \midrule")

    # Generate data rows
    for method in methods:
        row_parts = [method]

        if aggregate:
            # For aggregated table, get data directly from the aggregated dataframe
            method_data = df[df['ablation'] == method]

            if len(method_data) == 0:
                # No data for this method
                for metric in metrics:
                    row_parts.append("--")
            else:
                for metric in metrics:
                    value = method_data.iloc[0][metric]
                    # Check if this is the best value
                    is_best = False
                    if bold_best and ('Aggregated', metric) in best_values:
                        is_best = pd.notna(value) and abs(value - best_values[('Aggregated', metric)]) < 1e-6
                    row_parts.append(format_value(value, is_best=is_best, is_edit_dist=(metric == 'Edit-Dist')))
        else:
            # For per-task table, iterate through tasks
            for task in tasks:
                task_data = df[(df['ablation'] == method) & (df['task'] == task)]

                if len(task_data) == 0:
                    for metric in metrics:
                        row_parts.append("--")
                else:
                    for metric in metrics:
                        value = task_data.iloc[0][metric]
                        is_best = False
                        if bold_best and (task, metric) in best_values:
                            is_best = pd.notna(value) and abs(value - best_values[(task, metric)]) < 1e-6
                        row_parts.append(format_value(value, is_best=is_best, is_edit_dist=(metric == 'Edit-Dist')))

        latex_lines.append("        " + " & ".join(row_parts) + r" \\")

    # Close table
    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    }")

    # Add caption notes
    latex_lines.append(r"    \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append(r"    \caption*{")
    latex_lines.append(r"        \textit{Note:} ")
    latex_lines.append(r"        F1@Xs represents F1 score at X second threshold. ")
    latex_lines.append(r"        Seg-F1@50\% represents segmental F1 score at 50\% overlap threshold. ")
    latex_lines.append(r"        Acc represents frame-level accuracy. ")
    latex_lines.append(r"        Edit-Dist represents edit distance (lower is better). ")
    latex_lines.append(r"        All values except Edit-Dist are reported as percentages. ")
    if bold_best:
        latex_lines.append(r"        Best results per metric are shown in \textbf{bold}. ")
    latex_lines.append(r"        '--' indicates no data or zero value.")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


def generate_average_table(df_indist, df_ood, bold_best=True):
    """
    Generate a weighted average table across all tasks.

    Args:
        df_indist: In-distribution DataFrame
        df_ood: Out-of-distribution DataFrame
        bold_best: Whether to bold the best values

    Returns:
        String containing the LaTeX table
    """
    # Metrics to include
    metrics_to_include = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@50%', 'Acc', 'Edit-Dist']

    # Get unique methods from both dataframes
    methods_indist = df_indist['ablation'].unique()
    methods_ood = df_ood['ablation'].unique()
    methods = list(set(methods_indist) | set(methods_ood))

    # Calculate weighted averages for each method and evaluation type
    results = []
    for method in methods:
        # In-distribution weighted averages
        indist_data = df_indist[df_indist['ablation'] == method]
        if len(indist_data) > 0:
            indist_agg = compute_weighted_aggregates(indist_data, TASK_WEIGHTS)
            if len(indist_agg) > 0:
                indist_avg = {'ablation': method, 'eval_type': 'In-Dist'}
                for metric in metrics_to_include:
                    if metric in indist_agg.columns:
                        indist_avg[metric] = indist_agg.iloc[0][metric]
                results.append(indist_avg)

        # Out-of-distribution weighted averages
        ood_data = df_ood[df_ood['ablation'] == method]
        if len(ood_data) > 0:
            ood_agg = compute_weighted_aggregates(ood_data, TASK_WEIGHTS)
            if len(ood_agg) > 0:
                ood_avg = {'ablation': method, 'eval_type': 'OOD'}
                for metric in metrics_to_include:
                    if metric in ood_agg.columns:
                        ood_avg[metric] = ood_agg.iloc[0][metric]
                results.append(ood_avg)

    df_avg = pd.DataFrame(results)

    # Filter to metrics that exist
    metrics = [m for m in metrics_to_include if m in df_avg.columns]

    # Find best values for each metric and eval_type
    best_values = {}
    if bold_best:
        for eval_type in ['In-Dist', 'OOD']:
            for metric in metrics:
                eval_data = df_avg[df_avg['eval_type'] == eval_type]
                # For Edit-Dist, lower is better; for others, higher is better
                if metric == 'Edit-Dist':
                    best_val = eval_data[metric].min()
                else:
                    best_val = eval_data[metric].max()
                if pd.notna(best_val) and best_val > 0:
                    best_values[(eval_type, metric)] = best_val

    # Start building LaTeX table
    latex_lines = []

    # Table header
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \caption{Average Performance Across All Tasks}")
    latex_lines.append(r"    \label{tab:multitask-average}")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append("")

    # Column format: method, eval_type, then metrics
    col_format = "l" + "l" + "r" * len(metrics)

    latex_lines.append(r"    \resizebox{\textwidth}{!}{")
    latex_lines.append(f"    \\begin{{tabular}}{{@{{}}{col_format}@{{}}}}")
    latex_lines.append(r"        \toprule")

    # Headers
    header_parts = [r"\textbf{Method}", r"\textbf{Eval Type}"]
    for metric in metrics:
        # Clean up metric names
        if metric == 'F1@1.0':
            metric_name = "F1@1s"
        elif metric == 'F1@2.0':
            metric_name = "F1@2s"
        elif metric == 'F1@3.0':
            metric_name = "F1@3s"
        elif metric == 'Seg-F1@50%':
            metric_name = r"Seg-F1@50\%"
        elif metric == 'Acc':
            metric_name = "Acc"
        elif metric == 'Edit-Dist':
            metric_name = "Edit-Dist"
        else:
            metric_name = metric.capitalize()
        header_parts.append(f"{{\\textbf{{{metric_name}}}}}")

    latex_lines.append("        " + " & ".join(header_parts) + r" \\")

    # Add units row
    units_parts = ["", ""]
    for metric in metrics:
        if metric == 'Edit-Dist':
            units_parts.append("")
        else:
            units_parts.append("{(\\%)}")
    latex_lines.append("        " + " & ".join(units_parts) + r" \\")

    latex_lines.append(r"        \midrule")

    # Generate data rows
    for method in methods:
        method_data = df_avg[df_avg['ablation'] == method]

        for idx, row in method_data.iterrows():
            eval_type = row['eval_type']
            row_parts = [method if eval_type == 'In-Dist' else '', eval_type]

            for metric in metrics:
                value = row[metric]
                is_best = False
                if bold_best and (eval_type, metric) in best_values:
                    is_best = pd.notna(value) and abs(value - best_values[(eval_type, metric)]) < 1e-6
                row_parts.append(format_value(value, is_best=is_best, is_edit_dist=(metric == 'Edit-Dist')))

            latex_lines.append("        " + " & ".join(row_parts) + r" \\")

        # Add separator between methods (except after last one)
        if method != methods[-1]:
            latex_lines.append(r"        \cmidrule(lr){1-" + str(2 + len(metrics)) + "}")

    # Close table
    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    }")

    # Add caption notes
    latex_lines.append(r"    \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append(r"    \caption*{")
    latex_lines.append(r"        \textit{Note:} ")
    latex_lines.append(r"        All values represent weighted averages across all tasks, ")
    latex_lines.append(r"        where each task is weighted by its number of trajectories. ")
    latex_lines.append(r"        F1@Xs represents F1 score at X second threshold. ")
    latex_lines.append(r"        Seg-F1@50\% represents segmental F1 score at 50\% overlap threshold. ")
    latex_lines.append(r"        Acc represents frame-level accuracy. ")
    latex_lines.append(r"        Edit-Dist represents edit distance (lower is better). ")
    latex_lines.append(r"        All values except Edit-Dist are reported as percentages. ")
    if bold_best:
        latex_lines.append(r"        Best results per metric are shown in \textbf{bold}. ")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


def generate_multitask_tables(indist_path, ood_path, output_path=None, bold_best=True, aggregate=True):
    """
    Generate all three LaTeX tables.

    Args:
        indist_path: Path to in-distribution CSV
        ood_path: Path to out-of-distribution CSV
        output_path: Optional path to save output
        bold_best: Whether to bold best values
        aggregate: Whether to generate aggregated tables (default) or per-task tables
    """
    # Read the CSVs
    df_indist = pd.read_csv(indist_path)
    df_ood = pd.read_csv(ood_path)

    # Generate individual tables
    table_indist = generate_single_table(
        df_indist,
        "Multi-Task Ablation: In-Distribution Performance" + (" (Weighted Aggregated)" if aggregate else ""),
        "indist",
        bold_best,
        aggregate
    )

    table_ood = generate_single_table(
        df_ood,
        "Multi-Task Ablation: Out-of-Distribution Performance" + (" (Weighted Aggregated)" if aggregate else ""),
        "ood",
        bold_best,
        aggregate
    )

    # Generate average table (always uses weighted aggregates)
    table_avg = generate_average_table(df_indist, df_ood, bold_best)

    # Combine all tables
    all_tables = "\n\n".join([table_indist, table_ood, table_avg])

    # Print to console
    print(all_tables)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(all_tables)
        print(f"\n\nLaTeX tables saved to: {output_path}")

    return all_tables


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from multi-task ablation summary CSVs. By default generates aggregated tables."
    )
    parser.add_argument(
        '--indist',
        type=str,
        default='/home/syx/ucsf/task_seg/multi_task_ablation_results/multi_task_ablation_summary_indist.csv',
        help='Path to in-distribution CSV file'
    )
    parser.add_argument(
        '--ood',
        type=str,
        default='/home/syx/ucsf/task_seg/multi_task_ablation_results/multi_task_ablation_summary_ood.csv',
        help='Path to out-of-distribution CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output .tex file (optional)'
    )
    parser.add_argument(
        '--bold-best',
        action='store_true',
        default=True,
        help='Bold the best values in each column (default: True)'
    )
    parser.add_argument(
        '--no-bold-best',
        dest='bold_best',
        action='store_false',
        help='Do not bold best values'
    )
    parser.add_argument(
        '--per-task',
        action='store_true',
        help='Generate per-task tables instead of aggregated tables'
    )

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.indist).exists():
        print(f"Error: In-distribution file not found: {args.indist}")
        return

    if not Path(args.ood).exists():
        print(f"Error: Out-of-distribution file not found: {args.ood}")
        return

    # Generate tables (aggregate by default unless --per-task is specified)
    aggregate = not args.per_task
    generate_multitask_tables(args.indist, args.ood, args.output, bold_best=args.bold_best, aggregate=aggregate)


if __name__ == "__main__":
    main()
