#!/usr/bin/env python3
"""
Generate LaTeX tables from multi-task ablation summary CSVs.
Creates three tables: in-distribution, out-of-distribution, and average.

python scripts/generate_multitask_latex_tables.py --output ablation_results/multitask_tables.tex
"""

import pandas as pd
import argparse
from pathlib import Path


def format_value(value, decimal_places=2, is_best=False):
    """Format numerical value for LaTeX table."""
    if pd.isna(value) or value == 0:
        return "--"
    formatted = f"{value * 100:.{decimal_places}f}"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def generate_single_table(df, title, label_suffix, bold_best=True):
    """
    Generate a single LaTeX table from a dataframe.

    Args:
        df: DataFrame with the data
        title: Table title/caption
        label_suffix: Suffix for the table label
        bold_best: Whether to bold the best values

    Returns:
        String containing the LaTeX table
    """
    # Get unique tasks and methods
    tasks = sorted(df['task'].unique())
    methods = df['ablation'].unique()

    # Metrics to include
    metrics_to_include = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@50%', 'Acc', 'Edit-Dist']
    metrics = [m for m in metrics_to_include if m in df.columns]

    # Find best values for each metric and task
    best_values = {}
    if bold_best:
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
                    row_parts.append(format_value(value, is_best=is_best))

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
    Generate an average table across all tasks.

    Args:
        df_indist: In-distribution DataFrame
        df_ood: Out-of-distribution DataFrame
        bold_best: Whether to bold the best values

    Returns:
        String containing the LaTeX table
    """
    # Metrics to include
    metrics_to_include = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@50%', 'Acc', 'Edit-Dist']

    # Combine both datasets
    df_combined = pd.concat([df_indist, df_ood], ignore_index=True)

    # Get unique methods
    methods = df_combined['ablation'].unique()

    # Calculate averages for each method and evaluation type
    results = []
    for method in methods:
        # In-distribution averages
        indist_data = df_indist[df_indist['ablation'] == method]
        indist_avg = {'ablation': method, 'eval_type': 'In-Dist'}
        for metric in metrics_to_include:
            if metric in indist_data.columns:
                indist_avg[metric] = indist_data[metric].mean()
        results.append(indist_avg)

        # Out-of-distribution averages
        ood_data = df_ood[df_ood['ablation'] == method]
        ood_avg = {'ablation': method, 'eval_type': 'OOD'}
        for metric in metrics_to_include:
            if metric in ood_data.columns:
                ood_avg[metric] = ood_data[metric].mean()
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
                row_parts.append(format_value(value, is_best=is_best))

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
    latex_lines.append(r"        All values represent averages across all tasks. ")
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


def generate_multitask_tables(indist_path, ood_path, output_path=None, bold_best=True):
    """
    Generate all three LaTeX tables.

    Args:
        indist_path: Path to in-distribution CSV
        ood_path: Path to out-of-distribution CSV
        output_path: Optional path to save output
        bold_best: Whether to bold best values
    """
    # Read the CSVs
    df_indist = pd.read_csv(indist_path)
    df_ood = pd.read_csv(ood_path)

    # Generate individual tables
    table_indist = generate_single_table(
        df_indist,
        "Multi-Task Ablation: In-Distribution Performance",
        "indist",
        bold_best
    )

    table_ood = generate_single_table(
        df_ood,
        "Multi-Task Ablation: Out-of-Distribution Performance",
        "ood",
        bold_best
    )

    # Generate average table
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
        description="Generate LaTeX tables from multi-task ablation summary CSVs"
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

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.indist).exists():
        print(f"Error: In-distribution file not found: {args.indist}")
        return

    if not Path(args.ood).exists():
        print(f"Error: Out-of-distribution file not found: {args.ood}")
        return

    # Generate tables
    generate_multitask_tables(args.indist, args.ood, args.output, bold_best=args.bold_best)


if __name__ == "__main__":
    main()
