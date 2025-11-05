#!/usr/bin/env python3
"""
Generate LaTeX table from ablation summary CSV.
The table will have tasks as column groups and methods as rows.

python scripts/generate_latex_table.py --output ablation_results/ablation_table.tex
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


def generate_latex_table(csv_path, output_path=None, bold_best=True, use_colors=False):
    """
    Generate LaTeX table from ablation summary CSV.

    Args:
        csv_path: Path to the CSV file
        output_path: Optional path to save the LaTeX output
        bold_best: Whether to bold the best values in each column
        use_colors: Whether to use colored column groups
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Get unique tasks and methods
    tasks = sorted(df['task'].unique())
    methods = df['ablation'].unique()

    # Metrics to include (excluding ablation, task, status)
    metrics = [col for col in df.columns if col not in ['ablation', 'task', 'status']]

    # Find best values for each metric and task (if bold_best is enabled)
    best_values = {}
    if bold_best:
        for task in tasks:
            for metric in metrics:
                task_data = df[df['task'] == task]
                max_val = task_data[metric].max()
                if pd.notna(max_val) and max_val > 0:
                    best_values[(task, metric)] = max_val

    # Start building LaTeX table
    latex_lines = []

    # Add color definitions if using colors
    if use_colors:
        latex_lines.append("% Define colors for column groups")
        latex_lines.append(r"\definecolor{f1clr}{RGB}{200,220,255}")
        latex_lines.append(r"\definecolor{otherclr}{RGB}{200,255,220}")
        latex_lines.append("")

    # Table header
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \caption{Ablation Study Results: Performance Comparison Across Tasks and Methods}")
    latex_lines.append(r"    \label{tab:ablation-results}")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \setlength{\tabcolsep}{2.5pt}")
    latex_lines.append("")

    # Calculate number of columns: 1 for method name + (num_metrics * num_tasks)
    num_cols = 1 + len(metrics) * len(tasks)

    # Define column format: l for method name, r for each metric (right-aligned numbers)
    col_format = "l" + "r" * (len(metrics) * len(tasks))

    latex_lines.append(r"    \resizebox{\textwidth}{!}{")
    latex_lines.append(f"    \\begin{{tabular}}{{@{{}}{col_format}@{{}}}}")
    latex_lines.append(r"        \toprule")

    # Generate colored metric group headers if enabled
    if use_colors:
        # Group metrics by type
        f1_metrics = [m for m in metrics if m.startswith('F1@')]
        other_metrics = [m for m in metrics if not m.startswith('F1@')]

        if f1_metrics and other_metrics:
            header_parts = [r"\multicolumn{1}{c}{}"]  # Skip method column

            # F1 metrics group
            f1_span = len(f1_metrics) * len(tasks)
            header_parts.append(f"\\multicolumn{{{f1_span}}}{{c}}{{\\cellcolor{{f1clr}}\\textbf{{F1 Scores}}}}")

            # Other metrics group
            other_span = len(other_metrics) * len(tasks)
            header_parts.append(f"\\multicolumn{{{other_span}}}{{c}}{{\\cellcolor{{otherclr}}\\textbf{{Other Metrics}}}}")

            latex_lines.append("        " + " & ".join(header_parts) + r" \\")

            # Add cmidrules
            f1_start = 2
            f1_end = f1_start + f1_span - 1
            other_start = f1_end + 1
            other_end = other_start + other_span - 1
            latex_lines.append(f"        \\cmidrule(lr){{{f1_start}-{f1_end}}}\\cmidrule(l){{{other_start}-{other_end}}}")

    # Generate task group headers (spanning multiple metrics)
    if len(tasks) > 1:
        header_line = r"        \multicolumn{1}{c}{} " # Skip method column
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
            if metric.startswith('F1@'):
                metric_name = f"F1@{metric.split('@')[1]}"
            else:
                metric_name = metric.capitalize()
            header_parts.append(f"{{\\textbf{{{metric_name}}}}}")

    latex_lines.append("        " + " & ".join(header_parts) + r" \\")

    # Add units row (all percentages)
    units_parts = [""]
    for task in tasks:
        for metric in metrics:
            units_parts.append("{(\\%)}")
    latex_lines.append("        " + " & ".join(units_parts) + r" \\")

    latex_lines.append(r"        \midrule")

    # Generate data rows
    for method in methods:
        row_parts = [method]

        for task in tasks:
            # Get the data for this method and task
            task_data = df[(df['ablation'] == method) & (df['task'] == task)]

            if len(task_data) == 0:
                # No data for this combination
                for metric in metrics:
                    row_parts.append("--")
            else:
                for metric in metrics:
                    value = task_data.iloc[0][metric]
                    # Check if this is the best value
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
    latex_lines.append(r"        F1@X.0 represents F1 score at different thresholds. ")
    latex_lines.append(r"        All values are reported as percentages. ")
    if bold_best:
        latex_lines.append(r"        Best results per metric are shown in \textbf{bold}. ")
    latex_lines.append(r"        '--' indicates no data or zero value.")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table*}")

    # Join all lines
    latex_output = "\n".join(latex_lines)

    # Print to console
    print(latex_output)

    # Optionally save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_output)
        print(f"\n\nLaTeX table saved to: {output_path}")

    return latex_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from ablation summary CSV"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/home/syx/ucsf/task_seg/ablation_results/ablation_summary.csv',
        help='Path to input CSV file'
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
        '--use-colors',
        action='store_true',
        # default = True,
        help='Use colored column groups'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Generate table
    generate_latex_table(args.input, args.output, bold_best=args.bold_best, use_colors=args.use_colors)


if __name__ == "__main__":
    main()
