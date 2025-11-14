#!/usr/bin/env python3
"""
Generate LaTeX tables for threshold sensitivity analysis.
"""

import pandas as pd
import numpy as np


def format_value(mean, is_best=False):
    """Format mean as percentage with 2 decimal places."""
    mean_pct = mean * 100
    if is_best:
        return f"\\textbf{{{mean_pct:.2f}}}"
    else:
        return f"{mean_pct:.2f}"


def generate_merged_latex_table(indist_csv_path, ood_csv_path, caption, label):
    """Generate a merged LaTeX table with in-dist and ood as meta-columns."""

    # Read both CSVs
    df_indist = pd.read_csv(indist_csv_path)
    df_ood = pd.read_csv(ood_csv_path)

    # Extract relevant columns for in-distribution
    thresholds = df_indist['threshold'].values
    indist_f1_1s = df_indist['avg_f1@1.0'].values
    indist_f1_2s = df_indist['avg_f1@2.0'].values
    indist_f1_3s = df_indist['avg_f1@3.0'].values
    indist_seg_f1 = df_indist['avg_seg_f1@50'].values

    # Extract relevant columns for out-of-distribution
    ood_f1_1s = df_ood['avg_f1@1.0'].values
    ood_f1_2s = df_ood['avg_f1@2.0'].values
    ood_f1_3s = df_ood['avg_f1@3.0'].values
    ood_seg_f1 = df_ood['avg_seg_f1@50'].values

    # Find best values for each metric
    # In-distribution best
    indist_best_f1_1s_idx = np.argmax(indist_f1_1s)
    indist_best_f1_2s_idx = np.argmax(indist_f1_2s)
    indist_best_f1_3s_idx = np.argmax(indist_f1_3s)
    indist_best_seg_f1_idx = np.argmax(indist_seg_f1)

    # Out-of-distribution best
    ood_best_f1_1s_idx = np.argmax(ood_f1_1s)
    ood_best_f1_2s_idx = np.argmax(ood_f1_2s)
    ood_best_f1_3s_idx = np.argmax(ood_f1_3s)
    ood_best_seg_f1_idx = np.argmax(ood_seg_f1)

    # Start building the LaTeX table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("    \\centering")
    latex.append("    \\captionsetup{font=scriptsize, justification=justified}")
    latex.append("")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append("    \\footnotesize")
    latex.append("    \\setlength{\\tabcolsep}{2.5pt}")
    latex.append("")
    latex.append("    \\resizebox{\\linewidth}{!}{")
    latex.append("    \\begin{tabular}{@{}l|rrrr|rrrr@{}}")
    latex.append("        \\toprule")
    latex.append("        \\textbf{Threshold} & \\multicolumn{4}{c|}{\\textbf{In-Distribution}} & \\multicolumn{4}{c}{\\textbf{Out-of-Distribution}} \\\\")
    latex.append("        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    latex.append("         & {\\textbf{F1@1s}} & {\\textbf{F1@2s}} & {\\textbf{F1@3s}} & {\\textbf{Seg-F1@50\\%}} & {\\textbf{F1@1s}} & {\\textbf{F1@2s}} & {\\textbf{F1@3s}} & {\\textbf{Seg-F1@50\\%}} \\\\")
    latex.append("         & {(\\%)} & {(\\%)} & {(\\%)} & {(\\%)} & {(\\%)} & {(\\%)} & {(\\%)} & {(\\%)} \\\\")
    latex.append("        \\midrule")

    # Add data rows
    for i in range(len(thresholds)):
        threshold = thresholds[i]

        # In-distribution values
        indist_f1_1s_str = format_value(indist_f1_1s[i], i == indist_best_f1_1s_idx)
        indist_f1_2s_str = format_value(indist_f1_2s[i], i == indist_best_f1_2s_idx)
        indist_f1_3s_str = format_value(indist_f1_3s[i], i == indist_best_f1_3s_idx)
        indist_seg_f1_str = format_value(indist_seg_f1[i], i == indist_best_seg_f1_idx)

        # Out-of-distribution values
        ood_f1_1s_str = format_value(ood_f1_1s[i], i == ood_best_f1_1s_idx)
        ood_f1_2s_str = format_value(ood_f1_2s[i], i == ood_best_f1_2s_idx)
        ood_f1_3s_str = format_value(ood_f1_3s[i], i == ood_best_f1_3s_idx)
        ood_seg_f1_str = format_value(ood_seg_f1[i], i == ood_best_seg_f1_idx)

        latex.append(f"        {threshold:.1f} & {indist_f1_1s_str} & {indist_f1_2s_str} & {indist_f1_3s_str} & {indist_seg_f1_str} & {ood_f1_1s_str} & {ood_f1_2s_str} & {ood_f1_3s_str} & {ood_seg_f1_str} \\\\")

    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("    }")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    # Generate merged table
    caption = "\\textbf{Threshold Sensitivity Analysis}. Performance metrics across different temporal thresholds for action segmentation on in-distribution and out-of-distribution tasks."
    label = "tab:threshold-sensitivity"
    merged_table = generate_merged_latex_table(
        "threshold_sensitivity_results/threshold_sensitivity_indist.csv",
        "threshold_sensitivity_results/threshold_sensitivity_ood.csv",
        caption,
        label
    )

    # Print table
    print(merged_table)

    # Save to file
    output_path = "threshold_sensitivity_results/threshold_sensitivity_tables.tex"
    with open(output_path, 'w') as f:
        f.write(merged_table)

    print(f"\n\nTable saved to {output_path}")


if __name__ == "__main__":
    main()
