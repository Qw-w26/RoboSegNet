#!/usr/bin/env python3
"""
Generate LaTeX table from task statistics CSV.
Includes task ID, average transitions, average length, scenarios, and skills.

Usage:
    python scripts/generate_task_statistics_table.py --output task_statistics_table.tex
"""

import pandas as pd
import argparse
from pathlib import Path
import re
import ast
from typing import Dict, List, Tuple


def extract_skills_from_action_text(action_text_str: str) -> List[str]:
    """Extract unique skills from the action_text column.

    Args:
        action_text_str: String representation of action text list with skills in parentheses

    Returns:
        List of unique skills
    """
    # Parse the string as a Python list
    try:
        actions = ast.literal_eval(action_text_str)
    except:
        return []

    # Extract skills from parentheses
    skills = []
    for action in actions:
        # Find text in parentheses at the end: e.g., "text.(Skill)"
        match = re.search(r'\((\w+)\)$', action)
        if match:
            skills.append(match.group(1))

    # Return unique skills, preserving order
    seen = set()
    unique_skills = []
    for skill in skills:
        if skill not in seen:
            seen.add(skill)
            unique_skills.append(skill)

    return unique_skills


def load_task_parquet_statistics(
    task_id: int,
    meta_dir: str = '/shared/projects/agibot/full_dataset'
) -> Tuple[float, float, int]:
    """Load parquet files for a task and compute statistics.

    Args:
        task_id: Task ID
        meta_dir: Directory containing task_*_processed folders

    Returns:
        Tuple of (avg_transitions, avg_length_sec, num_trajectories)
    """
    meta_path = Path(meta_dir)
    task_dir = meta_path / f'task_{task_id}_processed'

    if not task_dir.exists():
        print(f"Warning: Task directory not found: {task_dir}")
        return None, None, 0

    # Load all parquet files (train, val, test)
    dfs = []
    for split in ['train', 'val', 'test']:
        parquet_file = task_dir / f'{split}_processed.parquet'
        if parquet_file.exists():
            dfs.append(pd.read_parquet(parquet_file))

    if not dfs:
        print(f"Warning: No parquet files found for task {task_id}")
        return None, None, 0

    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Compute statistics
    num_trajectories = len(df)

    # Average number of transitions (segments)
    avg_transitions = df['segments'].apply(len).mean()

    # Average length in seconds (proprio_shape[0] / 30 Hz)
    def get_proprio_length(proprio_shape):
        # proprio_shape is like [1117, 14] (can be numpy array or list)
        try:
            if hasattr(proprio_shape, '__len__') and len(proprio_shape) >= 1:
                return float(proprio_shape[0]) / 30.0  # 30 Hz
        except:
            pass
        return None

    proprio_lengths = df['proprio_shape'].apply(get_proprio_length)
    avg_length_sec = proprio_lengths.mean()

    return avg_transitions, avg_length_sec, num_trajectories


def generate_latex_table(
    csv_path: str,
    meta_dir: str = '/shared/projects/agibot/full_dataset',
    output_path: str = None,
    bold_best: bool = False,
    indist_task_ids: List[int] = None,
    ood_task_ids: List[int] = None
):
    """
    Generate LaTeX table from task statistics CSV.

    Args:
        csv_path: Path to the CSV file with task statistics
        meta_dir: Directory containing task_*_processed folders
        output_path: Optional path to save the LaTeX output
        bold_best: Whether to bold the best values (not used for this table)
        indist_task_ids: List of in-distribution task IDs (optional)
        ood_task_ids: List of out-of-distribution task IDs (optional)
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Filter to specific tasks if provided
    if indist_task_ids is not None or ood_task_ids is not None:
        all_task_ids = []
        if indist_task_ids:
            all_task_ids.extend(indist_task_ids)
        if ood_task_ids:
            all_task_ids.extend(ood_task_ids)
        df = df[df['task_id'].isin(all_task_ids)]

    # Collect data for each task
    table_rows = []

    for _, row in df.iterrows():
        task_id = row['task_id']
        task_name = row['task_name_EN']
        scenarios = row['scenarios']
        action_text = row['example of action_text']

        # Extract skills
        skills = extract_skills_from_action_text(action_text)
        skills_str = ', '.join([skill.lower() for skill in skills])

        # Load parquet statistics
        avg_transitions, avg_length_sec, num_trajectories = load_task_parquet_statistics(
            task_id, meta_dir
        )

        # Format values
        if avg_transitions is not None:
            avg_transitions_str = f"{avg_transitions:.1f}"
        else:
            avg_transitions_str = "--"

        if avg_length_sec is not None:
            avg_length_str = f"{avg_length_sec:.1f}"
        else:
            avg_length_str = "--"

        table_rows.append({
            'task_id': task_id,
            'task_name': task_name,
            'avg_transitions': avg_transitions_str,
            'avg_length': avg_length_str,
            'num_trajectories': num_trajectories if num_trajectories > 0 else '--',
            'scenarios': scenarios,
            'skills': skills_str
        })

    # Sort table rows: in-dist tasks first, then OOD tasks
    if indist_task_ids is not None and ood_task_ids is not None:
        # Create sorting key
        def sort_key(row):
            task_id = row['task_id']
            if task_id in indist_task_ids:
                return (0, indist_task_ids.index(task_id))
            elif task_id in ood_task_ids:
                return (1, ood_task_ids.index(task_id))
            else:
                return (2, task_id)

        table_rows.sort(key=sort_key)

    # Calculate summary statistics for caption (totals)
    total_trajectories = sum(row['num_trajectories'] for row in table_rows if isinstance(row['num_trajectories'], int))

    # Calculate total transitions and total length
    total_transitions = 0
    total_length = 0

    for row in table_rows:
        if row['avg_transitions'] != '--' and isinstance(row['num_trajectories'], int):
            total_transitions += float(row['avg_transitions']) * row['num_trajectories']
        if row['avg_length'] != '--' and isinstance(row['num_trajectories'], int):
            total_length += float(row['avg_length']) * row['num_trajectories']

    # Start building LaTeX table
    latex_lines = []

    # Table header with summary statistics in caption
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    caption_text = f"Task Statistics: Dataset Overview ({total_trajectories} trajectories, {int(total_transitions)} transitions, {total_length:.0f}s total length)"
    latex_lines.append(f"    \\caption{{{caption_text}}}")
    latex_lines.append(r"    \label{tab:task-statistics}")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append("")

    # Define column format: Task ID (c), Task Name (l), 3 numeric cols (r), scenarios (l)
    col_format = "clrrrl"

    latex_lines.append(r"    \resizebox{\textwidth}{!}{")
    latex_lines.append(f"    \\begin{{tabular}}{{@{{}}{col_format}@{{}}}}")
    latex_lines.append(r"        \toprule")

    # Column headers
    header_parts = [
        r"\textbf{Task ID}",
        r"\textbf{Task Name}",
        r"\textbf{Avg. Trans.}",
        r"\textbf{Avg. Length (s)}",
        r"\textbf{\# Traj.}",
        r"\textbf{Scenarios}"
    ]
    latex_lines.append("        " + " & ".join(header_parts) + r" \\")
    latex_lines.append(r"        \midrule")

    # Generate data rows
    for i, row_data in enumerate(table_rows):
        row_parts = [
            str(row_data['task_id']),
            row_data['task_name'],
            row_data['avg_transitions'],
            row_data['avg_length'],
            str(row_data['num_trajectories']),
            row_data['scenarios']
        ]
        latex_lines.append("        " + " & ".join(row_parts) + r" \\")

        # Add midrule after in-distribution tasks (before OOD tasks)
        if indist_task_ids is not None and ood_task_ids is not None:
            if i + 1 < len(table_rows):
                current_task_id = row_data['task_id']
                next_task_id = table_rows[i + 1]['task_id']
                # Check if we're transitioning from in-dist to OOD
                if current_task_id in indist_task_ids and next_task_id in ood_task_ids:
                    latex_lines.append(r"        \midrule")

    # Close table
    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    }")

    # Add caption notes
    latex_lines.append(r"    \captionsetup{font=scriptsize, justification=justified}")
    latex_lines.append(r"    \caption*{")
    latex_lines.append(r"        \textit{Note:} ")
    latex_lines.append(r"        Average transitions and length are computed per task across all trajectories (train/val/test). ")
    latex_lines.append(r"        Length is measured from proprioception data recorded at 30Hz. ")
    latex_lines.append(r"        Caption shows total counts across all tasks and trajectories. ")
    latex_lines.append(r"        '--' indicates missing or unavailable data.")
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

    # Also generate a summary CSV
    summary_df = pd.DataFrame(table_rows)
    summary_csv = Path(csv_path).parent / 'task_statistics_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")

    return latex_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from task statistics CSV"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/home/syx/ucsf/task_seg/AgiBot World Task Statistics - AgiBotWorld-Alpha.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--meta-dir',
        type=str,
        default='/shared/projects/agibot/full_dataset',
        help='Directory containing task_*_processed folders'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output .tex file (optional)'
    )
    parser.add_argument(
        '--use-ablation-tasks',
        action='store_true',
        help='Use only the 21 tasks from multi-task ablation study'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Define task IDs from multi-task ablation study
    if args.use_ablation_tasks:
        # In-distribution tasks (training tasks)
        indist_task_ids = [374, 398, 327, 385, 389, 376, 375, 357, 358, 356, 354,
                          360, 372, 359, 378, 390, 377, 384]
        # Out-of-distribution tasks
        ood_task_ids = [380, 424, 392]
    else:
        indist_task_ids = None
        ood_task_ids = None

    # Generate table
    generate_latex_table(args.input, args.meta_dir, args.output,
                        indist_task_ids=indist_task_ids,
                        ood_task_ids=ood_task_ids)


if __name__ == "__main__":
    main()
