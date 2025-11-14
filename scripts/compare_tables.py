#!/usr/bin/env python3
"""
Compare generated table values with expected values.
"""

import pandas as pd
from pathlib import Path

# Expected values from user's LaTeX tables
EXPECTED_SINGLE_TASK_COMPARISON = {
    'TSC': [1.91, 2.87, 3.92, 31.55, 63.37, 13.20],
    'GMM-HMM': [1.46, 2.74, 4.10, 42.49, 66.32, 6.60],
    'MS-TCN': [15.88, 23.77, 29.06, 53.67, 83.92, 1.61],
    'ASFormer': [45.66, 56.24, 60.33, 76.60, 77.13, 0.88],
    'Our Method': [50.53, 67.34, 72.03, 84.27, 92.48, 0.76],
}

EXPECTED_SINGLE_TASK_ABLATION = {
    'No Visual': [21.41, 25.54, 27.51, 21.55, 45.33, 3.89],
    'No Kinematics': [21.49, 25.76, 26.64, 17.72, 25.69, 4.65],
    'MLP + MSE': [23.52, 27.80, 29.03, 25.78, 36.69, 4.10],
    'Our Method': [50.53, 67.34, 72.03, 84.27, 92.48, 0.76],
}

EXPECTED_MULTITASK_INDIST = {
    'ASFormer': [15.04, 19.70, 22.62, 66.08, 84.97, 1.08],
    'MS-TCN': [13.29, 17.58, 20.09, 67.04, 90.41, 0.97],
    'No Kinematics': [19.36, 22.15, 22.17, 13.09, 4.69, 5.28],
    'No Visual': [21.52, 29.05, 33.41, 37.67, 4.99, 2.87],
    'Our Method': [21.06, 39.02, 47.16, 52.38, 85.55, 1.87],
}

EXPECTED_MULTITASK_OOD = {
    'ASFormer': [5.65, 8.28, 10.09, 71.47, 94.59, 0.72],
    'MS-TCN': [5.31, 7.18, 8.44, 67.39, 93.94, 0.83],
    'No Visual': [17.55, 20.05, 20.05, 11.73, 4.19, 5.41],
    'No Kinematics': [18.31, 24.65, 28.55, 32.24, 4.28, 3.14],
    'Our Method': [18.47, 34.40, 42.08, 54.18, 87.20, 1.67],
}

METRICS = ['F1@1s', 'F1@2s', 'F1@3s', 'Seg-F1@50%', 'Acc', 'Edit-Dist']


def compare_values(expected, actual, table_name):
    """Compare expected vs actual values and print differences."""
    print(f"\n{'='*80}")
    print(f"{table_name}")
    print('='*80)

    has_differences = False

    for method in expected.keys():
        if method not in actual:
            print(f"\n⚠️  METHOD MISSING: '{method}' not found in generated table")
            has_differences = True
            continue

        exp_values = expected[method]
        act_values = actual[method]

        method_has_diff = False
        for i, (metric, exp_val, act_val) in enumerate(zip(METRICS, exp_values, act_values)):
            diff = abs(exp_val - act_val)
            if diff > 0.01:  # Threshold for significant difference
                if not method_has_diff:
                    print(f"\n{method}:")
                    method_has_diff = True
                    has_differences = True

                pct_diff = (diff / exp_val * 100) if exp_val != 0 else 0
                print(f"  {metric:20s}: Expected {exp_val:6.2f}, Got {act_val:6.2f}  "
                      f"(Δ={diff:5.2f}, {pct_diff:+5.1f}%)")

    if not has_differences:
        print("\n✅ All values match expected!")

    # Check for extra methods in actual
    for method in actual.keys():
        if method not in expected:
            print(f"\n⚠️  EXTRA METHOD: '{method}' found in generated table but not expected")


def load_actual_values():
    """Load actual values from generated CSVs."""
    base_dir = Path('/home/syx/ucsf/task_seg')

    # Load and compute single-task comparison
    print("\nLoading data files...")

    import sys
    sys.path.insert(0, str(base_dir))

    # Task weights
    TASK_WEIGHTS = {
        374: 91, 440: 165, 398: 173, 327: 209, 385: 214, 389: 312,
        376: 403, 375: 422, 357: 428, 358: 470, 356: 475, 354: 516,
        360: 517, 372: 537, 359: 644, 378: 739, 390: 750, 377: 794,
        384: 808, 380: 815, 424: 916, 392: 977,
    }

    # Define functions locally
    def compute_weighted_average(df, metric, method_col='method', task_col='task'):
        """Compute weighted average for each method across tasks."""
        methods = df[method_col].unique()
        results = {}

        for method in methods:
            method_data = df[df[method_col] == method]
            weighted_sum = 0.0
            total_weight = 0.0

            for _, row in method_data.iterrows():
                task_name = row[task_col]
                metric_value = row[metric]

                if isinstance(task_name, str):
                    try:
                        task_id = int(task_name.replace('task_', ''))
                    except (ValueError, AttributeError):
                        continue
                else:
                    task_id = int(task_name)

                if task_id not in TASK_WEIGHTS or pd.isna(metric_value):
                    continue

                weight = TASK_WEIGHTS[task_id]
                weighted_sum += float(metric_value) * weight
                total_weight += weight

            if total_weight > 0:
                results[method] = weighted_sum / total_weight
            else:
                results[method] = None

        return results

    # Single task comparison
    df_ablation = pd.read_csv(base_dir / 'ablation_results/ablation_summary.csv')
    df_asformer = pd.read_csv(base_dir / 'asformer_results/asformer_summary.csv')
    df_ms_tcn = pd.read_csv(base_dir / 'ms_tcn_results/ms_tcn_summary.csv')
    df_tsc = pd.read_csv(base_dir / 'tsc_baselines_full/tsc_evaluation_summary.csv')
    df_tsc = df_tsc.rename(columns={'task_id': 'task', 'Framewise-Acc': 'Acc'})

    metrics_raw = ['F1@1.0', 'F1@2.0', 'F1@3.0', 'Seg-F1@50%', 'Acc', 'Edit-Dist']

    actual_st_comp = {}

    # TSC
    df_tsc_method = df_tsc[df_tsc['method'] == 'Change Point Detection']
    tsc_vals = []
    for m in metrics_raw:
        if m in df_tsc_method.columns:
            avg = compute_weighted_average(df_tsc_method, m, 'method', 'task')['Change Point Detection']
            tsc_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_st_comp['TSC'] = tsc_vals

    # GMM-HMM
    df_gmm = df_tsc[df_tsc['method'] == 'Time-Varying GMM']
    gmm_vals = []
    for m in metrics_raw:
        if m in df_gmm.columns:
            avg = compute_weighted_average(df_gmm, m, 'method', 'task')['Time-Varying GMM']
            gmm_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_st_comp['GMM-HMM'] = gmm_vals

    # MS-TCN
    ms_tcn_vals = []
    for m in metrics_raw:
        if m in df_ms_tcn.columns:
            avg = compute_weighted_average(df_ms_tcn, m, 'method', 'task').get('MS-TCN (2 stages, 5 layers)')
            ms_tcn_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_st_comp['MS-TCN'] = ms_tcn_vals

    # ASFormer
    asf_vals = []
    for m in metrics_raw:
        if m in df_asformer.columns:
            avg = compute_weighted_average(df_asformer, m, 'method', 'task').get('ASFormer')
            asf_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_st_comp['ASFormer'] = asf_vals

    # Our method
    df_ours = df_ablation[df_ablation['ablation'] == 'Single Task Baseline']
    our_vals = []
    for m in metrics_raw:
        if m in df_ours.columns:
            avg = compute_weighted_average(df_ours, m, 'ablation', 'task').get('Single Task Baseline')
            our_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_st_comp['Our Method'] = our_vals

    # Single task ablation
    actual_st_abl = {}
    ablation_methods = {
        'Pure Proprioceptive': 'No Visual',
        'Pure Visual': 'No Kinematics',
        'MLP + MSE': 'MLP + MSE',
        'Single Task Baseline': 'Our Method'
    }

    for method_key, method_display in ablation_methods.items():
        df_method = df_ablation[df_ablation['ablation'] == method_key]
        vals = []
        for m in metrics_raw:
            if m in df_method.columns and len(df_method) > 0:
                avg = compute_weighted_average(df_method, m, 'ablation', 'task').get(method_key)
                vals.append(avg * 100 if m != 'Edit-Dist' else avg)
        actual_st_abl[method_display] = vals

    # Multi-task in-dist
    df_mt_asf_indist = pd.read_csv(base_dir / 'multi_task_asformer_results/asformer_aggregate_indist.csv')
    df_mt_mstcn_indist = pd.read_csv(base_dir / 'multi_task_ms_tcn_results/multi_task_ms_tcn_summary_indist.csv')
    df_mt_abl_indist = pd.read_csv(base_dir / 'multi_task_ablation_results/multi_task_ablation_aggregate_indist.csv')

    actual_mt_indist = {}

    # ASFormer
    asf_row = df_mt_asf_indist[df_mt_asf_indist['ablation'] == 'ASFormer']
    if len(asf_row) > 0:
        actual_mt_indist['ASFormer'] = [asf_row.iloc[0][m] * 100 if m != 'Edit-Dist' else asf_row.iloc[0][m]
                                        for m in metrics_raw]

    # MS-TCN (weighted average)
    ms_vals = []
    for m in metrics_raw:
        if m in df_mt_mstcn_indist.columns:
            values, weights = [], []
            for _, row in df_mt_mstcn_indist.iterrows():
                task_name = row['task']
                task_id = int(task_name.replace('task_', '')) if isinstance(task_name, str) else int(task_name)
                if task_id in TASK_WEIGHTS and not pd.isna(row[m]):
                    values.append(float(row[m]))
                    weights.append(TASK_WEIGHTS[task_id])
            if weights:
                avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                ms_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_mt_indist['MS-TCN'] = ms_vals

    # Ablations
    abl_map_mt = {
        'Pure Visual': 'No Kinematics',
        'Pure Proprioceptive': 'No Visual',
        'Multi-Task Baseline': 'Our Method'
    }
    for method_key, method_display in abl_map_mt.items():
        method_row = df_mt_abl_indist[df_mt_abl_indist['ablation'] == method_key]
        if len(method_row) > 0:
            actual_mt_indist[method_display] = [method_row.iloc[0][m] * 100 if m != 'Edit-Dist' else method_row.iloc[0][m]
                                                for m in metrics_raw if m in method_row.columns]

    # Multi-task OOD
    df_mt_asf_ood = pd.read_csv(base_dir / 'multi_task_asformer_results/asformer_aggregate_ood.csv')
    df_mt_mstcn_ood = pd.read_csv(base_dir / 'multi_task_ms_tcn_results/multi_task_ms_tcn_summary_ood.csv')
    df_mt_abl_ood = pd.read_csv(base_dir / 'multi_task_ablation_results/multi_task_ablation_aggregate_ood.csv')

    actual_mt_ood = {}

    # ASFormer
    asf_row = df_mt_asf_ood[df_mt_asf_ood['ablation'] == 'ASFormer']
    if len(asf_row) > 0:
        actual_mt_ood['ASFormer'] = [asf_row.iloc[0][m] * 100 if m != 'Edit-Dist' else asf_row.iloc[0][m]
                                     for m in metrics_raw]

    # MS-TCN
    ms_vals = []
    for m in metrics_raw:
        if m in df_mt_mstcn_ood.columns:
            values, weights = [], []
            for _, row in df_mt_mstcn_ood.iterrows():
                task_name = row['task']
                task_id = int(task_name.replace('task_', '')) if isinstance(task_name, str) else int(task_name)
                if task_id in TASK_WEIGHTS and not pd.isna(row[m]):
                    values.append(float(row[m]))
                    weights.append(TASK_WEIGHTS[task_id])
            if weights:
                avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                ms_vals.append(avg * 100 if m != 'Edit-Dist' else avg)
    actual_mt_ood['MS-TCN'] = ms_vals

    # Ablations
    for method_key, method_display in abl_map_mt.items():
        method_row = df_mt_abl_ood[df_mt_abl_ood['ablation'] == method_key]
        if len(method_row) > 0:
            actual_mt_ood[method_display] = [method_row.iloc[0][m] * 100 if m != 'Edit-Dist' else method_row.iloc[0][m]
                                             for m in metrics_raw if m in method_row.columns]

    return actual_st_comp, actual_st_abl, actual_mt_indist, actual_mt_ood


def main():
    print("Comparing generated tables with expected values...")

    actual_st_comp, actual_st_abl, actual_mt_indist, actual_mt_ood = load_actual_values()

    compare_values(EXPECTED_SINGLE_TASK_COMPARISON, actual_st_comp,
                   "Table 1: Single Task Performance Comparison")

    compare_values(EXPECTED_SINGLE_TASK_ABLATION, actual_st_abl,
                   "Table 2: Single Task Ablation Studies")

    compare_values(EXPECTED_MULTITASK_INDIST, actual_mt_indist,
                   "Table 3: Multi-Task In-Distribution")

    compare_values(EXPECTED_MULTITASK_OOD, actual_mt_ood,
                   "Table 4: Multi-Task Out-of-Distribution")

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
