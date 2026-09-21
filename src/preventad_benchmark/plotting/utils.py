import re
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import numpy as np


# Feature display names
FEATURE_NAMES = {
    'timeseries': 'Timeseries, top 75 PCs',
    'connectivity': 'Functional connectivity',
    'fmri_mean': 'fMRI (mean)',
    't1_mean': 'T1 (mean)',
    'harmonizer_cls': 'Harmonizer (CLS)',
    'harmonizer_latent_mean': 'Harmonizer (latent)',
    'cls_token': 'CLS Token',
    'cls_embedding': 'CLS Embedding',
    'mean_embedding': 'Mean Embedding',
    'max_embedding': 'Max Embedding',
}

# Target display names
TARGET_NAMES = {
    'sex': 'Sex',
    'age': 'Age (years)',
    'splifhalfage': 'Age (binary)',
    'progess2mci': 'MCI Progression',
    'centiloidbin': 'Centiloid > 20',
    'centiloid': 'Centiloid',
    'abSUVR': 'β-amyloid SUVR',
    'abSUVRbin': 'β-amyloid SUVR > 1.26',
}

# Reverse lookup: display name -> target key
_TARGET_NAMES_REVERSE = {v: k for k, v in TARGET_NAMES.items()}


def parse_filename(filepath: Path) -> dict:
    """Parse filename to extract feature, target, and classifier type."""
    # Pattern: x-{feature}_y-{target}_{classifier}_prediction.tsv
    pattern = r'x-(.+)_y-(.+)_(svm|linear|dummy)_prediction\.tsv'
    match = re.match(pattern, filepath.name)
    if match:
        return {
            'feature': match.group(1),
            'target': match.group(2),
            'classifier': match.group(3),
        }
    return None


def _load_baseline_results(input_dirs: list[Path]) -> pd.DataFrame:
    """Load baseline result files (x-{feat}_y-{target}_{clf}_prediction.tsv format)."""
    records = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping")
            continue

        source = input_dir.name  # e.g., 'baseline.brainharmonix'
        files = list(input_dir.glob('*.tsv'))
        print(f"  Found {len(files)} files in {source}")
        variation, foundation_model = source.split('.')
        atlas = 'Schaefer400' if 'brainharmonix' in source else 'A424'

        for filepath in files:
            parsed = parse_filename(filepath)
            if parsed is None:
                print("couldn't parse {filepath}")
                continue

            df = pd.read_csv(filepath, sep='\t', index_col=0)

            # Determine if classification or regression based on columns
            is_classification = 'test_acc' in df.columns

            for idx, row in df.iterrows():
                record = {
                    'feature': parsed['feature'],
                    'target': parsed['target'],
                    'classifier': parsed['classifier'],
                    'foundation_model': foundation_model,
                    'variation': variation,
                    'atlas': atlas,
                    'split': idx,
                }

                if is_classification:
                    record['precision'] = row['test_precision']
                    record['accuracy'] = row['test_acc']
                    record['auc'] = row['test_auc']
                    record['f1'] = row['test_f1']
                    record['task_type'] = 'classification'
                else:
                    record['rmse'] = -row['test_nrmse']  # Convert from negative
                    record['mae'] = -row['test_nmae']
                    record['r2'] = row['test_r2']
                    record['task_type'] = 'regression'

                records.append(record)

    return pd.DataFrame(records)


def _load_foundation_results(input_dirs: list[Path]) -> pd.DataFrame:
    """Load foundation model result files ({variation}.{model}[.{finetuned}].split{N}.tsv format)."""
    records = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping")
            continue

        files = sorted(input_dir.glob('*.split*.tsv'))
        print(f"  Found {len(files)} split files in {input_dir.name}")

        for filepath in files:
            # Parse filename: {variation}.{foundation_model}[.{finetuned}].split{N}.tsv
            match = re.match(r'(\w+)\.(\w+)\.(finetuned\.)?split(\d+)\.tsv', filepath.name)
            if match is None:
                continue
            variation = f"{match.group(1)}"
            if match.group(3):
                variation = f"{match.group(3)}{match.group(1)}"
            foundation_model = match.group(2)
            split_idx = int(match.group(4))

            # Determine atlas from model name
            if 'brainharmonix' in foundation_model.lower():
                atlas = 'Schaefer400'
            elif 'brainlm' in foundation_model.lower():
                atlas = 'A424'
            else:
                atlas = foundation_model

            df = pd.read_csv(filepath, sep='\t', index_col=0)

            for _, row in df.iterrows():

                # Reverse-lookup Target display name -> target key
                target_display = row['Target']
                target_key = _TARGET_NAMES_REVERSE.get(target_display, target_display)
                feature = row['Features']
                classifier = row['Classifier'].lower()
                if "finetuned" in variation and feature in ["t1_mean", "fmri_mean"]:
                    continue  # the modality specific encoders are not finetuned

                # Determine task type from available columns
                is_classification = pd.notna(row.get('test_acc'))

                record = {
                    'feature': feature,
                    'target': target_key,
                    'classifier': classifier,
                    'foundation_model': foundation_model,
                    'variation': variation,
                    'atlas': atlas,
                    'split': split_idx,
                }

                if is_classification:
                    record['accuracy'] = row['test_acc']
                    record['auc'] = row['test_auc']
                    record['f1'] = row['test_f1']
                    record['precision'] = row['test_precision']
                    record['task_type'] = 'classification'
                else:
                    record['rmse'] = -row['test_nrmse']
                    record['mae'] = -row['test_nmae']
                    record['r2'] = row['test_r2']
                    record['task_type'] = 'regression'

                records.append(record)

    return pd.DataFrame(records)


def load_results(input_dirs: list[Path]) -> pd.DataFrame:
    """Load all result files from multiple directories into a single DataFrame.

    Auto-detects format per directory: directories with *.split*.tsv files use
    the foundation model loader; others use the baseline loader.
    """
    baseline_dirs = []
    foundation_dirs = []

    for d in input_dirs:
        d = Path(d).resolve()
        if d.exists() and list(d.glob('*.split*.tsv')):
            foundation_dirs.append(d)
        else:
            baseline_dirs.append(d)

    print(f"Found {len(baseline_dirs)} basline and {len(foundation_dirs)} foundation model results")

    dfs = []
    if baseline_dirs:
        dfs.append(_load_baseline_results(baseline_dirs))
    if foundation_dirs:
        dfs.append(_load_foundation_results(foundation_dirs))

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _cal_mean_ci95(values):
    mean = values.mean()
    sd = values.std(ddof=1)
    n = len(values)
    ci_lower, ci_upper = stats.t.interval(
        0.95,
        df=n-1,
        loc=mean,
        scale=sd / np.sqrt(n)
    )
    return mean, ci_lower, ci_upper


def _ttest_greater(values, reference_values):
    """One-sided Welch's t-test: does `values` exceed `reference_values`?

    Unpaired, since baseline results come from independent StratifiedShuffleSplit
    CV folds (pipelines.py) while foundation-model results come from the fixed
    train_test_split.json partitions -- these are not matched samples, so a
    paired test would be invalid. Welch's (equal_var=False) avoids assuming
    the two groups have equal variance.

    Returns (t_statistic, degrees_of_freedom, p_value). The t statistic is
    positive when `values` has the higher mean; the degrees of freedom are the
    (generally fractional) Welch-Satterthwaite approximation. All three are NaN
    if either group has fewer than 2 samples or scipy can't compute them (e.g.
    zero variance).
    """
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    if len(values) < 2 or len(reference_values) < 2:
        return np.nan, np.nan, np.nan
    try:
        result = stats.ttest_ind(values, reference_values, equal_var=False, alternative='greater')
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan, np.nan
    if np.isnan(result.statistic) or np.isnan(result.pvalue):
        # scipy still returns a placeholder df (1.0) when the test is undefined
        return np.nan, np.nan, np.nan
    return result.statistic, result.df, result.pvalue


def _get_baseline_group(baseline_df, feature, target, atlas='Schaefer400'):
    """Select one baseline comparison group's raw per-split rows.

    Restricted to `atlas` (default Schaefer400) since the baseline experiments
    were run once per atlas (Schaefer400 for brainharmonix, A424 for brainlm)
    and mixing them would compare against the wrong feature set.
    """
    if baseline_df is None or baseline_df.empty:
        return None
    mask = (
        (baseline_df['variation'] == 'baseline')
        & (baseline_df['feature'] == feature)
        & (baseline_df['atlas'] == atlas)
        & (baseline_df['target'] == target)
    )
    matched = baseline_df[mask]
    return matched if not matched.empty else None


def make_summary_table(df: pd.DataFrame, output_dir: Path = None, baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create summary table with mean and CI95% for all metrics, split into
    separate columns (`METRIC`, `METRIC_CI_LOW`, `METRIC_CI_HIGH`).

    For classification results, also runs a one-sided t-test (see _ttest_greater)
    on accuracy and precision against the Schaefer400 functional-connectivity
    baseline and the dummy-classifier baseline, adding `METRIC_T_VS_FC`/
    `METRIC_DF_VS_FC`/`METRIC_P_VS_FC`/`METRIC_SIG_VS_FC` and `METRIC_T_VS_DUMMY`/
    `METRIC_DF_VS_DUMMY`/`METRIC_P_VS_DUMMY`/`METRIC_SIG_VS_DUMMY` columns
    (T = Welch's t statistic, DF = Welch-Satterthwaite degrees of freedom,
    SIG = p < 0.05). Baseline/dummy rows themselves are skipped (comparing a
    baseline against itself isn't meaningful) and get NaN in these columns.

    Args:
        df: results to summarize (from load_results).
        output_dir: if given, writes summary_classification.tsv / summary_regression.tsv here.
        baseline_df: results to compare against for the t-tests. Defaults to `df`
            itself, so callers that already include baseline rows in `df` (e.g.
            experiment='all' or 'baselines') don't need to pass anything extra;
            callers summarizing only a foundation model's own results (e.g.
            experiment='brainharmonix') should pass the baseline results here
            explicitly so the comparison has something to compare against.
    """
    if baseline_df is None:
        baseline_df = df

    summary_records = []

    for (foundation_model, variation, feature, target, classifier, atlas), group in df.groupby(['foundation_model', 'variation', 'feature', 'target', 'classifier', 'atlas']):
        record = {
            'Foundation Model': foundation_model,
            'Atlas': atlas,
            'Variation': variation,
            'Feature': FEATURE_NAMES.get(feature, feature),
            'Target': TARGET_NAMES.get(target, target),
            'Classifier': classifier.upper(),
        }

        if group['task_type'].iloc[0] == 'classification':
            for metric in ['accuracy', 'auc', 'f1', 'precision']:
                mean, ci_lower, ci_upper = _cal_mean_ci95(group[metric])
                record[metric.upper()] = mean
                record[f'{metric.upper()}_CI_LOW'] = ci_lower
                record[f'{metric.upper()}_CI_HIGH'] = ci_upper

            is_baseline_row = variation == 'baseline'
            fc_group = None if is_baseline_row else _get_baseline_group(baseline_df, 'connectivity', target)
            dummy_group = None if is_baseline_row else _get_baseline_group(baseline_df, 'dummy', target)
            for metric in ['accuracy', 'precision']:
                for ref_name, ref_group in [('FC', fc_group), ('DUMMY', dummy_group)]:
                    if ref_group is not None:
                        t_stat, dof, p_value = _ttest_greater(group[metric], ref_group[metric])
                    else:
                        t_stat, dof, p_value = np.nan, np.nan, np.nan
                    record[f'{metric.upper()}_T_VS_{ref_name}'] = t_stat
                    record[f'{metric.upper()}_DF_VS_{ref_name}'] = dof
                    record[f'{metric.upper()}_P_VS_{ref_name}'] = p_value
                    record[f'{metric.upper()}_SIG_VS_{ref_name}'] = (p_value < 0.05) if pd.notna(p_value) else np.nan
        else:
            for metric, col in [('RMSE', 'rmse'), ('MAE', 'mae'), ('R²', 'r2')]:
                mean, ci_lower, ci_upper = _cal_mean_ci95(group[col])
                record[metric] = mean
                record[f'{metric}_CI_LOW'] = ci_lower
                record[f'{metric}_CI_HIGH'] = ci_upper
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        clf_metric_cols = []
        for metric in ['ACCURACY', 'AUC', 'F1', 'PRECISION']:
            clf_metric_cols += [metric, f'{metric}_CI_LOW', f'{metric}_CI_HIGH']
        for metric in ['ACCURACY', 'PRECISION']:
            for ref_name in ['FC', 'DUMMY']:
                clf_metric_cols += [f'{metric}_T_VS_{ref_name}', f'{metric}_DF_VS_{ref_name}', f'{metric}_P_VS_{ref_name}', f'{metric}_SIG_VS_{ref_name}']
        clf_cols = ['Foundation Model', 'Atlas', 'Variation', 'Feature', 'Target', 'Classifier'] + clf_metric_cols

        reg_metric_cols = []
        for metric in ['RMSE', 'MAE', 'R²']:
            reg_metric_cols += [metric, f'{metric}_CI_LOW', f'{metric}_CI_HIGH']
        reg_cols = ['Foundation Model', 'Atlas', 'Variation', 'Feature', 'Target', 'Classifier'] + reg_metric_cols

        clf_df = summary_df[summary_df['ACCURACY'].notna()][
            [c for c in clf_cols if c in summary_df.columns]
        ] if 'ACCURACY' in summary_df.columns else pd.DataFrame()
        reg_df = summary_df[summary_df['RMSE'].notna()][
            [c for c in reg_cols if c in summary_df.columns]
        ] if 'RMSE' in summary_df.columns else pd.DataFrame()

        if not clf_df.empty:
            clf_df.to_csv(output_dir / 'summary_classification.tsv', index=False, sep='\t')
        if not reg_df.empty:
            reg_df.to_csv(output_dir / 'summary_regression.tsv', index=False, sep='\t')

    return summary_df
