import re
from pathlib import Path
import pandas as pd


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

    dfs = []
    if baseline_dirs:
        dfs.append(_load_baseline_results(baseline_dirs))
    if foundation_dirs:
        dfs.append(_load_foundation_results(foundation_dirs))

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_summary_table(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """Create summary table with mean ± std for all metrics."""
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
                mean = group[metric].mean()
                # report confience interval instead of SD as these
                # splits are autocorrelated
                ci_lower = group[metric].quantile(0.025)
                ci_upper = group[metric].quantile(0.975)
                record[metric.upper()] = f'{mean:.3f} [{ci_lower:.3f} {ci_upper:.3f}]'
        else:
            for metric, col in [('RMSE', 'rmse'), ('MAE', 'mae'), ('R²', 'r2')]:
                mean = group[col].mean()
                # report confience interval instead of SD as these
                # splits are autocorrelated
                ci_lower = group[col].quantile(0.025)
                ci_upper = group[col].quantile(0.975)
                record[metric] = f'{mean:.3f} [{ci_lower:.3f} {ci_upper:.3f}]'

        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        clf_cols = ['Foundation Model', 'Atlas', 'Variation', 'Feature', 'Target', 'Classifier', 'ACCURACY', 'AUC', 'F1', 'PRECISION']
        reg_cols = ['Foundation Model', 'Atlas', 'Variation', 'Feature', 'Target', 'Classifier', 'RMSE', 'MAE', 'R²']

        clf_df = summary_df[summary_df['ACCURACY'].notna()][
            [c for c in clf_cols if c in summary_df.columns]
        ]
        reg_df = summary_df[summary_df['RMSE'].notna()][
            [c for c in reg_cols if c in summary_df.columns]
        ]

        if not clf_df.empty:
            clf_df.to_csv(output_dir / 'summary_classification.tsv', index=False, sep='\t')
        if not reg_df.empty:
            reg_df.to_csv(output_dir / 'summary_regression.tsv', index=False, sep='\t')

    return summary_df
