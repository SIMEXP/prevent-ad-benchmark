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
    'cls-token': 'CLS Token',
    'cls-embedding': 'CLS Embedding',
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


def parse_filename(filepath: Path) -> dict:
    """Parse filename to extract feature, target, and classifier type."""
    # Pattern: x-{feature}_y-{target}_{classifier}_prediction.tsv
    pattern = r'x-(.+)_y-(.+)_(svm|linear)_prediction\.tsv'
    match = re.match(pattern, filepath.name)
    if match:
        return {
            'feature': match.group(1),
            'target': match.group(2),
            'classifier': match.group(3),
        }
    return None


def load_results(input_dirs: list[Path]) -> pd.DataFrame:
    """Load all result files from multiple directories into a single DataFrame."""
    records = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping")
            continue

        source = input_dir.name  # e.g., 'baseline' or 'brainharmonix'
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
            for metric in ['accuracy', 'auc', 'f1']:
                mean = group[metric].mean()
                std = group[metric].std()
                record[metric.upper()] = f'{mean:.3f} ± {std:.3f}'
        else:
            for metric, col in [('RMSE', 'rmse'), ('MAE', 'mae'), ('R²', 'r2')]:
                mean = group[col].mean()
                std = group[col].std()
                record[metric] = f'{mean:.3f} ± {std:.3f}'

        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / 'summary_table.tsv', index=False, sep='\t')
    return summary_df
