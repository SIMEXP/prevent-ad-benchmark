"""Plot downstream prediction results.

Usage:
    preventad-plot
    preventad-plot --output-dir figures/
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Feature display names
FEATURE_NAMES = {
    'fmri_mean': 'fMRI (mean)',
    't1_mean': 'T1 (mean)',
    'harmonizer_cls': 'Harmonizer (CLS)',
    'harmonizer_latent_mean': 'Harmonizer (latent)',
    'timeseries': 'Timeseries',
    'connectivity': 'Connectivity',
    'cls-token': 'CLS Token',
    'cls-embedding': 'CLS Embedding',
}

# Target display names
TARGET_NAMES = {
    'sex': 'Sex',
    'age': 'Age (years)',
    'splifhalfage': 'Age (binary)',
    'progess2mci': 'MCI Progression',
    'aps': 'Amyloid Probability Score',
    'pet': 'Amyloid Index',
}

# Get project root (src/preventad_benchmark/cli -> project root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

# Default input directories (absolute paths)
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / 'outputs/downstreams/baseline',
    PROJECT_ROOT / 'outputs/downstreams/brainharmonix',
]


def parse_filename(filepath: Path) -> dict:
    """Parse filename to extract feature, target, and model type."""
    # Pattern: x-{feature}_y-{target}_{model}_prediction.tsv
    pattern = r'x-(.+)_y-(.+)_(svm|linear)_prediction\.tsv'
    match = re.match(pattern, filepath.name)
    if match:
        return {
            'feature': match.group(1),
            'target': match.group(2),
            'model': match.group(3),
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
                    'model': parsed['model'],
                    'source': source,
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


def bar_with_scatter(data, x, y, hue, ax, palette='Set2', scatter_alpha=0.3, scatter_size=3):
    """Create a bar plot with scatter overlay showing individual data points."""
    # Calculate means and stds for bar plot
    summary = data.groupby([x, hue])[y].agg(['mean', 'std']).reset_index()

    # Get unique categories
    x_categories = data[x].unique()
    hue_categories = data[hue].unique()
    n_hue = len(hue_categories)

    # Set up positions
    x_positions = np.arange(len(x_categories))
    width = 0.8 / n_hue

    colors = sns.color_palette(palette, n_hue)

    for i, hue_val in enumerate(hue_categories):
        hue_data = summary[summary[hue] == hue_val]

        # Match order with x_categories
        means = []
        stds = []
        for cat in x_categories:
            row = hue_data[hue_data[x] == cat]
            if len(row) > 0:
                means.append(row['mean'].values[0])
                stds.append(row['std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        positions = x_positions + (i - n_hue / 2 + 0.5) * width

        # Bar plot
        bars = ax.bar(positions, means, width, label=hue_val, color=colors[i], alpha=0.7)

        # Error bars
        ax.errorbar(positions, means, yerr=stds, fmt='none', color='black', capsize=2, linewidth=1)

        # Scatter overlay
        for j, cat in enumerate(x_categories):
            cat_data = data[(data[x] == cat) & (data[hue] == hue_val)][y]
            jitter = np.random.normal(0, width * 0.15, len(cat_data))
            ax.scatter(
                positions[j] + jitter,
                cat_data,
                color=colors[i],
                alpha=scatter_alpha,
                s=scatter_size,
                zorder=5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_categories)

    return ax


def plot_classification_results(df: pd.DataFrame, output_dir: Path = None) -> plt.Figure:
    """Create bar plots with scatter for classification metrics."""
    df_cls = df[df['task_type'] == 'classification'].copy()
    if df_cls.empty:
        return None

    # Map feature and target names for display
    df_cls['Feature'] = df_cls['feature'].map(lambda x: FEATURE_NAMES.get(x, x))
    df_cls['Target'] = df_cls['target'].map(lambda x: TARGET_NAMES.get(x, x))

    targets = df_cls['Target'].unique()
    models = df_cls['model'].unique()
    metrics = ['accuracy', 'auc', 'f1']
    metric_labels = ['Accuracy', 'AUC', 'F1 Score']

    n_targets = len(targets)
    fig, axes = plt.subplots(n_targets, 3, figsize=(14, 4 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for i, target in enumerate(targets):
        df_target = df_cls[df_cls['Target'] == target]

        for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]

            bar_with_scatter(
                data=df_target,
                x='Feature',
                y=metric,
                hue='model',
                ax=ax,
            )

            ax.set_title(f'{target} - {label}')
            ax.set_xlabel('')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)

            if j == 2:  # Only show legend on rightmost plot
                ax.legend(title='Model', loc='upper right')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'classification_results.png', dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / 'classification_results.pdf', bbox_inches='tight')

    return fig


def plot_regression_results(df: pd.DataFrame, output_dir: Path = None) -> plt.Figure:
    """Create bar plots with scatter for regression metrics."""
    df_reg = df[df['task_type'] == 'regression'].copy()
    if df_reg.empty:
        return None

    # Map feature names for display
    df_reg['Feature'] = df_reg['feature'].map(lambda x: FEATURE_NAMES.get(x, x))
    df_reg['Target'] = df_reg['target'].map(lambda x: TARGET_NAMES.get(x, x))

    targets = df_reg['Target'].unique()
    metrics = ['rmse', 'mae', 'r2']
    metric_labels = ['RMSE', 'MAE', 'R²']

    n_targets = len(targets)
    fig, axes = plt.subplots(n_targets, 3, figsize=(14, 4 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for i, target in enumerate(targets):
        df_target = df_reg[df_reg['Target'] == target]

        for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]

            bar_with_scatter(
                data=df_target,
                x='Feature',
                y=metric,
                hue='model',
                ax=ax,
            )

            ax.set_title(f'{target} - {label}')
            ax.set_xlabel('')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)

            if j == 2:
                ax.legend(title='Model', loc='upper right')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'regression_results.png', dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / 'regression_results.pdf', bbox_inches='tight')

    return fig


def plot_summary_table(df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """Create summary table with mean ± std for all metrics."""
    summary_records = []

    for (feature, target, model), group in df.groupby(['feature', 'target', 'model']):
        record = {
            'Feature': FEATURE_NAMES.get(feature, feature),
            'Target': TARGET_NAMES.get(target, target),
            'Model': model.upper(),
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
        summary_df.to_csv(output_dir / 'summary_table.csv', index=False)

    return summary_df


def plot_feature_comparison(df: pd.DataFrame, output_dir: Path = None) -> plt.Figure:
    """Create a heatmap comparing features across targets using primary metrics."""
    # Aggregate by feature, target, model
    pivot_data = []

    for (feature, target, model), group in df.groupby(['feature', 'target', 'model']):
        if group['task_type'].iloc[0] == 'classification':
            metric_value = group['auc'].mean()
        else:
            metric_value = group['r2'].mean()

        pivot_data.append({
            'Feature': FEATURE_NAMES.get(feature, feature),
            'Target': TARGET_NAMES.get(target, target),
            'Model': model.upper(),
            'Value': metric_value,
        })

    pivot_df = pd.DataFrame(pivot_data)

    # Create separate heatmaps for SVM and Linear
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model in zip(axes, ['SVM', 'LINEAR']):
        model_df = pivot_df[pivot_df['Model'] == model]

        if model_df.empty:
            continue

        heatmap_data = model_df.pivot(index='Feature', columns='Target', values='Value')

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f'{model} - Performance (AUC/R²)')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'feature_comparison_heatmap.png', dpi=150, bbox_inches='tight')
        fig.savefig(output_dir / 'feature_comparison_heatmap.pdf', bbox_inches='tight')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot downstream prediction results')
    parser.add_argument(
        '--input-dirs',
        type=Path,
        nargs='+',
        default=[Path(d) for d in DEFAULT_INPUT_DIRS],
        help='Directories containing result TSV files (default: baseline + brainharmonix)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'outputs/downstreams/figures',
        help='Directory to save figures',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively',
    )
    args = parser.parse_args()

    print(f"Loading results from {[str(d) for d in args.input_dirs]}")
    df = load_results(args.input_dirs)
    print(f"Loaded {len(df)} records from {df['feature'].nunique()} features, "
          f"{df['target'].nunique()} targets, {df['model'].nunique()} models")

    print("\nGenerating plots...")

    # Classification results
    fig_cls = plot_classification_results(df, args.output_dir)
    if fig_cls:
        print(f"  Saved classification_results.png/pdf")

    # Regression results
    fig_reg = plot_regression_results(df, args.output_dir)
    if fig_reg:
        print(f"  Saved regression_results.png/pdf")

    # Feature comparison heatmap
    fig_heatmap = plot_feature_comparison(df, args.output_dir)
    print(f"  Saved feature_comparison_heatmap.png/pdf")

    # Summary table
    summary_df = plot_summary_table(df, args.output_dir)
    print(f"  Saved summary_table.csv")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
