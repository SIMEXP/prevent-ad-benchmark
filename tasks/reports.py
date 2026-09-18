"""Reports generation tasks.

This module contains invoke tasks for generating result visualizations.
"""
from pathlib import Path
import pandas as pd
from preventad_benchmark.plotting.utils import load_results, make_summary_table
from preventad_benchmark.plotting.learning_curves import (
    plot_brainharmony_curves,
    plot_brainlm_curves,
    plot_brainlm_curves_by_condition,
    plot_combined_curves,
)
from preventad_benchmark.plotting.classification_summary import plot_classification_summary
import invoke


PROJECT_ROOT = Path(__file__).parents[1]

# Default input directories (absolute paths)
INPUT_DIRS = {
    'baselines': [
        PROJECT_ROOT / 'outputs/downstreams/baseline.brainharmonix',
        PROJECT_ROOT / 'outputs/downstreams/baseline.brainlm',
    ],
    'brainharmonix': [
        PROJECT_ROOT / 'outputs/downstreams/brainharmonix',
    ],
    'brainlm': [
        PROJECT_ROOT / 'outputs/downstreams/brainlm',
    ],
}
@invoke.task(
    help={
        "model": "brainharmony, brainlm, or combined (default: combined)",
        "output-dir": "Directory to save figures (default: outputs/reports/learning_curves/)",
    }
)
def plot_learning_curves(c, model="combined", output_dir=None):
    """Plot finetuning learning curves from saved trainer state files.

    Both BrainHarmony and BrainLM report train + val loss per epoch (from config.json).

    Examples::

        inv reports.plot-learning-curves
        inv reports.plot-learning-curves --model brainharmony
        inv reports.plot-learning-curves --model brainlm
        inv reports.plot-learning-curves --output-dir outputs/reports/learning_curves/
    """
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs/reports/learning_curves"
    bh_dir = PROJECT_ROOT / "outputs/finetune/brainharmonix"
    bl_dir = PROJECT_ROOT / "outputs/finetune/brainlm"

    if model in ("brainharmony", "combined"):
        plot_brainharmony_curves(
            finetune_dir=bh_dir,
            output_path=out_dir / "brainharmony_learning_curves.png",
        )
    if model in ("brainlm", "combined"):
        plot_brainlm_curves(
            finetune_dir=bl_dir,
            output_path=out_dir / "brainlm_learning_curves.png",
        )
        plot_brainlm_curves_by_condition(
            finetune_dir=bl_dir,
            output_path=out_dir / "brainlm_learning_curves_by_condition.png",
        )
    if model == "combined":
        plot_combined_curves(
            brainharmony_finetune_dir=bh_dir,
            brainlm_finetune_dir=bl_dir,
            output_path=out_dir / "combined_learning_curves.png",
        )


@invoke.task(
    help={
        "experiment": "all, baselines, brainharmonix, brainlm",
        "output-dir": "Directory to save summary table (default: outputs/reports/)",
    }
)
def generate_summary(c, experiment='baselines', output_dir=PROJECT_ROOT / 'outputs/reports/'):
    """Generate summary table from downstream experiment results.
    The default will only include the baseline experiments, but you can specify other directories with the --experiment argument.

    Example:
        inv reports.generate-summary
        inv reports.generate-summary --experiment brainharmonix --output-dir outputs/reports/
    """
    if experiment == 'all':
        input_dirs = [dir for dirs in INPUT_DIRS.values() for dir in dirs]
    else:
        input_dirs = INPUT_DIRS[experiment]
    df = load_results(input_dirs)
    # Always load baselines separately for the vs-FC/vs-dummy t-tests, even when
    # summarizing a single foundation model's own results (which wouldn't
    # otherwise include the baseline/dummy rows to compare against).
    baseline_df = df if experiment in ('all', 'baselines') else load_results(INPUT_DIRS['baselines'])
    make_summary_table(df, output_dir=output_dir / experiment, baseline_df=baseline_df)


@invoke.task(
    help={
        "output-dir": "Directory to save figures (default: outputs/reports/classification_summary/)",
    }
)
def plot_classification(c, output_dir=None):
    """Plot per-target classification summary bar charts (accuracy + precision),
    ranked against the Schaefer400 functional-connectivity and dummy baselines.

    Requires `inv reports.generate-summary --experiment all` (or any experiment
    that includes both a foundation model and the baselines) to have been run
    first, since this reads summary_classification.tsv.

    Example:
        inv reports.generate-summary --experiment all
        inv reports.plot-classification
    """
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs/reports/classification_summary"
    summary_path = PROJECT_ROOT / "outputs/reports/all/summary_classification.tsv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found -- run `inv reports.generate-summary --experiment all` first."
        )
    df = pd.read_csv(summary_path, sep="\t")
    plot_classification_summary(df, output_dir=out_dir)
