"""Reports generation tasks.

This module contains invoke tasks for generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.plotting.utils import load_results, make_summary_table
from preventad_benchmark.plotting.learning_curves import (
    plot_brainharmony_curves,
    plot_brainlm_curves,
    plot_combined_curves,
)
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

    BrainHarmony: train + val loss per epoch (from config.json).
    BrainLM: training loss per epoch only (val loss was not logged).

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
    make_summary_table(df, output_dir=output_dir / experiment)
