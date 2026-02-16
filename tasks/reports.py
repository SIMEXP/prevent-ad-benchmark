"""Reports generation tasks.

This module contains invoke tasks for generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.plotting.utils import load_results, make_summary_table
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
        "experiment": "baseline, brainharmonix, brainlm",
        "output-dir": "Directory to save summary table (default: outputs/reports/)",
    }
)
def generate_summary(c, experiment='baselines', output_dir=PROJECT_ROOT / 'outputs/reports/'):
    """Generate summary table from downstream experiment results.
    The default will only include the baseline experiments, but you can specify other directories with the --input-dirs argument.

    Example:
        inv reports.generate-summary
        inv reports.generate-summary --input-dirs outputs/downstreams/brainharmonix outputs/downstreams/brainlm --output-dir outputs/reports/
    """
    input_dirs = INPUT_DIRS[experiment]
    df = load_results(input_dirs)
    make_summary_table(df, output_dir=output_dir / experiment)
