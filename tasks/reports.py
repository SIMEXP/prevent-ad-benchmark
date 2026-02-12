"""Reports generation tasks.

This module contains invoke tasks for generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.plotting.utils import load_results, make_summary_table
import invoke


# Default input directories (absolute paths)
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / 'outputs/downstreams/baseline.brainharmonix',
    PROJECT_ROOT / 'outputs/downstreams/baseline.brainlm',
]

@invoke.task(
    help={
        "input-dirs": "Space-separated list of directories with result TSV files (default: outputs/downstreams/baseline.brainharmonix and outputs/downstreams/baseline.brainlm)",
        "output-dir": "Directory to save summary table (default: outputs/reports/baseline/)",
    }
)
def generate_summary(c, input_dirs=None, output_dir=PROJECT_ROOT / 'outputs/reports/baseline/'):
    """Generate summary table from downstream experiment results.
    The default will only include the baseline experiments, but you can specify other directories with the --input-dirs argument.

    Example:
        inv reports.generate-summary
        inv reports.generate-summary --input-dirs outputs/downstreams/brainharmonix outputs/downstreams/brainlm --output-dir outputs/reports/
    """
    input_dirs = input_dirs or DEFAULT_INPUT_DIRS
    df = load_results(input_dirs)
    summary_df = make_summary_table(df, output_dir=output_dir)
