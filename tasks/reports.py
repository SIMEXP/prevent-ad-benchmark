"""Reports generation tasks.

This module contains invoke tasks for generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.plotting.utils import load_results, make_summary_table
import invoke


PROJECT_ROOT = Path(__file__).parents[1]

# Default input directories (absolute paths)
INPUT_DIRS = {
    'baseline':[
        PROJECT_ROOT / 'outputs/downstreams/baseline.brainharmonix',
        PROJECT_ROOT / 'outputs/downstreams/baseline.brainlm',
    ],
    'brainharmonix':[
        PROJECT_ROOT / 'outputs/downstreams/brainharmonix/zscore_finetuned.brainharmonix',
        PROJECT_ROOT / 'outputs/downstreams/brainharmonix/zscore.brainharmonix',
        PROJECT_ROOT / 'outputs/downstreams/brainharmonix/nozscore_finetuned.brainharmonix',
        PROJECT_ROOT / 'outputs/downstreams/brainharmonix/nozscore.brainharmonix',
    ],
    'brainlm':[
        PROJECT_ROOT / 'outputs/downstreams/nozscore.brainlm.111M.direct_transfer',
        PROJECT_ROOT / 'outputs/downstreams/nozscore.brainlm.650M.direct_transfer',
        PROJECT_ROOT / 'outputs/downstreams/zscore.gigaconnectome.111M.direct_transfer',
        PROJECT_ROOT / 'outputs/downstreams/zscore.gigaconnectome.650M.direct_transfer',
        PROJECT_ROOT / 'outputs/downstreams/nozscore.brainlm.111M.finetuned',
        PROJECT_ROOT / 'outputs/downstreams/zscore.gigaconnectome.111M.finetuned',
    ]
}
@invoke.task(
    help={
        "experiment": "baseline, brainharmonix, brainlm",
        "output-dir": "Directory to save summary table (default: outputs/reports/baseline/)",
    }
)
def generate_summary(c, experiment='baseline', output_dir=PROJECT_ROOT / 'outputs/reports/baseline/'):
    """Generate summary table from downstream experiment results.
    The default will only include the baseline experiments, but you can specify other directories with the --input-dirs argument.

    Example:
        inv reports.generate-summary
        inv reports.generate-summary --input-dirs outputs/downstreams/brainharmonix outputs/downstreams/brainlm --output-dir outputs/reports/
    """
    input_dirs = INPUT_DIRS[experiment]
    df = load_results(input_dirs)
    make_summary_table(df, output_dir=output_dir)
