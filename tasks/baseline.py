"""Downstream evaluation tasks.

This module contains invoke tasks for running downstream prediction
experiments and generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.evaluation import run_baseline_experiment
from preventad_benchmark.plotting.utils import load_results, make_summary_table

import invoke


# Default paths
DEFAULT_OUTPUT_DIR = Path("outputs/downstreams")
BRAINHARMONIX_BASELINE_FEATURE = Path("data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow")
BRAINLM_BASELINE_FEATURE = Path("data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow")


@invoke.task(
    help={
        "experiment": "Which experiment to run: all, brainharmonix (default: all)",
    }
)
def run(c, experiment="all"):
    """Run baseline downstream prediction experiments.

    Uses SVM and linear models with cross-validation.

    Example:
        inv baseline.run
        inv baseline.run --experiment=brainharmonix
    """
    if experiment not in ["all", "brainharmonix", "brainlm"]:
        raise NotImplementedError

    inputs = []
    outputs = []
    if experiment in ["all", "brainharmonix"]:
        inputs.append(BRAINHARMONIX_BASELINE_FEATURE)
        outputs.append(DEFAULT_OUTPUT_DIR / "baseline.brainharmonix")

    if experiment in ["all", "brainlm"]:
        inputs.append(BRAINLM_BASELINE_FEATURE)
        outputs.append(DEFAULT_OUTPUT_DIR / "baseline.brainlm")


    for i, o in zip(inputs, outputs):
        print(f"Running {o.name}")
        run_baseline_experiment(i, o)

    # generate reports
    df = load_results(outputs)
    make_summary_table(df, output_dir=Path("outputs/reports/baselines"))
