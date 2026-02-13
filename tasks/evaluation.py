"""Downstream evaluation tasks.

This module contains invoke tasks for running downstream prediction
experiments and generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.evaluation import baseline_experiment, brainharmonix_experiment
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
def run_baseline(c, experiment="all"):
    """Generate baseline for downstream prediction experiments.

    Uses SVM and linear models with 100-fold cross-validation.

    Example:
        inv evaluation.run-baseline
        inv evaluation.run-baseline --experiment=brainharmonix
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
        baseline_experiment(i, o)


@invoke.task(
    help={
        "experiment": "Which experiment to run: all, finetune, representation (default: all)",
        "output-dir": "Output directory for experiment results (default: outputs/downstreams/brainharmonix)",
    }
)
def run_brainharmonix_experiment(c, experiment="all", output_dir=DEFAULT_OUTPUT_DIR / "brainharmonix"):
    """Run downstream experiments with BrainHarmonix features."""

    if experiment not in ["all", "finetune", "representation"]:
        raise NotImplementedError

    emb_paths = []
    outputs = []
    if experiment in ["all", "finetune"]:
        emb_paths.append("outputs/embeddings/brainharmonix/nozscore.brainharmonix.finetuned.embeddings.npz")
        outputs.append(output_dir / "nozscore.finetuned.brainharmonix")
        emb_paths.append("outputs/embeddings/brainharmonix/zscore.brainharmonix.finetuned.embeddings.npz")
        outputs.append(output_dir / "zscore.finetuned.brainharmonix")


    if experiment in ["all", "representation"]:
        emb_paths.append("outputs/embeddings/brainharmonix/nozscore.brainharmonix.embeddings.npz")
        outputs.append(output_dir / "nozscore.brainharmonix")
        emb_paths.append("outputs/embeddings/brainharmonix/zscore.brainharmonix.embeddings.npz")
        outputs.append(output_dir / "zscore.brainharmonix")

    print(f"Running {len(emb_paths)} experiments.")

    for emb_path, output_dir in zip(emb_paths, outputs):
        brainharmonix_experiment(emb_path=emb_path, output_dir=output_dir)
