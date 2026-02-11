"""Downstream evaluation tasks.

This module contains invoke tasks for running downstream prediction
experiments and generating result visualizations.
"""
from pathlib import Path
from preventad_benchmark.evaluation import baseline_experiment
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
        "input-dirs": "Space-separated list of directories with result TSV files",
        "output-dir": "Directory to save figures (default: figures/)",
        "show": "Display plots interactively",
    }
)
def plot_results(c, input_dirs=None, output_dir="figures/", show=False):
    """Generate plots from downstream experiment results.

    Creates visualizations:
    - Bar plots with scatter overlays for each metric
    - Heatmaps comparing features and targets
    - Summary tables in CSV format

    Example:
        inv evaluation.plot-results
        inv evaluation.plot-results --output-dir=./paper-figures/
        inv evaluation.plot-results --show
    """
    cmd = "preventad-plot"
    if input_dirs:
        cmd += f" --input-dirs {input_dirs}"
    cmd += f" --output-dir {output_dir}"
    if show:
        cmd += " --show"

    print(f"Running: {cmd}")
    c.run(cmd)


@invoke.task(
    help={
        "dataset": "Path to BrainHarmonix Arrow dataset",
        "harmonizer-ckpt": "Path to fine-tuned harmonizer checkpoint",
        "output-dir": "Output directory for embeddings",
        "output-prefix": "Prefix for output files",
    }
)
def extract_brainharmonix(
    c,
    dataset=None,
    harmonizer_ckpt=None,
    output_dir="outputs/embeddings/brainharmonix",
    output_prefix="brainharmonix",
):
    """Extract embeddings from BrainHarmonix model.

    Extracts embeddings from:
    - fMRI encoder (Harmonix-F): 7200 tokens × 768 dim
    - T1 encoder (Harmonix-S): 1200 tokens × 768 dim
    - Harmonizer: 129 tokens × 768 dim (1 CLS + 128 latent)

    Example:
        inv evaluation.extract-brainharmonix --dataset=./data/processed/dataset.arrow
        inv evaluation.extract-brainharmonix --harmonizer-ckpt=./outputs/finetuned/checkpoint.pt
    """
    dataset = dataset or "data/processed/dataset-preventad.brainharmonix.NoGrandmeanScaling.arrow"

    cmd = f"preventad-extract-brainharmonix --dataset {dataset} --output-dir {output_dir} --output-prefix {output_prefix}"
    if harmonizer_ckpt:
        cmd += f" --harmonizer-ckpt {harmonizer_ckpt}"

    print(f"Running: {cmd}")
    c.run(cmd)


@invoke.task(
    help={
        "dataset": "Path to BrainHarmonix Arrow dataset",
        "task": "Task type: self-supervised, classification, or regression",
        "target": "Target column for supervised tasks (e.g., Sex, Candidate_Age)",
        "epochs": "Number of training epochs (default: 50)",
        "lr": "Learning rate (default: 1e-5)",
        "batch-size": "Batch size (default: 4)",
        "output-dir": "Output directory for checkpoints",
    }
)
def finetune_brainharmonix(
    c,
    dataset=None,
    task="self-supervised",
    target=None,
    epochs=50,
    lr="1e-5",
    batch_size=4,
    output_dir=None,
):
    """Fine-tune BrainHarmonix harmonizer.

    Supports three training modes:
    - self-supervised: Adapt to dataset distribution (no labels needed)
    - classification: Supervised classification (e.g., sex, disease)
    - regression: Supervised regression (e.g., age prediction)

    Example:
        inv evaluation.finetune-brainharmonix --task=self-supervised
        inv evaluation.finetune-brainharmonix --task=classification --target=Sex
        inv evaluation.finetune-brainharmonix --task=regression --target=Candidate_Age
    """
    dataset = dataset or "data/processed/dataset-preventad.brainharmonix.NoGrandmeanScaling.arrow"
    output_dir = output_dir or f"outputs/finetune/brainharmonix-{task}"

    cmd = f"preventad-finetune-brainharmonix --dataset {dataset} --task {task} --epochs {epochs} --lr {lr} --batch-size {batch_size} --output-dir {output_dir}"
    if target and task != "self-supervised":
        cmd += f" --target {target}"

    print(f"Running: {cmd}")
    c.run(cmd)
