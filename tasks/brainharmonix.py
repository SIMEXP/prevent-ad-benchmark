"""Invoke tasks for BrainHarmonix evaluation and fine-tuning."""

import invoke

from .slurm import load_slurm_config, submit_job_array
from preventad_benchmark.config import EVALUATION_N_SPLITS


@invoke.task(
    help={
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def finetune(c, split_index=0):
    """Fine-tune BrainHarmonix harmonizer and run prediction pipeline. For interactive jobs and debugging."""

    print("Fine-tuning BrainHarmonix harmonizer with self-supervised objective...")

    # zscore experiment
    cmd = f"preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/zscore.self-supervised/{split_index} --epoch=50 --lr=1e-4 --weight-decay=0.01 --split-index {split_index}"

    print(f"Running: {cmd}")
    c.run(cmd)

    # zscore extract
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/{split_index}/harmonizer_checkpoint_best.pt --output-prefix zscore.brainharmonix.finetuned --split-index {split_index}"

    c.run(cmd)


    # nozscore experiment
    cmd = f"preventad-finetune-brainharmonix --dataset=data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --task=self-supervised --output-dir=outputs/finetune/brainharmonix/nozscore.self-supervised/{split_index} --epoch=25 --lr=1e-4 --weight-decay=0.01 --split-index {split_index}"

    print(f"Running: {cmd}")
    c.run(cmd)

    # nozscore extract
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --harmonizer-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/{split_index}/harmonizer_checkpoint_best.pt --output-prefix nozscore.brainharmonix.finetuned --split-index {split_index}"
    c.run(cmd)


@invoke.task()
def evaluate(c, split_index=0):
    """Run pretrained BrainHarmonix prediction pipeline. For interactive jobs and debugging."""

    print("Use BrainHarmonix harmonizer as feature extractor directly.")
    # zscore extract on default checkpoint
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow --output-prefix zscore.brainharmonix --split-index {split_index}"

    c.run(cmd)

    # nozscore extract on default checkpoint
    cmd = f"preventad-extract-brainharmonix --dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow --output-prefix nozscore.brainharmonix --split-index {split_index}"

    c.run(cmd)


@invoke.task(
    help={
        "n-splits": "Number of train/test splits (default: 20)",
        "dry-run": "Print generated scripts without submitting (default: False)",
    }
)
def submit_evaluate(c, n_splits=EVALUATION_N_SPLITS, dry_run=False):
    """Submit SLURM job arrays for pretrained BrainHarmonix prediction."""
    slurm_params = load_slurm_config("extract_brainharmonix")
    array_range = f"0-{n_splits - 1}"

    configs = [
        (
            "brainharmonix_extract_zscore",
            "preventad-extract-brainharmonix "
            "--dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow "
            "--output-prefix zscore.brainharmonix "
            "--split-index $SLURM_ARRAY_TASK_ID",
        ),
        (
            "brainharmonix_extract_nozscore",
            "preventad-extract-brainharmonix "
            "--dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow "
            "--output-prefix nozscore.brainharmonix "
            "--split-index $SLURM_ARRAY_TASK_ID",
        ),
    ]

    for job_name, command in configs:
        submit_job_array(job_name, command, array_range, slurm_params, dry_run=dry_run)


@invoke.task(
    help={
        "n-splits": "Number of train/test splits (default: 20)",
        "dry-run": "Print generated scripts without submitting (default: False)",
    }
)
def submit_finetune(c, n_splits=EVALUATION_N_SPLITS, rerun_finetune=False, dry_run=False):
    """Submit SLURM job arrays for BrainHarmonix fine-tuning + prediction."""
    slurm_params = load_slurm_config("finetune_brainharmonix")
    array_range = f"0-{n_splits - 1}"

    if rerun_finetune:
        print("WARNING: This will re-run fine-tuning for all splits, which may be time-consuming. Make sure to set --rerun-finetune=False if you only want to re-run the prediction step with existing fine-tuned checkpoints.")
        configs = [
            (
                "brainharmonix_finetune_zscore",
                "preventad-finetune-brainharmonix "
                "--dataset=data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow "
                "--task=self-supervised "
                "--output-dir=outputs/finetune/brainharmonix/zscore.self-supervised/$SLURM_ARRAY_TASK_ID "
                "--epoch=100 --lr=1e-4 --weight-decay=0.01 "
                "--split-index $SLURM_ARRAY_TASK_ID"
                "\n\n"
                "preventad-extract-brainharmonix "
                "--dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow "
                "--harmonizer-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/$SLURM_ARRAY_TASK_ID/harmonizer_checkpoint_best.pt "
                "--output-prefix zscore.brainharmonix.finetuned "
                "--split-index $SLURM_ARRAY_TASK_ID",
            ),
            (
                "brainharmonix_finetune_nozscore",
                "preventad-finetune-brainharmonix "
                "--dataset=data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow "
                "--task=self-supervised "
                "--output-dir=outputs/finetune/brainharmonix/nozscore.self-supervised/$SLURM_ARRAY_TASK_ID "
                "--epoch=50 --lr=1e-4 --weight-decay=0.01 "
                "--split-index $SLURM_ARRAY_TASK_ID"
                "\n\n"
                "preventad-extract-brainharmonix "
                "--dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow "
                "--harmonizer-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/$SLURM_ARRAY_TASK_ID/harmonizer_checkpoint_best.pt "
                "--output-prefix nozscore.brainharmonix.finetuned "
                "--split-index $SLURM_ARRAY_TASK_ID",
            ),
        ]
    else:
        print("Only re-running prediction with existing fine-tuned checkpoints. Make sure to set --rerun-finetune=True if you want to re-run fine-tuning for all splits.")
        slurm_params["time"] = "00:30:00"  # reduce time for prediction-only jobs
        configs = [
            (
                "brainharmonix_finetune_zscore",
                "preventad-extract-brainharmonix "
                "--dataset data/processed/dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow "
                "--harmonizer-ckpt outputs/finetune/brainharmonix/zscore.self-supervised/$SLURM_ARRAY_TASK_ID/harmonizer_checkpoint_best.pt "
                "--output-prefix zscore.brainharmonix.finetuned "
                "--split-index $SLURM_ARRAY_TASK_ID",
            ),
            (
                "brainharmonix_finetune_nozscore",
                "preventad-extract-brainharmonix "
                "--dataset data/processed/dataset-preventad.fmri.NoZscore.gigaconnectome.schaefer400.arrow "
                "--harmonizer-ckpt outputs/finetune/brainharmonix/nozscore.self-supervised/$SLURM_ARRAY_TASK_ID/harmonizer_checkpoint_best.pt "
                "--output-prefix nozscore.brainharmonix.finetuned "
                "--split-index $SLURM_ARRAY_TASK_ID",
            ),
        ]

    for job_name, command in configs:
        submit_job_array(job_name, command, array_range, slurm_params, dry_run=dry_run)
