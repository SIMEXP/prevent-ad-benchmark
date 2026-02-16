"""Model training and feature extraction tasks for BrainLM.

This module contains invoke tasks for fine-tuning BrainLM foundation models
and extracting embeddings for downstream evaluation.
"""
from pathlib import Path

import invoke

from .slurm import load_slurm_config, submit_job_array


# Default paths
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs")

# Input Arrow datasets (matching prepare.py output naming)
EXTRACTOION_PARAM = {
    "brainlm":{
        "input_path": DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.brainlm.a424.arrow",
        "image_column": "Subtract_Mean_Divide_Global_STD_Normalized_Recording",
        "output_suffix": "nozscore_brainlm",
        "normalize": False,  # precomuted and integrated
    },
    "brainlm_z":{
        "input_path": DEFAULT_DATA_DIR / "dataset-preventad.fmri.zscored.brainlm.a424.arrow",
        "image_column": "Subtract_Mean_Divide_Global_STD_Normalized_Recording",
        "output_suffix": "zscore_brainlm",
        "normalize": False,  # precomuted and integrated
    },
    "gigaconnectome_z":{
        "input_path": DEFAULT_DATA_DIR / "dataset-preventad.fmri.zscored.gigaconnectome.a424.arrow",
        "image_column": "raw_timeseries",
        "output_suffix": "zscore_gigaconnectome",
        "normalize": False,   # everything is already centred at 0
    },
    "gigaconnectome":{
        "input_path": DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.gigaconnectome.a424.arrow",
        "image_column": "raw_timeseries",
        "output_suffix": "nozscore_gigaconnectome",
        "normalize": True,
    },
}


@invoke.task(
    help={
        "preprocessing": "Preprocessing type: brainlm, brainlm_z, gigaconnectome, or gigaconnectome_z (default: brainlm)",
        "model-params": "Model size: 111M or 650M (default: 650M)",
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def finetune(
    c,
    preprocessing="brainlm",
    model_params="650M",
    split_index=0,
):
    """Fine-tune BrainLM ViT-MAE model and extract finetuned features.

    Freezes all parameters except patch_embed and cls_token layers.
    Trains using masked autoencoder reconstruction objective.

    Example:
        inv brainlm.finetune
        inv brainlm.finetune --preprocessing=gigaconnectome --model-params=650M
    """
    cfg = EXTRACTOION_PARAM[preprocessing]
    input_path = str(cfg["input_path"])
    image_column = cfg["image_column"]
    output_suffix = cfg["output_suffix"]
    output_path = str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm/{output_suffix}.{model_params}.selfsupervised/{split_index}")

    normalize_flag = " --normalize" if cfg.get("normalize") else ""
    cmd = (
        f"preventad-finetune-brainlm --dataset {input_path} "
        f"--output-dir {output_path} --image-column-name {image_column} "
        f"--model-params {model_params} --split-index {split_index}"
        f"{normalize_flag}"
    )
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")
    extract_prefix = f"{output_suffix}.brainlm{model_params}.finetuned"
    extract_cmd = (
        f"preventad-extract-brainlm --dataset {input_path} "
        f"--model-path {output_path} --output-dir {extract_dir} "
        f"--output-prefix {extract_prefix} --image-column-name {image_column} "
        f"--split-index {split_index}{normalize_flag}"
    )
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "model-size": "Model size: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, brainlm_z, gigaconnectome, gigaconnectome_z, or all (default: all)",
        "n-splits": "Number of train/test splits (default: 20)",
        "dry-run": "Print generated scripts without submitting (default: False)",
    }
)
def submit_finetune(c, model_size="all", preprocessing="all", n_splits=20, dry_run=False):
    """Submit SLURM job arrays for BrainLM fine-tuning + finetuned feature extraction.

    Each array task fine-tunes BrainLM on one train/test split,
    then extracts embeddings from the finetuned model.

    Example:
        inv brainlm.submit-finetune
        inv brainlm.submit-finetune --model-size=650M --preprocessing=brainlm
        inv brainlm.submit-finetune --dry-run
    """
    slurm_params = load_slurm_config("finetune_brainlm")
    array_range = f"0-{n_splits - 1}"

    sizes = ["111M", "650M"] if model_size == "all" else [model_size]
    preps = ["brainlm", "brainlm_z", "gigaconnectome", "gigaconnectome_z"] if preprocessing == "all" else [preprocessing]

    extract_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")

    for size in sizes:
        for prep in preps:
            cfg = EXTRACTOION_PARAM[prep]
            input_path = str(cfg["input_path"])
            output_suffix = cfg["output_suffix"]
            finetune_dir = str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm/{output_suffix}.{size}.selfsupervised")
            job_name = f"brainlm_finetune_{output_suffix}_{size}".replace(".", "_")

            normalize_flag = " --normalize" if cfg.get("normalize") else ""
            finetune_cmd = (
                f"preventad-finetune-brainlm "
                f"--dataset {input_path} "
                f"--output-dir {finetune_dir}/split$SLURM_ARRAY_TASK_ID "
                f"--image-column-name {cfg['image_column']} "
                f"--model-params {size} "
                f"--split-index $SLURM_ARRAY_TASK_ID"
                f"{normalize_flag}"
            )
            extract_prefix = f"{output_suffix}.brainlm{size}.finetuned"
            extract_cmd = (
                f"preventad-extract-brainlm "
                f"--dataset {input_path} "
                f"--model-path {finetune_dir}/split$SLURM_ARRAY_TASK_ID "
                f"--output-dir {extract_dir} "
                f"--output-prefix {extract_prefix} "
                f"--image-column-name {cfg['image_column']} "
                f"--split-index $SLURM_ARRAY_TASK_ID"
                f"{normalize_flag}"
            )
            command = f"{finetune_cmd} && \\\n{extract_cmd}"

            submit_job_array(job_name, command, array_range, slurm_params, dry_run=dry_run)


@invoke.task(
    help={
        "model-size": "Model size to use: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, gigaconnectome, or all (default: all)",
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def evaluate(c, model_size="650M", preprocessing="all", split_index=0):
    """Run pretrained BrainLM prediction pipeline (no fine-tuning).

    Feature extraction evaluation - uses published model weights without
    any adaptation to the target dataset.

    Example:
        inv brainlm.evaluate
        inv brainlm.evaluate --model-size=650M
        inv brainlm.evaluate --preprocessing=gigaconnectome
    """
    # Build list of configurations to run
    sizes = ["111M", "650M"] if model_size == "all" else [model_size]
    preps = ["brainlm", "brainlm_z", "gigaconnectome", "gigaconnectome_z"] if preprocessing == "all" else [preprocessing]

    for size in sizes:
        for prep in preps:
            params = EXTRACTOION_PARAM[prep]

            model_path = str(DEFAULT_MODEL_DIR / f"brainlm/vitmae_{size}")
            output_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")
            output_prefix = f"{params['output_suffix']}.brainlm{size}"

            normalize_flag = " --normalize" if params.get("normalize") else ""
            cmd = f"preventad-extract-brainlm --dataset {params['input_path']} --model-path {model_path} --output-dir {output_dir} --output-prefix {output_prefix} --image-column-name {params['image_column']} --split-index {split_index}{normalize_flag}"
            print(f"Running: {cmd}")
            c.run(cmd)


@invoke.task(
    help={
        "model-size": "Model size: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, brainlm_z, gigaconnectome, gigaconnectome_z, or all (default: all)",
        "n-splits": "Number of train/test splits (default: 20)",
        "dry-run": "Print generated scripts without submitting (default: False)",
    }
)
def submit_evaluate(c, model_size="all", preprocessing="all", n_splits=20, dry_run=False):
    """Submit SLURM job arrays for pretrained BrainLM prediction (no fine-tuning).

    Example:
        inv brainlm.submit-evaluate
        inv brainlm.submit-evaluate --model-size=650M --preprocessing=brainlm
        inv brainlm.submit-evaluate --dry-run
    """
    slurm_params = load_slurm_config("extract_brainlm")
    array_range = f"4-{n_splits - 1}"

    sizes = ["111M", "650M"] if model_size == "all" else [model_size]
    preps = ["brainlm", "brainlm_z", "gigaconnectome", "gigaconnectome_z"] if preprocessing == "all" else [preprocessing]

    output_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")

    for size in sizes:
        for prep in preps:
            cfg = EXTRACTOION_PARAM[prep]
            model_path = str(DEFAULT_MODEL_DIR / f"brainlm/vitmae_{size}")
            output_prefix = f"{cfg['output_suffix']}.brainlm{size}"
            job_name = f"brainlm_extract_{cfg['output_suffix']}_{size}".replace(".", "_")

            normalize_flag = " --normalize" if cfg.get("normalize") else ""
            command = (
                f"preventad-extract-brainlm "
                f"--dataset {cfg['input_path']} "
                f"--model-path {model_path} "
                f"--output-dir {output_dir} "
                f"--output-prefix {output_prefix} "
                f"--image-column-name {cfg['image_column']} "
                f"--split-index $SLURM_ARRAY_TASK_ID"
                f"{normalize_flag}"
            )

            submit_job_array(job_name, command, array_range, slurm_params, dry_run=dry_run)
