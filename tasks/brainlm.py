"""Model training and feature extraction tasks.

This module contains invoke tasks for fine-tuning foundation models
and extracting embeddings for downstream evaluation.
"""
from pathlib import Path

import invoke


# Default paths
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs")


@invoke.task(
    help={
        "input-path": "Path to Arrow dataset (default: BrainLM preprocessed)",
        "output-path": "Output directory for finetuned model",
        "model-params": "Model size: 111M or 650M (default: 111M)",
        "image-column": "Column name for timeseries data",
    }
)
def finetune_brainlm(
    c,
    input_path=None,
    output_path=None,
    model_params="111M",
    image_column="Subtract_Mean_Divide_Global_STD_Normalized_Recording",
):
    """Fine-tune BrainLM ViT-MAE model on PreventAD data.

    Freezes all parameters except patch_embed and cls_token layers.
    Trains using masked autoencoder reconstruction objective.

    Example:
        inv models.finetune-brainlm
        inv models.finetune-brainlm --model-params=650M
        inv models.finetune-brainlm --input-path=./data/processed/custom.arrow
    """
    input_path = input_path or str(DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.brainlm.a424.arrow")
    output_path = output_path or str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm/nozscore.{model_params}.brainlm.selfsupervised")

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm/nozscore.{model_params}.brainlm.selfsupervised.embeddings")
    extract_cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {output_path} --output-dir {extract_path} --image-column-name {image_column}"
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "input-path": "Path to Arrow dataset (default: GigaConnectome preprocessed)",
        "output-path": "Output directory for finetuned model",
        "model-params": "Model size: 111M or 650M (default: 111M)",
        "image-column": "Column name for timeseries data",
    }
)
def finetune_brainlm_gigaconnectome(
    c,
    input_path=None,
    output_path=None,
    model_params="111M",
    image_column="robustscaler_timeseries",
):
    """Fine-tune BrainLM on GigaConnectome-preprocessed data.

    Uses RobustScaler normalized timeseries from GigaConnectome pipeline.

    Example:
        inv models.finetune-brainlm-gigaconnectome
        inv models.finetune-brainlm-gigaconnectome --model-params=650M
    """
    input_path = input_path or str(DEFAULT_DATA_DIR / "dataset-preventad.gigaconnectome.arrow")
    output_path = output_path or str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm-{model_params}-gigaconnectome")

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm.vitmae_{model_params}.finetuned.gigaconnectome")
    extract_cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {output_path} --output-dir {extract_path} --image-column-name {image_column}"
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "model-size": "Model size to use: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, gigaconnectome, or all (default: all)",
    }
)
def direct_transfer(c, model_size="all", preprocessing="all"):
    """Extract features using pre-trained BrainLM models (no fine-tuning).

    Direct transfer evaluation - uses published model weights without
    any adaptation to the target dataset.

    Example:
        inv models.direct-transfer
        inv models.direct-transfer --model-size=650M
        inv models.direct-transfer --preprocessing=gigaconnectome
    """
    configs = []

    # Build list of configurations to run
    sizes = ["111M", "650M"] if model_size == "all" else [model_size]
    preps = ["brainlm", "gigaconnectome"] if preprocessing == "all" else [preprocessing]

    for size in sizes:
        for prep in preps:
            if prep == "brainlm":
                input_path = str(DEFAULT_DATA_DIR / "dataset-preventad.brainlm.arrow")
                image_column = "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                output_suffix = "og"
            else:
                input_path = str(DEFAULT_DATA_DIR / "dataset-preventad.gigaconnectome.arrow")
                image_column = "robustscaler_timeseries"
                output_suffix = "gigaconnectome"

            model_path = str(DEFAULT_MODEL_DIR / f"brainlm/vitmae_{size}")
            output_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm.vitmae_{size}.direct_transfer.{output_suffix}")

            cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {model_path} --output-dir {output_path} --image-column-name {image_column}"
            print(f"Running: {cmd}")
            c.run(cmd)
