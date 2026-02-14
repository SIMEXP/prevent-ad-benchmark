"""Model training and feature extraction tasks for BrainLM.

This module contains invoke tasks for fine-tuning BrainLM foundation models
and extracting embeddings for downstream evaluation.
"""
from pathlib import Path

import invoke


# Default paths
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs")

# Input Arrow datasets (matching prepare.py output naming)
BRAINLM_DATASET = DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.brainlm.a424.arrow"
BRAINLM_DATASET_Z = DEFAULT_DATA_DIR / "dataset-preventad.fmri.zscored.brainlm.a424.arrow"
GIGACONNECTOME_DATASET = DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.gigaconnectome.a424.arrow"
GIGACONNECTOME_NORM_PARAMS = DEFAULT_DATA_DIR / "dataset-preventad.fmri.NoZscore.gigaconnectome.a424.norm_params.npz"
GIGACONNECTOME_DATASET_Z = DEFAULT_DATA_DIR / "dataset-preventad.fmri.zscored.gigaconnectome.a424.arrow"
GIGACONNECTOME_NORM_PARAMS_Z = DEFAULT_DATA_DIR / "dataset-preventad.fmri.zscored.gigaconnectome.a424.norm_params.npz"


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
    """Fine-tune BrainLM ViT-MAE model on BrainLM-preprocessed PreventAD data.

    Freezes all parameters except patch_embed and cls_token layers.
    Trains using masked autoencoder reconstruction objective.

    Example:
        inv brainlm.finetune-brainlm
        inv brainlm.finetune-brainlm --model-params=650M
    """
    input_path = input_path or str(BRAINLM_DATASET)
    output_path = output_path or str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm/nozscore.brainlm.{model_params}.selfsupervised")

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm/nozscore_brainlm.brainlm{model_params}.finetuned.embeddings.npz")
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
    image_column="raw_timeseries",
):
    """Fine-tune BrainLM on GigaConnectome-preprocessed data.

    Uses raw timeseries from GigaConnectome pipeline with dataset-level normalization.

    Example:
        inv brainlm.finetune-brainlm-gigaconnectome
        inv brainlm.finetune-brainlm-gigaconnectome --model-params=650M
    """
    input_path = input_path or str(GIGACONNECTOME_DATASET)
    norm_params = str(GIGACONNECTOME_NORM_PARAMS)
    output_path = output_path or str(DEFAULT_OUTPUT_DIR / f"finetune/brainlm/zscore.gigaconnectome.{model_params}.selfsupervised")

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params} --norm-params {norm_params}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm/zscore_gigaconnectome.brainlm{model_params}.finetuned.embeddings.npz")
    extract_cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {output_path} --output-dir {extract_path} --image-column-name {image_column} --norm-params {norm_params}"
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "model-size": "Model size to use: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, gigaconnectome, or all (default: all)",
    }
)
def extract_representation(c, model_size="all", preprocessing="all"):
    """Extract features using pre-trained BrainLM models (no fine-tuning).

    Direct transfer evaluation - uses published model weights without
    any adaptation to the target dataset.

    Example:
        inv brainlm.direct-transfer
        inv brainlm.direct-transfer --model-size=650M
        inv brainlm.direct-transfer --preprocessing=gigaconnectome
    """
    # Build list of configurations to run
    sizes = ["111M", "650M"] if model_size == "all" else [model_size]
    preps = ["brainlm", "brainlm_z", "gigaconnectome", "gigaconnectome_z"] if preprocessing == "all" else [preprocessing]

    for size in sizes:
        for prep in preps:
            if prep == "brainlm":
                input_path = str(BRAINLM_DATASET)
                image_column = "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                output_suffix = "nozscore_brainlm"
                norm_flag = ""
            elif prep == "brainlm_z":
                input_path = str(BRAINLM_DATASET_Z)
                image_column = "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                output_suffix = "zscore_brainlm"
                norm_flag = ""
            elif prep =="gigaconnectome_z":
                input_path = str(GIGACONNECTOME_DATASET_Z)
                image_column = "raw_timeseries"
                output_suffix = "zscore_gigaconnectome"
                norm_flag = f" --norm-params {GIGACONNECTOME_NORM_PARAMS_Z}"
            else:
                input_path = str(GIGACONNECTOME_DATASET)
                image_column = "raw_timeseries"
                output_suffix = "nozscore.gigaconnectome"
                norm_flag = f" --norm-params {GIGACONNECTOME_NORM_PARAMS}"

            model_path = str(DEFAULT_MODEL_DIR / f"brainlm/vitmae_{size}")
            output_path = str(DEFAULT_OUTPUT_DIR / f"embeddings/brainlm/{output_suffix}.brainlm{size}.embeddings.npz")

            cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {model_path} --output-dir {output_path} --image-column-name {image_column}{norm_flag}"
            print(f"Running: {cmd}")
            c.run(cmd)
