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
        "model-params": "Model size: 111M or 650M (default: 650M)",
        "image-column": "Column name for timeseries data",
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def finetune_brainlm(
    c,
    input_path=None,
    output_path=None,
    model_params="650M",
    image_column="Subtract_Mean_Divide_Global_STD_Normalized_Recording",
    split_index=0,
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

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params} --split-index {split_index}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model
    extract_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")
    extract_prefix = f"nozscore_brainlm.brainlm{model_params}.finetuned"
    extract_cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {output_path} --output-dir {extract_dir} --output-prefix {extract_prefix} --image-column-name {image_column} --split-index {split_index}"
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "input-path": "Path to Arrow dataset (default: GigaConnectome preprocessed)",
        "output-path": "Output directory for finetuned model",
        "model-params": "Model size: 111M or 650M (default: 650M)",
        "image-column": "Column name for timeseries data",
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def finetune_brainlm_gigaconnectome(
    c,
    input_path=None,
    output_path=None,
    model_params="650M",
    image_column="raw_timeseries",
    split_index=0,
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

    cmd = f"preventad-finetune-brainlm --dataset {input_path} --output-dir {output_path} --image-column-name {image_column} --model-params {model_params} --norm-params {norm_params} --split-index {split_index}"
    print(f"Running: {cmd}")
    c.run(cmd)

    # Extract features from finetuned model using training-set norm_params
    train_norm_params = str(Path(output_path) / "train_norm_params.npz")
    extract_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")
    extract_prefix = f"zscore_gigaconnectome.brainlm{model_params}.finetuned"
    extract_cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {output_path} --output-dir {extract_dir} --output-prefix {extract_prefix} --image-column-name {image_column} --norm-params {train_norm_params} --split-index {split_index}"
    print(f"Running: {extract_cmd}")
    c.run(extract_cmd)


@invoke.task(
    help={
        "model-size": "Model size to use: 111M, 650M, or all (default: all)",
        "preprocessing": "Preprocessing type: brainlm, gigaconnectome, or all (default: all)",
        "split-index": "Index of the train/test split (default: 0)",
    }
)
def extract_features(c, model_size="650M", preprocessing="all", split_index=0):
    """Extract features using pre-trained BrainLM models (no fine-tuning).

    Direct transfer evaluation - uses published model weights without
    any adaptation to the target dataset.

    Example:
        inv brainlm.extract-representation
        inv brainlm.extract-representation --model-size=650M
        inv brainlm.extract-representation --preprocessing=gigaconnectome
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
            elif prep == "brainlm_z":
                input_path = str(BRAINLM_DATASET_Z)
                image_column = "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                output_suffix = "zscore_brainlm"
            elif prep =="gigaconnectome_z":
                input_path = str(GIGACONNECTOME_DATASET_Z)
                image_column = "raw_timeseries"
                output_suffix = "zscore_gigaconnectome"
            else:
                input_path = str(GIGACONNECTOME_DATASET)
                image_column = "raw_timeseries"
                output_suffix = "nozscore.gigaconnectome"

            model_path = str(DEFAULT_MODEL_DIR / f"brainlm/vitmae_{size}")
            output_dir = str(DEFAULT_OUTPUT_DIR / "downstreams/brainlm")
            output_prefix = f"{output_suffix}.brainlm{size}"

            cmd = f"preventad-extract-brainlm --dataset {input_path} --model-path {model_path} --output-dir {output_dir} --output-prefix {output_prefix} --image-column-name {image_column} --split-index {split_index}"
            print(f"Running: {cmd}")
            c.run(cmd)
