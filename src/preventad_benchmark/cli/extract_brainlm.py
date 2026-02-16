# cleaned up for using published weights for feature extraction with CLS token
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import ViTMAEConfig
from datasets import load_from_disk

from preventad_benchmark.models.brainlm_mae.utils import (
    timeseires_to_images, get_attention_cls_token, padding_timeseries_For_vitmae
)
from preventad_benchmark.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
try:
    from preventad_benchmark.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
    replace_vitmae_attn_with_flash_attn()
except ImportError:
    print('not using flash attention')

from preventad_benchmark.dataset.utils import compute_normalization_params
from preventad_benchmark.evaluation import run_foundation_model_experiment
from preventad_benchmark.evaluation.targets import load_prediction_targets

from preventad_benchmark.config import BRAINLM_MODEL_ARGUMENTS, BRAINLM_TIMESERIES_LENGTH

TIMESERIES_LENGTH = BRAINLM_TIMESERIES_LENGTH
MODEL_ARGUMENTS = BRAINLM_MODEL_ARGUMENTS


def _extract_embeddings(dataset, model, device, image_column_name, norm_params=None):
    """Extract embeddings from a dataset using the BrainLM model.

    Returns dict with cls_token, cls_embedding, mean_embedding, max_embedding arrays.
    """
    timeseires_to_images_kargs = {
        "image_column_name": image_column_name,
        "timeseries_length": TIMESERIES_LENGTH,
        "max_val_to_scale": None,
        "norm_params": norm_params,
    }

    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)

    dataset.set_transform(transform_func)

    list_cls_tokens = []
    list_attn_cls_tokens = []
    all_embeddings = []
    with torch.no_grad():
        for recording in tqdm(dataset, desc="Getting CLS tokens"):
            pixel_values = recording["pixel_values"].unsqueeze(0).half().to(device)
            pixel_values = padding_timeseries_For_vitmae(pixel_values, model.config.image_size)

            encoder_output = model.vit(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            cls_token = encoder_output.last_hidden_state[:,0,:].detach().cpu().numpy()
            embedding = encoder_output.last_hidden_state[:,1:,:].detach().cpu().numpy()

            attn_cls_token = get_attention_cls_token(encoder_output.attentions)
            list_attn_cls_tokens.append(attn_cls_token)
            list_cls_tokens.append(cls_token)
            all_embeddings.append(embedding)

    cls_embeds = np.concatenate(list_cls_tokens, axis=0)
    all_mean_embeddings = np.concatenate([e.mean(axis=1) for e in all_embeddings], axis=0)
    all_maxpool_embeddings = np.concatenate([e.max(axis=1) for e in all_embeddings], axis=0)

    return dict(
        cls_token=np.concatenate(list_attn_cls_tokens, axis=0).squeeze(),
        cls_embedding=cls_embeds,
        mean_embedding=all_mean_embeddings,
        max_embedding=all_maxpool_embeddings,
    )


def main():
    """Extract BrainLM embeddings via checkpoint."""
    parser = argparse.ArgumentParser(description="Extract BrainLM embeddings via checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Arrow dataset",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained BrainLM model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/downstreams/brainlm",
        help=(
            "Output directory for downstream experiment results "
            "(default: outputs/downstreams/brainlm)"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="brainlm",
        help="Prefix for output files (default: brainlm)",
    )
    parser.add_argument(
        "--image-column-name",
        default="raw_timeseries",
        help="Column name for the image data (default: raw_timeseries)",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Index of the train/test split (default: 0)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Compute and apply dataset-level normalization (for non-zscored data)",
    )
    args = parser.parse_args()
    inputs_path = args.dataset
    model_path = args.model_path
    output_dir = Path(args.output_dir)
    image_column_name = args.image_column_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = output_dir / f"{args.output_prefix}.split{args.split_index}.tsv"
    if output_path.exists():
        print(f"{output_path} exists, skip")
        return

    # load model
    config = ViTMAEConfig.from_pretrained(model_path)
    config.update(MODEL_ARGUMENTS)
    model = ViTMAEForPreTraining.from_pretrained(
            model_path,
            config=config,
        ).to(device)

    model = model.half()  # half precision data type
    model.eval()
    # multiple train modes (auto-encoder, causal attention, predict last, etc)
    model.config.train_mode = "auto_encode"

    fmri_ds = load_from_disk(inputs_path)
    split_path = Path("data/processed/train_test_split.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split_ids = json.load(f)

    # Filter train set
    train_ids = set(split_ids[args.split_index]["train"])
    # BrainLM participant_ids may have extra suffixes (e.g. _space-..._desc-...);
    # match by checking if any split ID is a prefix of the dataset participant_id
    train_ds = fmri_ds.filter(
        lambda x: any(x["participant_id"].startswith(tid) for tid in train_ids)
    )
    print(f"Training set: {len(train_ds)} samples (from {len(fmri_ds)} total)")

    # Compute normalization params only for non-zscored data
    train_norm_params = None
    if args.normalize:
        train_norm_params = compute_normalization_params(train_ds)

    print("\nExtracting train embeddings...")
    train_features = _extract_embeddings(
        train_ds, model, device, image_column_name, train_norm_params
    )
    train_labels = load_prediction_targets(participant_ids=train_ids)

    # Filter test set
    test_ids = set(split_ids[args.split_index]["test"])
    test_ds = fmri_ds.filter(lambda x: any(x["participant_id"].startswith(tid) for tid in test_ids))
    print(f"Test set: {len(test_ds)} samples (from {len(fmri_ds)} total)")

    # Compute normalization params only for non-zscored data
    test_norm_params = None
    if args.normalize:
        test_norm_params = compute_normalization_params(test_ds)

    print("\nExtracting test embeddings...")
    test_features = _extract_embeddings(test_ds, model, device, image_column_name, test_norm_params)
    test_labels = load_prediction_targets(participant_ids=test_ids)

    test_split_results = []
    for feat_name, train_feat in train_features.items():
        print(f"Running experiments with {feat_name} features.")
        feat_results = run_foundation_model_experiment(
            train_feat,
            train_labels,
            test_features[feat_name],
            test_labels,
            feat_name,
        )
        feat_results["Features"] = feat_name
        test_split_results.append(feat_results)

    test_split_results = pd.concat(test_split_results).reset_index(drop=True)
    test_split_results['split'] = args.split_index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_split_results.to_csv(output_path, sep='\t')


if __name__ == "__main__":
    main()
