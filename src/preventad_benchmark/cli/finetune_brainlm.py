"""
Fine tune the embedding layer of BrainLM ViT-MAE model on fMRI data.

https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
import json
from pathlib import Path

from preventad_benchmark.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from preventad_benchmark.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig, Trainer, TrainingArguments

from datasets import load_from_disk, DatasetDict
import numpy as np
import torch
from preventad_benchmark.models.brainlm_mae.utils import timeseires_to_images, collate_fn
from preventad_benchmark.models.brainlm_mae.metrics import MetricsCalculator
from preventad_benchmark.dataset.utils import compute_normalization_params
import argparse
try:
    from preventad_benchmark.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
    replace_vitmae_attn_with_flash_attn()
except ImportError:
    print('not using flash attention')


from preventad_benchmark.config import (
    BRAINLM_IMAGE_COLUMNS,
    BRAINLM_MODEL_ARGUMENTS,
    BRAINLM_TIMESERIES_LENGTH,
)

timeseries_length = BRAINLM_TIMESERIES_LENGTH
image_column_name_kw = BRAINLM_IMAGE_COLUMNS
model_arguments = BRAINLM_MODEL_ARGUMENTS


def main():
    """Fine-tune BrainLM ViT-MAE on PreventAD fMRI data.

    Freezes encoder transformer blocks, leaving patch_embeddings, cls_token,
    and the full decoder trainable. Trains with masked autoencoding on the
    training split, then saves the fine-tuned model.
    """
    parser = argparse.ArgumentParser(description="Fine-tune BrainLM ViT-MAE on fMRI data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Arrow dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/finetune/brainlm",
        help="Output directory for fine-tuned model (default: outputs/finetune/brainlm)",
    )
    parser.add_argument(
        "--image-column-name",
        default="raw_timeseries",
        help="Column name for the image data (default: raw_timeseries)",
    )
    parser.add_argument(
        "--model-params",
        default="111M",
        choices=["111M", "650M"],
        help="BrainLM model size (default: 111M)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained BrainLM model (default: ./models/brainlm/vitmae_{model-params})",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Index of the train/test split to use for finetuning (default: 0)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Compute and apply dataset-level normalization (for non-zscored data)",
    )
    args = parser.parse_args()
    inputs_path = args.dataset
    outputs_path = args.output_dir
    image_column_name = args.image_column_name
    model_params = args.model_params
    model_path = args.model_path or f"./models/brainlm/vitmae_{model_params}"

    fmri_ds = load_from_disk(inputs_path)

    # Load pre-computed train/test split
    split_path = Path("data/processed/train_test_split.json")
    with open(split_path) as f:
        split_ids = json.load(f)

    train_ids = set(split_ids[args.split_index]["train"])

    # Filter to training set only, then split into train/val for finetuning
    # BrainLM participant_ids may have extra suffixes (e.g. _space-..._desc-...);
    # match by checking if any split ID is a prefix of the dataset participant_id
    train_ds = fmri_ds.filter(lambda x: any(x["participant_id"].startswith(tid) for tid in train_ids))
    print(f"Training set: {len(train_ds)} samples (from {len(fmri_ds)} total)")

    # Compute normalization params only for non-zscored data
    norm_params = None
    if args.normalize:
        print("Computing normalization parameters from training set only...")
        norm_params = compute_normalization_params(train_ds)

    timeseires_to_images_kargs = {
        "image_column_name": image_column_name,
        "timeseries_length": timeseries_length, # this is for developmental dataset, full length
        "axis_index": "Y",
        "max_val_to_scale": None,  # max_val_to_scale = 5.6430855  # this is weird.
        "norm_params": norm_params,
    }
    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)

    # Detect sex column name (BrainLM native: 'Sex', gigaconnectome: 'sex')
    sex_col = 'Sex' if 'Sex' in train_ds.column_names else 'sex'
    train_ds = train_ds.class_encode_column(sex_col)
    # 80/20 train/val split within the training set
    train_val = train_ds.train_test_split(
        test_size=0.2,
        stratify_by_column=sex_col
    )
    train_test_dataset = DatasetDict({
        'train': train_val['train'],
        'test': train_val['test']})

    train_test_dataset.set_transform(transform_func)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replace_vitmae_attn_with_flash_attn()

    config = ViTMAEConfig.from_pretrained(model_path)
    config.update(model_arguments)
    config.train_mode = "auto_encode"
    model = ViTMAEForPreTraining.from_pretrained(
            model_path,
            config=config,
        ).to(device)

    # Freeze encoder transformer blocks; keep patch_embeddings, cls_token, and decoder trainable
    # position_embeddings - consider training this if the input sequence length differs from the 
    # pretrained model (e.g. if using shorter timeseries length)

    # Patch Embedding

    # Purpose: Converts raw input into tokens the transformer can process.

    # - Splits the input (image, signal, fMRI timeseries) into fixed-size chunks ("patches")
    # - Projects each patch into a d_model-dimensional vector via a linear layer (or Conv layer)
    # - Answers: "What is in this segment?"

    # input [T, C] → split into patches → linear projection → [N, d_model]

    # ---
    # Position Embedding

    # Purpose: Tells the transformer where each token is in the sequence.

    # - Transformers have no inherent sense of order (attention is permutation-invariant)
    # - Adds a learned or fixed vector to each token based on its position
    # - Answers: "Where in the sequence is this token?"

    # token[i] = patch_embed[i] + pos_embed[i]
    for name, param in model.named_parameters():
        if all(keywords not in name for keywords in["patch_embed", "cls_token", "decoder"]):
            param.requires_grad = False

    # read this: https://medium.com/@kdk199604/fpt-time-series-analysis-powered-by-frozen-pretrained-transformers-7f6d6fc64186

    metrics_calculator = MetricsCalculator()

    training_args = TrainingArguments(
        output_dir=outputs_path,
        remove_unused_columns=False,
        include_for_metrics=['inputs'],
        logging_steps=1,
        num_train_epochs=25,
        learning_rate=1e-05,
        weight_decay=0.01,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
    )
    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
        data_collator=collate_fn,
        compute_metrics=metrics_calculator
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()