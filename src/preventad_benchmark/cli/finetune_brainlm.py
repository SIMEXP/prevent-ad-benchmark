"""
Fine tune the embedding layer of BrainLM ViT-MAE model on fMRI data.

https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
from preventad_benchmark.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from preventad_benchmark.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig, Trainer, TrainingArguments

from datasets import load_from_disk, DatasetDict
import numpy as np
import torch
from preventad_benchmark.models.brainlm_mae.utils import timeseires_to_images, collate_fn
from preventad_benchmark.models.brainlm_mae.metrics import MetricsCalculator
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
        "--norm-params",
        type=str,
        default=None,
        help="Path to .norm_params.npz file for dataset-level normalization (required for gigaconnectome raw_timeseries)",
    )
    args = parser.parse_args()
    inputs_path = args.dataset
    outputs_path = args.output_dir
    image_column_name = args.image_column_name
    model_params = args.model_params
    model_path = args.model_path or f"./models/brainlm/vitmae_{model_params}"

    norm_params = None
    if args.norm_params:
        norm_params = dict(np.load(args.norm_params))

    timeseires_to_images_kargs = {
        "image_column_name": image_column_name,
        "timeseries_length": timeseries_length, # this is for developmental dataset, full length
        "axis_index": "Y",
        "max_val_to_scale": None,  # max_val_to_scale = 5.6430855  # this is weird.
        "norm_params": norm_params,
    }
    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)

    fmri_ds = load_from_disk(inputs_path)
    # Detect sex column name (BrainLM native: 'Sex', gigaconnectome: 'sex')
    sex_col = 'Sex' if 'Sex' in fmri_ds.column_names else 'sex'
    fmri_ds = fmri_ds.class_encode_column(sex_col)
    # 80% train, 20% test
    train_test = fmri_ds.train_test_split(
        train_size=0.8,
        stratify_by_column=sex_col
    )
    # gather everyone if you want to have a single DatasetDict
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']})

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

    # Freeze the model parameters aside from the embedding layer
    for name, param in model.named_parameters():
        if all(keywords not in name for keywords in["patch_embed", "cls_token"]):
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