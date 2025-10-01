"""
Fine tune the embedding layer of BrainLM ViT-MAE model on fMRI data.

https://github.com/SalvoCalcagno/quantformer2024/blob/98286c88d79cf562966545b5509e93e611ae049b/src/trainers/trainer_brainlm.py#L55
https://github.com/Shef-AIRE/FMM_TC/blob/main/FMM_TC-tutorial.ipynb
https://github.com/wenhui0206/MeTSK/blob/main/meta_learning.py#L8
BrainLM/continue_train_same_wandb.py
"""
from hfplayground.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
from hfplayground.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from transformers import ViTMAEConfig, Trainer, TrainingArguments

from datasets import load_from_disk, DatasetDict
import torch
from hfplayground.models.brainlm_mae.utils import timeseires_to_images, collate_fn
from hfplayground.models.brainlm_mae.metrics import MetricsCalculator
import click
try:
    from hfplayground.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
    replace_vitmae_attn_with_flash_attn()
except ImportError:
    print('not using flash attention')


timeseries_length = 140

image_column_name_kw = {
    "gigaconnectome": "robustscaler_timeseries",
    "brainlm": "Subtract_Mean_Divide_Global_STD_Normalized_Recording"  # this works
}

model_arguments = {  # BrainLM/train.py::ModelArguments
    "mask_ratio": 0.75,  # The ratio of the number of masked tokens per brain region.
    "timepoint_patching_size": 20,  #Length of moving window of timepoints from each brain regions signal for 1 sample.
    "num_timepoints_per_voxel": timeseries_length,  # Number of timepoints for each brain region given in 1 sample input to model.  Must be divisible by timepoint_patching_size.
    "hidden_dropout_prob": 0.0,  # Dropout probability for layer activations in CellLM
    "attention_probs_dropout_prob": 0.0,  # Dropout probability for attention coefficients in CellLM.
    "output_attentions": True,
}

@click.command()
@click.argument('inputs-path')
@click.argument('outputs-path')
@click.option(
    '--image-column-name',
    default="robustscaler_timeseries",
    help='Column name for the image data. if you use giga connectome, use robustscaler_timeseries, if you use brainlm, use Subtract_Mean_Divide_Global_STD_Normalized_Recording or Voxelwise_RobustScaler_Normalized_Recording.'
)
@click.option('--model-params', default="111M", type=click.Choice(["111M", "650M"]), help='BrainLM model parameters to use.')
def main(inputs_path, image_column_name, outputs_path, model_params):
    timeseires_to_images_kargs = {
        "image_column_name": image_column_name,
        "timeseries_length": timeseries_length, # this is for developmental dataset, full length
        "axis_index": "Y",
        "max_val_to_scale": None  # max_val_to_scale = 5.6430855  # this is weird.
    }
    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)

    fmri_ds = load_from_disk(inputs_path).class_encode_column("Sex")
    fmri_ds = load_from_disk(inputs_path)
    # 80% train, 20% test
    train_test = fmri_ds.train_test_split(
        train_size=0.8,
        stratify_by_column='Sex'
    )
    # gather everyone if you want to have a single DatasetDict
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']})

    train_test_dataset.set_transform(transform_func)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replace_vitmae_attn_with_flash_attn()

    config = ViTMAEConfig.from_pretrained(f"./models/brainlm/vitmae_{model_params}")
    config.update(model_arguments)
    config.train_mode = "auto_encode"
    model = ViTMAEForPreTraining.from_pretrained(
            f"./models/brainlm/vitmae_{model_params}",
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
        num_train_epochs=30,
        learning_rate=5e-05,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
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