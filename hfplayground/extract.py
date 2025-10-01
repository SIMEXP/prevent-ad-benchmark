# cleaned up for using published weights for direct transfer with CLS token
from transformers import ViTMAEConfig
from datasets import load_from_disk, Dataset
import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path

from hfplayground.models.brainlm_mae.utils import timeseires_to_images, get_attention_cls_token, padding_timeseries_For_vitmae
from hfplayground.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
try:
    from hfplayground.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
    replace_vitmae_attn_with_flash_attn()
except ImportError:
    print('not using flash attention')
import click


timeseries_length = 140


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
@click.argument('model-path')
@click.argument('outputs-path')
@click.option(
    '--image-column-name',
    default="robustscaler_timeseries",
    help='Column name for the image data. if you use giga connectome, use robustscaler_timeseries, if you use brainlm, use Subtract_Mean_Divide_Global_STD_Normalized_Recording or Voxelwise_RobustScaler_Normalized_Recording.'
)
def main(inputs_path, image_column_name, outputs_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timeseires_to_images_kargs = {
        "image_column_name": image_column_name,
        "timeseries_length": timeseries_length, # this is for developmental dataset, full length
        "max_val_to_scale": None  # max_val_to_scale = 5.6430855  # this is weird.
    }

    def transform_func(batch):
        return timeseires_to_images(batch, **timeseires_to_images_kargs)

    # load model
    config = ViTMAEConfig.from_pretrained(model_path)
    config.update(model_arguments)
    model = ViTMAEForPreTraining.from_pretrained(
            model_path,
            config=config,
        ).to(device)

    model = model.half()  # half precision data type
    model.eval()
    # multiple train modes (auto-encoder, causal attention, predict last, etc)
    model.config.train_mode = "auto_encode"

    train_ds = load_from_disk(inputs_path)
    train_ds.set_transform(transform_func)

    list_subject_id = []
    list_sex = []
    list_ageclass = []
    list_cls_tokens = []
    list_attn_cls_tokens = []
    all_embeddings = []
    # all_index = []
    with torch.no_grad():
        for recording in tqdm(train_ds, desc="Getting CLS tokens"):
            pixel_values = recording["pixel_values"].unsqueeze(0).half().to(device)
            pixel_values = padding_timeseries_For_vitmae(pixel_values, model.config.image_size)

            encoder_output = model.vit(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

            cls_token = encoder_output.last_hidden_state[:,0,:].detach().cpu().numpy()  # torch.Size([1, 256])? (I got 1, 241)
            embedding = encoder_output.last_hidden_state[:,1:,:].detach().cpu().numpy()

            attn_cls_token = get_attention_cls_token(encoder_output.attentions)
            list_subject_id.append(recording['participant_id'])
            list_sex.append(recording['Sex'])  # need refactor
            list_ageclass.append(recording['Candidate_Age'])  # need refactor
            list_attn_cls_tokens.append(attn_cls_token)
            list_cls_tokens.append(cls_token)
            all_embeddings.append(embedding)

    # pooling or
    cls_embeds = np.concatenate(list_cls_tokens, axis=0)
    all_mean_embeddings = [e.mean(axis=1) for e in all_embeddings]
    all_mean_embeddings = np.concatenate(all_mean_embeddings, axis=0)
    all_maxpool_embeddings = [e.max(axis=1) for e in all_embeddings]
    all_maxpool_embeddings = np.concatenate(all_maxpool_embeddings, axis=0)

    # save all padded recording
    all_recordings = []
    for _, batch in enumerate(tqdm(train_ds)):
        signal = batch["pixel_values"]  # (1, 3, num_parcel, timeseries_length)
        recording = signal.flatten(start_dim=1)
        recording = np.array(recording, dtype=np.float32)
        all_recordings.append(recording)

    results = {
        'participant_id': list_subject_id,
        'Sex': list_sex,   # need refactor
        'Candidate_Age': list_ageclass,   # need refactor
        'cls_token': list_attn_cls_tokens,
        'cls_embedding': cls_embeds,
        'mean_embedding': all_mean_embeddings,
        'max_embedding': all_maxpool_embeddings,
        'padded_recording': all_recordings
    }
    arrow_results = Dataset.from_dict(results)
    arrow_results.save_to_disk(outputs_path)


if __name__ == "__main__":
    main()