# cleaned up for using published weights for direct transfer with CLS token

from transformers import ViTMAEConfig
from datasets import load_from_disk, Dataset
import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path

from preventad_benchmark.models.brainlm_mae.utils import timeseires_to_images, get_attention_cls_token, padding_timeseries_For_vitmae
from preventad_benchmark.models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
try:
    from preventad_benchmark.models.brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
    replace_vitmae_attn_with_flash_attn()
except ImportError:
    print('not using flash attention')
import argparse

from preventad_benchmark.config import BRAINLM_MODEL_ARGUMENTS, BRAINLM_TIMESERIES_LENGTH

timeseries_length = BRAINLM_TIMESERIES_LENGTH
model_arguments = BRAINLM_MODEL_ARGUMENTS


def main():
    parser = argparse.ArgumentParser(description="Extract BrainLM embeddings via direct transfer")
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
        default="outputs/embeddings/brainlm",
        help="Output path for Arrow dataset with embeddings (default: outputs/embeddings/brainlm)",
    )
    parser.add_argument(
        "--image-column-name",
        default="robustscaler_timeseries",
        help="Column name for the image data (default: robustscaler_timeseries)",
    )
    args = parser.parse_args()
    inputs_path = args.dataset
    model_path = args.model_path
    outputs_path = args.output_dir
    image_column_name = args.image_column_name
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