from pathlib import Path
from random import randint
from datasets import load_from_disk
import torch
import numpy as np
import torch.nn.functional as F


COORD_DS_PATH = Path("resource/brainlm/atlases/brainregion_coordinates.arrow")

def timeseires_to_images(examples, image_column_name, timeseries_length, axis_index="Y", max_val_to_scale=None):
    """
    Preprocess timeseires as if they are pixel value of images with three
    colour channels, and order the timeseires by an given order along a
    certain axis.
    This fuction should be passed to `set_transform` of an arrow dataset.

    Args:
        examples : one arrow dataset instnace?
        image_column_name (str): a key
        timeseries_length (int): length of the full time seire.
        axis_index (str): default "Y". It's exposed as I was unsure if brainlm used X or Y (conflicts between comments and variable name in the original source code.)
        max_val_to_scale (float, None):an weird scaling value (5.6430855) from
            the original code. Default to None.

    Notes:
        - https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Dataset.set_transform
    """
    reorder_ids_by_coord = order_parcel_by_axis(COORD_DS_PATH, axis_index)
    fmri_images_list, labels = [], []
    for idx in range(len(examples[image_column_name])):
        signal_window = torch.tensor(
            examples[image_column_name][idx],
            dtype=torch.float32
        ) # [timepoints, num_parcel]

        # Choose random starting index, take window of timeseries_length points for each region
        start_idx = randint(0, signal_window.shape[0] - timeseries_length)
        end_idx = start_idx + timeseries_length
        signal_window = signal_window[start_idx: end_idx, :]
        signal_window = torch.movedim(signal_window, 0, 1)  # --> [num_parcel, timeseries_length]

        # reorder voxels according to coordinate
        signal_window = signal_window[reorder_ids_by_coord, :]
        if max_val_to_scale:
            signal_window = signal_window / max_val_to_scale

        # Repeat tensor for 3 channels (R,G,B)
        signal_window = signal_window.unsqueeze(0).repeat(3, 1, 1)

        fmri_images_list.append(signal_window)
        labels.append(reorder_ids_by_coord.tolist())

    examples["pixel_values"] = fmri_images_list  # No transformation or resizing; model will do padding
    examples["labels"] = labels
    return examples


def order_parcel_by_axis(coords_ds_path, axis_index="Y"):
    """
    Reorder parcel by axis. It's extremely unclear what did they do in the
    source code. The code says Y-axis but the code says X-axis.
    Before I actually figure it out, this is a function to allow me to iterate
    and find the correct one.

    Args:
        coords_ds_path (Path, str): path to the arrow dataset of
        parcel (x, y, z) coordinate.
        axis_index (str): axis to sort data with. Options: "X", "Y", "Z".

    Returns:
        Numpy array: Parcel index ordered by the coordinate.
    """
    coords_ds = load_from_disk(coords_ds_path)

    parcel_coords_list = coords_ds[axis_index]  # sort by y
    reorder_ids_by_coord = sorted(range(len(parcel_coords_list)), key=lambda k: parcel_coords_list[k])
    reorder_ids_by_coord = np.array(reorder_ids_by_coord)
    return reorder_ids_by_coord



def get_attention_cls_token(attn_probs):
    """Get the attention CLS token from the model from
    the last attention layer / first head.

    Args:
        attn_probs (tensor): attentions from model output.

    Return:
        numpu array
    """
    attn_probs_heads = attn_probs[-1].squeeze(0)
    attn_probs_avg = attn_probs_heads.mean(dim=0, keepdim=True)
    return attn_probs_avg[:, 0, :].cpu().numpy()


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    if "labels" not in examples:
        labels = torch.tensor([1 for _ in range(len(pixel_values))])
    labels = torch.stack([torch.tensor(example["labels"]) for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": pixel_values,
        "labels": labels
    }


def padding_timeseries_For_vitmae(pixel_values, image_size):
    # pixel_values is [batch, channels=3, number of parcels, number of time points]. Pad to [batch, channels=3, 434, 434]
    height_pad_total = image_size[0] - pixel_values.shape[2]
    height_pad_total_half = height_pad_total // 2
    width_pad_total = image_size[1] - pixel_values.shape[3]
    width_pad_total_half = width_pad_total // 2
    return F.pad(pixel_values, (width_pad_total_half, width_pad_total_half, height_pad_total_half, height_pad_total_half), "constant", -1)
