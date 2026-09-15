"""Dataset adapters bridging PreventAD Arrow datasets to BrainHarmonix model inputs.

None of BrainHarmony's own dataset classes (BrainHarmony/datasets/datasets.py) are reusable
as-is -- they expect raw per-subject files at hardcoded paths, not a HuggingFace Arrow
dataset. BrainHarmonixDataset below is a from-scratch adapter; only the *algorithms* for
patch/attention-mask construction are ported from upstream, not the code itself. The T1
crop/pad step uses a generic center-crop-or-pad rather than upstream's pipeline-specific
hardcoded pad widths, since PreventAD's fmriprep-native T1 shape isn't guaranteed to match
BrainHarmony's own preprocessing pipeline.
"""
import torch
from torch.utils.data import Dataset

from preventad_benchmark.config import (
    BRAINHARMONIX_PREVENTAD_TR,
    BRAINHARMONIX_SCHAEFER_ROIS,
    BRAINHARMONIX_STANDARD_TIME,
    BRAINHARMONIX_T1_TARGET_SHAPE,
    BRAINHARMONIX_TARGET_NUM_PATCHES,
)

# Frame count per patch at PreventAD's TR, chosen so each patch spans the same
# wall-clock duration as upstream's UKB-pretrained patch_size=48 @ TR=0.735s.
# Kept in sync with the identical computation in loaders.py.
FMRI_PATCH_SIZE = round(BRAINHARMONIX_STANDARD_TIME / BRAINHARMONIX_PREVENTAD_TR)


def _center_crop_or_pad(volume: torch.Tensor, target_shape) -> torch.Tensor:
    """Center-crop (if larger) or zero-pad (if smaller) each axis to `target_shape`."""
    for axis, target in enumerate(target_shape):
        size = volume.shape[axis]
        if size > target:
            start = (size - target) // 2
            volume = volume.narrow(axis, start, target)
        elif size < target:
            pad_total = target - size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            # F.pad pads from the last dimension backward.
            pad = [0, 0] * volume.ndim
            pad_index = (volume.ndim - 1 - axis) * 2
            pad[pad_index] = pad_before
            pad[pad_index + 1] = pad_after
            volume = torch.nn.functional.pad(volume, pad)
    return volume


class BrainHarmonixDataset(Dataset):
    """Wraps one PreventAD Arrow dataset row into BrainHarmonix model inputs.

    Produces the fMRI patch grid `(1, 400, patch_size*18)`, the T1 volume
    `(1, *BRAINHARMONIX_T1_TARGET_SHAPE)`, and a flat `(400*18,)` attention mask
    (1 = real timepoint, 0 = zero-padded) matching what the fMRI encoder /
    harmonizer forward passes expect (see extract_brainharmonix.py).
    """

    def __init__(self, arrow_dataset):
        self.dataset = arrow_dataset
        self.patch_size = FMRI_PATCH_SIZE
        self.target_num_patches = BRAINHARMONIX_TARGET_NUM_PATCHES
        self.target_time_len = self.patch_size * self.target_num_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]

        timeseries = torch.tensor(row["raw_timeseries"], dtype=torch.float32)  # (T, 400)
        timeseries = timeseries.transpose(0, 1)  # (400, T)
        n_rois, n_frames = timeseries.shape
        if n_rois != BRAINHARMONIX_SCHAEFER_ROIS:
            raise ValueError(f"expected {BRAINHARMONIX_SCHAEFER_ROIS} ROIs, got {n_rois}")

        valid_len = min(n_frames, self.target_time_len)
        fmri = torch.zeros(n_rois, self.target_time_len, dtype=torch.float32)
        fmri[:, :valid_len] = timeseries[:, :valid_len]
        fmri = fmri.unsqueeze(0)  # (1, 400, target_time_len): channel dim for the Conv2d patch embed

        valid_patches = valid_len // self.patch_size
        patch_mask = torch.zeros(self.target_num_patches, dtype=torch.float32)
        patch_mask[:valid_patches] = 1.0
        attn_mask = patch_mask.unsqueeze(0).expand(n_rois, -1).reshape(-1)  # (400*18,)

        t1 = torch.load(row["t1_filepath"])
        if t1.ndim == 4:  # drop a leading singleton channel dim if the saved tensor has one
            t1 = t1.squeeze(0)
        t1 = _center_crop_or_pad(t1, BRAINHARMONIX_T1_TARGET_SHAPE)
        t1 = t1.unsqueeze(0)  # (1, H, W, D)

        return {
            "fmri": fmri,
            "t1": t1,
            "attn_mask": attn_mask,
            "patch_size": self.patch_size,
            "participant_id": row["participant_id"],
        }


class FineTuneDataset(Dataset):
    """Wraps BrainHarmonixDataset for a specific finetuning task.

    Only task="self-supervised" is implemented in this pass; classification/regression
    raise NotImplementedError rather than silently returning the wrong thing.
    """

    def __init__(self, base_dataset: BrainHarmonixDataset, target_column, task):
        if task != "self-supervised":
            raise NotImplementedError(
                f"FineTuneDataset only supports task='self-supervised' right now, got {task!r}"
            )
        self.base_dataset = base_dataset
        self.target_column = target_column
        self.task = task
        self.num_classes = None
        self.label_map = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx]
