"""Checkpoint utilities for BrainHarmonix finetuning.

Also re-exports BrainHarmonixDataset: finetune_brainharmonix.py imports it from
datasets.py while extract_brainharmonix.py imports it from here -- defined once in
datasets.py, re-exported below.
"""
import torch

from .datasets import BrainHarmonixDataset  # noqa: F401 (re-exported for extract_brainharmonix.py)


def save_checkpoint(model, optimizer, epoch, val_metrics, path, task, label_map=None):
    """Save a finetuning checkpoint.

    Only the harmonizer's state dict is saved: the fMRI/T1 encoders are frozen during
    finetuning in this pass (see models.BrainHarmonixSelfSupervisedModel) and are always
    reloaded from their original pretrained checkpoints, never from this file.
    """
    if task != "self-supervised":
        raise NotImplementedError(
            "save_checkpoint only supports task='self-supervised' right now."
        )

    torch.save(
        {
            "harmonizer": model.harmonizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "task": task,
            "label_map": label_map,
        },
        path,
    )
