"""Adapter model classes wrapping BrainHarmonix's fMRI/T1 encoders + harmonizer.

Precision follows extract_brainharmonix.py's own convention: the fMRI encoder runs in
fp32 with SDPA attention; the T1 encoder and harmonizer require flash-attention and run
in fp16.
"""
import torch
import torch.nn as nn


class BrainHarmonixSelfSupervisedModel(nn.Module):
    """Self-supervised finetuning: freeze the pretrained fMRI/T1 encoders, only adapt
    the harmonizer (fusion module) to PreventAD data -- the same "freeze the backbone,
    adapt a small piece" pattern used for BrainLM's finetune_brainlm.py.
    """

    def __init__(self, fmri_encoder, t1_encoder, harmonizer):
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.t1_encoder = t1_encoder
        self.harmonizer = harmonizer

        for param in self.fmri_encoder.parameters():
            param.requires_grad_(False)
        for param in self.t1_encoder.parameters():
            param.requires_grad_(False)
        self.fmri_encoder.eval()
        self.t1_encoder.eval()

    def forward(self, fmri, t1, attn_mask, patch_size):
        with torch.no_grad():
            fmri_embed = self.fmri_encoder(fmri, patch_size, attention_mask=attn_mask)  # fp32, SDPA
            t1_embed = self.t1_encoder(t1.half())  # fp16, flash-attn

        combined = torch.cat([fmri_embed.half(), t1_embed], dim=1)
        loss, pred, mask = self.harmonizer(combined, attn_mask)
        return loss, pred, mask


class BrainHarmonixSupervisedModel(nn.Module):
    """Classification/regression finetuning on top of the harmonizer's latent tokens.

    Not implemented in this pass (see plan doc) -- exists only so
    finetune_brainharmonix.py's unconditional import of this name succeeds.
    """

    def __init__(
        self,
        fmri_encoder,
        t1_encoder,
        harmonizer,
        num_classes,
        task,
        pooling="mean",
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        raise NotImplementedError(
            "BrainHarmonixSupervisedModel (classification/regression) is not implemented "
            "yet; only task='self-supervised' is supported."
        )
