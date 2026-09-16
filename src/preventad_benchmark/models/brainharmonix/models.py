"""Adapter model classes wrapping BrainHarmonix's fMRI/T1 encoders + harmonizer.

The fMRI encoder runs in fp32 with SDPA attention. The T1 encoder and harmonizer both
require flash-attention, which needs fp16 activations -- but the harmonizer is also the
only trainable component here, and AdamW updating parameters *stored* in fp16 corrupts
them to inf after a single step (verified directly: fp16's dynamic range is too narrow
for Adam's bias-corrected second-moment denominator). So the harmonizer's weights are
kept fp32 (see loaders.load_harmonizer) and its forward pass runs under torch.autocast
to get fp16 activations for flash-attn without touching the stored weight dtype -- the
standard mixed-precision pattern. The T1 encoder is frozen (no optimizer ever touches
it), so it's safe to keep it natively fp16 with no autocast needed.
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
            t1_embed = self.t1_encoder(t1.half())  # fp16, flash-attn (frozen, safe to store natively fp16)

        combined = torch.cat([fmri_embed.half(), t1_embed], dim=1)

        # autocast casts activations to fp16 for the harmonizer's flash-attn ops, without
        # touching its fp32-stored weights (see module docstring). Call forward_encoder/
        # forward_decoder directly rather than harmonizer(combined, attn_mask), so the
        # final (pred - target)**2 mean reduction can be done in fp32 below instead of
        # inside OneTokRegViT.forward_loss's fp16 compute -- a second, independent
        # source of precision loss on top of the weight-dtype issue.
        device_type = combined.device.type
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
            latent, target = self.harmonizer.forward_encoder(combined, attn_mask)
            pred = self.harmonizer.forward_decoder(latent)

        loss = ((pred.float() - target.float()) ** 2).mean()
        return loss, pred, None


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
