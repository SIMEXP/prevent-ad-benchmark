"""Loaders for BrainHarmonix's pretrained fMRI encoder, T1 encoder, and harmonizer.

Wraps the `brainharmonix` pip package (pinned in pyproject.toml to htwangtw/Brain-Harmony,
a fork of MedARC-AI/Brain-Harmony with a fixed `is_flash_attn_2_available` -- see
https://github.com/htwangtw/Brain-Harmony/commit/631bf4f18f6d410efa20684a2a690201461539de)
rather than the `BrainHarmony/` git submodule: the submodule's bare `from libs...`/
`from modules...` imports don't resolve without a sys.path hack, and it has a stray
`breakpoint()` plus an `attn_utils` import bug in its eager-attention path that the pip
fork fixes (it also adds the "sdpa" attention backend used below).

The fMRI encoder is portable to CPU (attn_mode="sdpa"). The T1 encoder and the harmonizer
both hardcode flash-attention with no override, so constructing/running them for real
requires a GPU with `flash-attn` installed -- expect ModuleNotFoundError here and verify
on a GPU node instead (see plan doc).
"""
from types import SimpleNamespace

import torch

from brainharmonix.libs.model import vit_base_flex
from brainharmonix.libs.position_embedding import (
    BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed,
)
from brainharmonix.modules.harmonizer.stage1_pretrain.models import (
    onetokreg_vit_base_patch16,
)
from brainharmonix.modules.harmonizer.util.t1_encoder import mae_vit_base_patch16

from preventad_benchmark.config import (
    BRAINHARMONIX_NUM_LATENT_TOKENS,
    BRAINHARMONIX_PREVENTAD_TR,
    BRAINHARMONIX_SCHAEFER_ROIS,
    BRAINHARMONIX_STANDARD_TIME,
    BRAINHARMONIX_T1_TARGET_SHAPE,
    BRAINHARMONIX_TARGET_NUM_PATCHES,
)

# Frame count per patch at PreventAD's TR, chosen so each patch spans the same
# wall-clock duration as upstream's UKB-pretrained patch_size=48 @ TR=0.735s.
# Kept in sync with the identical computation in datasets.py.
FMRI_PATCH_SIZE = round(BRAINHARMONIX_STANDARD_TIME / BRAINHARMONIX_PREVENTAD_TR)


def load_fmri_encoder(ckpt_path, gradient_path, geo_harm_path, device, is_finetuned=False):
    """Load the fMRI encoder (portable to CPU; attn_mode="sdpa").

    `is_finetuned` is accepted for interface parity with load_t1_encoder/load_harmonizer
    but currently unused: the fMRI encoder is frozen during finetuning in this pass (see
    models.BrainHarmonixSelfSupervisedModel) and always loads the original pretrained
    checkpoint.
    """
    pos_embed_config = SimpleNamespace(
        grid_size=(BRAINHARMONIX_SCHAEFER_ROIS, BRAINHARMONIX_TARGET_NUM_PATCHES),
        embed_dim=768,
        cls_token=False,
        grad_dim=30,
        gradient=str(gradient_path),
        geoh_dim=200,
        geo_harm=str(geo_harm_path),
        use_pos_embed_decoder=False,  # False for downstream/finetuning tasks (True only for upstream's own pretraining)
    )
    pos_embed = BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed(pos_embed_config)
    encoder = vit_base_flex(
        patch_size=FMRI_PATCH_SIZE,
        pos_embed=pos_embed,
        cls_token=None,
        attn_mode="sdpa",
    )

    # BrainHarmony/modules/harmonizer/stage0_embed/embedding_pretrain.py:84-104: the
    # checkpoint is a raw JEPA EMA-teacher state dict keyed "encoder_ema.<name>".
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "encoder_ema."
    state_dict = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    model_state_dict = encoder.state_dict()
    for key in list(state_dict):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            del state_dict[key]
    encoder.load_state_dict(state_dict, strict=False)

    return encoder.to(device)


def load_t1_encoder(ckpt_path, device, mode="inference", is_finetuned=False):
    """Load the T1 encoder. Requires flash-attn + CUDA to run a forward pass.

    `is_finetuned` is accepted for interface parity but currently unused -- the T1
    encoder is frozen during finetuning in this pass and always loads the original
    pretrained checkpoint (nested under a "model" key, per upstream convention).
    """
    encoder = mae_vit_base_patch16(img_size=BRAINHARMONIX_T1_TARGET_SHAPE)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device).half()  # fp16: required by flash-attn, matches extract_brainharmonix.py's convention
    encoder.train() if mode == "train" else encoder.eval()
    return encoder


def load_harmonizer(ckpt_path, device, mode="inference", is_finetuned=False):
    """Load the harmonizer (fusion module). Requires flash-attn + CUDA to run a forward pass.

    If `is_finetuned`, loads our own combined checkpoint's "harmonizer" key (see
    utils.save_checkpoint); otherwise loads the original pretrained checkpoint, nested
    under a "model" key per upstream convention.

    mode="inference" (extract_brainharmonix.py): weights are cast to fp16, matching
    upstream's own convention -- safe since nothing ever calls .step() on them here.

    mode="train" (finetune_brainharmonix.py): weights are kept fp32. AdamW updating
    fp16-stored params corrupts them to inf after a single step (verified directly --
    fp16's dynamic range is too narrow for Adam's bias-corrected second-moment
    denominator). The forward pass still needs fp16 for flash-attn --
    models.BrainHarmonixSelfSupervisedModel wraps it in torch.autocast instead of
    relying on the stored weight dtype.
    """
    harmonizer = onetokreg_vit_base_patch16(num_latent_tokens=BRAINHARMONIX_NUM_LATENT_TOKENS)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["harmonizer"] if is_finetuned else checkpoint["model"]
    harmonizer.load_state_dict(state_dict, strict=False)
    harmonizer = harmonizer.to(device)
    if mode == "train":
        harmonizer.train()
    else:
        harmonizer = harmonizer.half()
        harmonizer.eval()
    return harmonizer


def load_all_models(
    device,
    mode="train",
    harmonizer_ckpt=None,
    fmri_ckpt=None,
    t1_ckpt=None,
    gradient_path=None,
    geo_harm_path=None,
):
    """Load all three BrainHarmonix components for finetuning.

    The fMRI/T1 encoders always load the original pretrained checkpoint (is_finetuned=False):
    they're frozen during finetuning in this pass (see models.BrainHarmonixSelfSupervisedModel)
    and are never themselves fine-tuned.
    """
    fmri_encoder = load_fmri_encoder(fmri_ckpt, gradient_path, geo_harm_path, device, is_finetuned=False)
    t1_encoder = load_t1_encoder(t1_ckpt, device, mode=mode, is_finetuned=False)
    harmonizer = load_harmonizer(harmonizer_ckpt, device, mode=mode, is_finetuned=False)
    return fmri_encoder, t1_encoder, harmonizer
