"""Training/eval loops for BrainHarmonix finetuning.

Ported from BrainHarmony/modules/harmonizer/stage1_pretrain/engine_pretrain.py's
train_one_epoch/evaluate, stripped of distributed training, AMP loss-scaling, and
MetricLogger scaffolding down to a plain zero_grad -> forward -> backward -> step loop.
"""
import torch


def train_epoch_self_supervised(model, train_loader, optimizer, device, epoch):
    model.train()
    # Keep the frozen encoders in eval mode even though the composite model is in train().
    model.fmri_encoder.eval()
    model.t1_encoder.eval()

    total_loss = 0.0
    n_batches = 0
    for batch in train_loader:
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        patch_size = batch["patch_size"][0].item()

        optimizer.zero_grad()
        loss, _, _ = model(fmri, t1, attn_mask, patch_size)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def evaluate_self_supervised(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            fmri = batch["fmri"].to(device)
            t1 = batch["t1"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            patch_size = batch["patch_size"][0].item()

            loss, _, _ = model(fmri, t1, attn_mask, patch_size)
            total_loss += loss.item()
            n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


def train_epoch_supervised(model, train_loader, criterion, optimizer, device, epoch, scaler, task):
    raise NotImplementedError(
        "Classification/regression finetuning is not implemented yet; "
        "only task='self-supervised' is supported."
    )


def evaluate_supervised(model, val_loader, criterion, device, task):
    raise NotImplementedError(
        "Classification/regression finetuning is not implemented yet; "
        "only task='self-supervised' is supported."
    )
