"""Training/eval loops for BrainHarmonix finetuning.

Ported from BrainHarmony/modules/harmonizer/stage1_pretrain/engine_pretrain.py's
train_one_epoch/evaluate, stripped of distributed training, AMP loss-scaling, and
MetricLogger scaffolding down to a plain zero_grad -> forward -> backward -> step loop.
"""
import torch


def train_epoch_self_supervised(model, train_loader, optimizer, device, epoch, max_grad_norm=1.0):
    model.train()
    # Keep the frozen encoders in eval mode even though the composite model is in train().
    model.fmri_encoder.eval()
    model.t1_encoder.eval()

    total_loss = 0.0
    n_batches = 0
    n_skipped = 0
    for batch in train_loader:
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        patch_size = batch["patch_size"][0].item()

        optimizer.zero_grad()
        loss, _, _ = model(fmri, t1, attn_mask, patch_size)

        # The harmonizer runs fully in fp16 (flash-attn has no attention-backend override),
        # so an occasional overflowed batch is possible even with normalized inputs. Skip
        # the update rather than let a NaN/Inf gradient permanently corrupt the fp16 weights
        # (NaN propagates through every subsequent forward pass once it's in the weights).
        if not torch.isfinite(loss):
            print(f"WARNING: epoch {epoch} got non-finite loss ({loss.item()}); skipping this batch's update")
            n_skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.harmonizer.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "n_skipped_batches": n_skipped}
    return {"loss": total_loss / n_batches, "n_skipped_batches": n_skipped}


def evaluate_self_supervised(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    n_batches = 0
    n_skipped = 0
    with torch.no_grad():
        for batch in val_loader:
            fmri = batch["fmri"].to(device)
            t1 = batch["t1"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            patch_size = batch["patch_size"][0].item()

            loss, _, _ = model(fmri, t1, attn_mask, patch_size)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "n_skipped_batches": n_skipped}
    return {"loss": total_loss / n_batches, "n_skipped_batches": n_skipped}


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
