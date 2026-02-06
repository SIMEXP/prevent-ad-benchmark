#!/usr/bin/env python3
"""Fine-tune BrainHarmonix models for downstream tasks.

This script fine-tunes the BrainHarmonix harmonizer model. It supports:
- Self-supervised: Continued pre-training with reconstruction loss (no labels needed)
- Classification: Supervised learning for categorical targets (e.g., disease diagnosis)
- Regression: Supervised learning for continuous targets (e.g., age prediction)

The fine-tuned checkpoint can then be used with extract_brainharmonix.py to get
task-adapted embeddings.

Architecture:
    Self-supervised:
        Input embeddings → Harmonizer (encoder + decoder) → Reconstruction loss

    Classification/Regression:
        Input embeddings → Harmonizer encoder → Latent tokens → MLP Head → Output

Usage:
    # Self-supervised (no labels needed, adapts to your dataset)
    python finetune_brainharmonix.py --dataset data.arrow --task self-supervised

    # Classification (e.g., sex classification)
    python finetune_brainharmonix.py --dataset data.arrow --target Sex --task classification

    # Regression (e.g., age prediction)
    python finetune_brainharmonix.py --dataset data.arrow --target Candidate_Age --task regression

References:
    - BrainHarmony: https://github.com/hzlab/Brain-Harmony
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import brainharmonix.libs.model as model
import brainharmonix.libs.position_embedding as pos_embeds
from brainharmonix.configs.harmonizer.stage0_embed import conf_embed_downstream
from brainharmonix.modules.harmonizer.stage1_pretrain.models import (
    onetokreg_vit_base_patch16,
)
from brainharmonix.modules.harmonizer.util.t1_encoder import mae_vit_base_patch16

from hfplayground.models.brainharmonix.utils import BrainHarmonixDataset


# Default paths
DEFAULT_GRADIENT_PATH = "BrainHarmony/brainharmony_pos_embed/gradient_mapping_400.csv"
DEFAULT_GEO_HARM_PATH = "BrainHarmony/brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv"
DEFAULT_HARMONIZER_CKPT = "models/brain-harmonix/harmonizer/model.pth"
DEFAULT_FMRI_ENCODER_CKPT = "models/brain-harmonix/harmonix-f/model.pth"
DEFAULT_T1_ENCODER_CKPT = "models/brain-harmonix/harmonix-s/model.pth"


class MLPHead(nn.Module):
    """MLP classification/regression head."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class BrainHarmonixSelfSupervisedModel(nn.Module):
    """BrainHarmonix model for self-supervised fine-tuning.

    Uses the harmonizer's built-in reconstruction objective:
    - Encoder compresses fMRI+T1 embeddings into latent tokens
    - Decoder reconstructs the original embeddings from latent tokens
    - Loss = MSE between original and reconstructed embeddings
    """

    def __init__(
        self,
        fmri_encoder: nn.Module,
        t1_encoder: nn.Module,
        harmonizer: nn.Module,
        freeze_stage0_encoders: bool = True,
    ):
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.t1_encoder = t1_encoder
        self.harmonizer = harmonizer

        # Always freeze stage0 encoders (fMRI and T1)
        if freeze_stage0_encoders:
            for param in self.fmri_encoder.parameters():
                param.requires_grad = False
            for param in self.t1_encoder.parameters():
                param.requires_grad = False

        # Harmonizer is trainable for self-supervised learning

    def forward(
        self,
        fmri: torch.Tensor,
        t1: torch.Tensor,
        attn_mask: torch.Tensor,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction loss.

        Returns:
            loss: Reconstruction loss (MSE)
            pred: Reconstructed embeddings
        """
        # Encode fMRI (fp32 for SDPA)
        with torch.no_grad():
            fmri_embed = self.fmri_encoder(fmri, patch_size, attention_mask=attn_mask)
            t1_embed = self.t1_encoder(t1)

        # Combine embeddings (match harmonizer dtype - bf16 or fp16)
        harmonizer_dtype = next(self.harmonizer.parameters()).dtype
        combined = torch.cat([fmri_embed.to(harmonizer_dtype), t1_embed], dim=1)

        # Harmonizer forward returns (loss, pred, None)
        # Uses its built-in reconstruction objective
        loss, pred, _ = self.harmonizer(combined, attn_mask)

        return loss, pred


class BrainHarmonixSupervisedModel(nn.Module):
    """BrainHarmonix model with supervised fine-tuning head.

    Architecture:
        fMRI encoder (frozen) → fMRI embeddings
        T1 encoder (frozen) → T1 embeddings
        Concatenate → Harmonizer encoder → Latent tokens
        Pool latent tokens → MLP head → Output
    """

    def __init__(
        self,
        fmri_encoder: nn.Module,
        t1_encoder: nn.Module,
        harmonizer: nn.Module,
        num_classes: int,
        task: str = "classification",
        pooling: str = "mean",
        freeze_encoders: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fmri_encoder = fmri_encoder
        self.t1_encoder = t1_encoder
        self.harmonizer = harmonizer
        self.task = task
        self.pooling = pooling

        # Freeze encoders if requested
        if freeze_encoders:
            for param in self.fmri_encoder.parameters():
                param.requires_grad = False
            for param in self.t1_encoder.parameters():
                param.requires_grad = False
            for param in self.harmonizer.parameters():
                param.requires_grad = False

        # Harmonizer outputs 129 tokens × 768 dim (1 CLS + 128 latent)
        embed_dim = 768
        if pooling == "cls":
            in_features = embed_dim
        elif pooling == "mean":
            in_features = embed_dim
        elif pooling == "concat":
            in_features = embed_dim * 129
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        out_features = num_classes if task == "classification" else 1
        self.head = MLPHead(in_features, hidden_dim, out_features, dropout)

    def forward(
        self,
        fmri: torch.Tensor,
        t1: torch.Tensor,
        attn_mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        # Encode fMRI (fp32 for SDPA)
        fmri_embed = self.fmri_encoder(fmri, patch_size, attention_mask=attn_mask)

        # Encode T1 (bf16/fp16 for flash attention)
        t1_embed = self.t1_encoder(t1)

        # Combine and pass through harmonizer (match harmonizer dtype)
        harmonizer_dtype = next(self.harmonizer.parameters()).dtype
        combined = torch.cat([fmri_embed.to(harmonizer_dtype), t1_embed], dim=1)
        latent, _ = self.harmonizer.forward_encoder(combined, attn_mask)
        latent = latent.float()

        # Pool latent tokens
        if self.pooling == "cls":
            pooled = latent[:, 0]
        elif self.pooling == "mean":
            pooled = latent[:, 1:].mean(dim=1)
        elif self.pooling == "concat":
            pooled = latent.view(latent.size(0), -1)

        return self.head(pooled)


class FineTuneDataset(Dataset):
    """Wrapper dataset that optionally adds labels for fine-tuning."""

    def __init__(
        self,
        base_dataset: BrainHarmonixDataset,
        target_column: Optional[str] = None,
        task: str = "self-supervised",
        label_map: Optional[dict] = None,
    ):
        self.base_dataset = base_dataset
        self.target_column = target_column
        self.task = task
        self.label_map = label_map

        # Build label map for classification
        if task == "classification" and label_map is None and target_column is not None:
            unique_labels = set()
            for i in range(len(base_dataset)):
                sample = base_dataset.dataset[i]
                unique_labels.add(sample[target_column])
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            print(f"Label map: {self.label_map}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]

        # Add label for supervised tasks
        if self.task != "self-supervised" and self.target_column is not None:
            raw_sample = self.base_dataset.dataset[idx]
            target_value = raw_sample[self.target_column]

            if self.task == "classification":
                item["label"] = self.label_map[target_value]
            else:
                item["label"] = float(target_value)

        return item

    @property
    def num_classes(self) -> int:
        if self.task == "classification" and self.label_map is not None:
            return len(self.label_map)
        return 1


def get_pos_embed(name: str, **kwargs):
    """Create position embedding module by name."""
    return getattr(pos_embeds, name)(kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name: str, attn_mode: str = "sdpa", **kwargs):
    """Create encoder model by name."""
    return getattr(model, name)(
        pos_embed=pos_embed, cls_token=cls_token, attn_mode=attn_mode, **kwargs
    )


def load_models(args, device: torch.device, use_bfloat16: bool = True) -> tuple:
    """Load all three model components.

    Args:
        args: Command line arguments
        device: Target device
        use_bfloat16: Use bfloat16 for training stability (default: True)
    """
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    # Harmonizer (bf16/fp16 for flash attention)
    harmonizer = onetokreg_vit_base_patch16(
        norm_pix_loss=True, img_size=(160, 192, 160), num_latent_tokens=128
    )
    state_dict = torch.load(args.harmonizer_ckpt, map_location="cpu", weights_only=False)["model"]
    harmonizer.load_state_dict(state_dict, strict=False)
    harmonizer = harmonizer.to(device).to(dtype)

    # fMRI encoder (fp32 for SDPA)
    config = conf_embed_downstream.get_config()
    config.pos_embed.model_args.gradient = str(args.gradient_path)
    config.pos_embed.model_args.geo_harm = str(args.geo_harm_path)
    pos_embed = get_pos_embed(**config.pos_embed)
    fmri_encoder = get_encoder(pos_embed, None, **config.encoder)

    state_dict = torch.load(args.fmri_ckpt, map_location="cpu", weights_only=False)
    prefix = "encoder_ema."
    state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    state_dict.pop("pos_embed.emb_h_encoder", None)
    state_dict.pop("pos_embed.emb_h_decoder", None)
    fmri_encoder.load_state_dict(state_dict, strict=False)
    fmri_encoder = fmri_encoder.to(device)

    # T1 encoder (bf16/fp16 for flash attention)
    t1_encoder = mae_vit_base_patch16(img_size=(160, 192, 160))
    state_dict = torch.load(args.t1_ckpt, map_location="cpu", weights_only=False)["model"]
    t1_encoder.load_state_dict(state_dict, strict=False)
    t1_encoder = t1_encoder.to(device).to(dtype)

    return fmri_encoder, t1_encoder, harmonizer


def train_epoch_self_supervised(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train one epoch for self-supervised learning."""
    model.train()
    # Keep stage0 encoders in eval mode
    model.fmri_encoder.eval()
    model.t1_encoder.eval()

    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device).bfloat16()
        attn_mask = batch["attn_mask"].to(device).bool()
        patch_size = batch["patch_size"][0].item()

        optimizer.zero_grad()

        # No GradScaler needed - harmonizer is already FP16
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(fmri, t1, attn_mask, patch_size)

        loss.backward()
        optimizer.step()

        batch_size = fmri.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        pbar.set_postfix({"loss": loss.item()})

    return {"loss": total_loss / num_samples}


@torch.no_grad()
def evaluate_self_supervised(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate self-supervised model."""
    model.eval()

    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device).bfloat16()
        attn_mask = batch["attn_mask"].to(device).bool()
        patch_size = batch["patch_size"][0].item()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(fmri, t1, attn_mask, patch_size)

        batch_size = fmri.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return {"loss": total_loss / num_samples}


def train_epoch_supervised(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    task: str,
) -> dict:
    """Train one epoch for supervised learning."""
    model.train()
    model.fmri_encoder.eval()
    model.t1_encoder.eval()
    model.harmonizer.eval()

    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device).bfloat16()
        attn_mask = batch["attn_mask"].to(device).bool()
        patch_size = batch["patch_size"][0].item()
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(fmri, t1, attn_mask, patch_size)
            if task == "classification":
                loss = criterion(outputs, labels.long())
            else:
                loss = criterion(outputs.squeeze(), labels.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = fmri.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        pbar.set_postfix({"loss": loss.item()})

    return {"loss": total_loss / num_samples}


@torch.no_grad()
def evaluate_supervised(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> dict:
    """Evaluate supervised model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        fmri = batch["fmri"].to(device)
        t1 = batch["t1"].to(device).bfloat16()
        attn_mask = batch["attn_mask"].to(device).bool()
        patch_size = batch["patch_size"][0].item()
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(fmri, t1, attn_mask, patch_size)
            if task == "classification":
                loss = criterion(outputs, labels.long())
                preds = outputs.argmax(dim=1)
            else:
                loss = criterion(outputs.squeeze(), labels.float())
                preds = outputs.squeeze()

        batch_size = fmri.size(0)
        total_loss += loss.item() * batch_size
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = {"loss": total_loss / len(dataloader.dataset)}

    if task == "classification":
        accuracy = (all_preds == all_labels).float().mean().item()
        metrics["accuracy"] = accuracy
    else:
        mse = F.mse_loss(all_preds.float(), all_labels.float()).item()
        mae = F.l1_loss(all_preds.float(), all_labels.float()).item()
        metrics["mse"] = mse
        metrics["mae"] = mae

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    task: str,
    label_map: Optional[dict] = None,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "task": task,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if task == "self-supervised":
        # Save only the harmonizer state dict (compatible with extract script)
        checkpoint["model"] = model.harmonizer.state_dict()
    else:
        # Save full model state dict
        checkpoint["model_state_dict"] = model.state_dict()
        if label_map is not None:
            checkpoint["label_map"] = label_map

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BrainHarmonix for downstream tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Self-supervised (adapt to your dataset, no labels needed)
    python finetune_brainharmonix.py --dataset data.arrow --task self-supervised

    # Classification task
    python finetune_brainharmonix.py --dataset data.arrow --target Sex --task classification

    # Regression task
    python finetune_brainharmonix.py --dataset data.arrow --target Candidate_Age --task regression

    # Custom training parameters
    python finetune_brainharmonix.py --dataset data.arrow --task self-supervised \\
        --epochs 50 --lr 1e-5 --batch-size 4
        """,
    )

    # Data arguments
    parser.add_argument("--dataset", type=Path, required=True, help="Path to Arrow dataset")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column for prediction (required for classification/regression)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["self-supervised", "classification", "regression"],
        default="self-supervised",
        help="Task type (default: self-supervised)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Model arguments
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean", "concat"],
        default="mean",
        help="Pooling strategy for supervised tasks (default: mean)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of MLP head for supervised tasks (default: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for supervised tasks (default: 0.1)",
    )
    parser.add_argument(
        "--unfreeze-harmonizer",
        action="store_true",
        help="Unfreeze harmonizer for supervised tasks (default: frozen)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Checkpoint arguments
    parser.add_argument(
        "--harmonizer-ckpt",
        type=Path,
        default=Path(DEFAULT_HARMONIZER_CKPT),
        help="Path to harmonizer checkpoint",
    )
    parser.add_argument(
        "--fmri-ckpt",
        type=Path,
        default=Path(DEFAULT_FMRI_ENCODER_CKPT),
        help="Path to fMRI encoder checkpoint",
    )
    parser.add_argument(
        "--t1-ckpt",
        type=Path,
        default=Path(DEFAULT_T1_ENCODER_CKPT),
        help="Path to T1 encoder checkpoint",
    )
    parser.add_argument(
        "--gradient-path",
        type=Path,
        default=Path(DEFAULT_GRADIENT_PATH),
        help="Path to gradient mapping CSV",
    )
    parser.add_argument(
        "--geo-harm-path",
        type=Path,
        default=Path(DEFAULT_GEO_HARM_PATH),
        help="Path to geometric harmonics CSV",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/finetune"),
        help="Output directory (default: outputs/finetune)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.task in ["classification", "regression"] and args.target is None:
        parser.error(f"--target is required for {args.task} task")

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Task: {args.task}")
    if args.target:
        print(f"Target: {args.target}")

    # Load base dataset
    print("\nLoading dataset...")
    base_dataset = BrainHarmonixDataset(str(args.dataset))

    # Create dataset
    dataset = FineTuneDataset(base_dataset, args.target, args.task)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load pretrained models
    print("\nLoading pretrained models...")
    fmri_encoder, t1_encoder, harmonizer = load_models(args, device)

    # Create model based on task
    if args.task == "self-supervised":
        model = BrainHarmonixSelfSupervisedModel(
            fmri_encoder=fmri_encoder,
            t1_encoder=t1_encoder,
            harmonizer=harmonizer,
            freeze_stage0_encoders=True,
        )
        # Use lower learning rate for self-supervised
        if args.lr == 1e-4:  # Default was not changed
            args.lr = 1e-5
            print(f"Using lower learning rate for self-supervised: {args.lr}")
    else:
        model = BrainHarmonixSupervisedModel(
            fmri_encoder=fmri_encoder,
            t1_encoder=t1_encoder,
            harmonizer=harmonizer,
            num_classes=dataset.num_classes,
            task=args.task,
            pooling=args.pooling,
            freeze_encoders=not args.unfreeze_harmonizer,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")

    # Setup criterion for supervised tasks
    criterion = None
    if args.task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif args.task == "regression":
        criterion = nn.MSELoss()

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if args.task == "self-supervised":
            train_metrics = train_epoch_self_supervised(
                model, train_loader, optimizer, device, epoch
            )
            val_metrics = evaluate_self_supervised(model, val_loader, device)
            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

            # Track best by validation loss
            metric_key = "loss"
            is_best = val_metrics["loss"] < best_loss
            if is_best:
                best_loss = val_metrics["loss"]

        else:
            train_metrics = train_epoch_supervised(
                model, train_loader, criterion, optimizer, device, epoch, scaler, args.task
            )
            val_metrics = evaluate_supervised(model, val_loader, criterion, device, args.task)

            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}", end="")
            if args.task == "classification":
                print(f", Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
                metric_key = "accuracy"
                is_best = val_metrics["accuracy"] > best_loss  # best_loss used as best_metric here
                if is_best:
                    best_loss = val_metrics["accuracy"]
            else:
                print(f", Val Loss={val_metrics['loss']:.4f}, Val MAE={val_metrics['mae']:.4f}")
                metric_key = "mae"
                is_best = val_metrics["mae"] < best_loss
                if is_best:
                    best_loss = val_metrics["mae"]

        # Save best model
        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                args.output_dir / "checkpoint_best.pt",
                args.task,
                label_map=dataset.label_map if args.task == "classification" else None,
            )

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        val_metrics,
        args.output_dir / "checkpoint_final.pt",
        args.task,
        label_map=dataset.label_map if args.task == "classification" else None,
    )

    # Save config
    config = {
        "task": args.task,
        "target": args.target,
        "best_metric_value": best_loss,
        "metric_key": metric_key if args.task != "self-supervised" else "loss",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    if args.task == "classification":
        config["num_classes"] = dataset.num_classes
        config["label_map"] = dataset.label_map
    if args.task != "self-supervised":
        config["pooling"] = args.pooling
        config["hidden_dim"] = args.hidden_dim

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Best {metric_key}: {best_loss:.4f}")
    print(f"Checkpoints saved to {args.output_dir}")

    if args.task == "self-supervised":
        print(f"\nTo use the fine-tuned harmonizer with extract_brainharmonix.py:")
        print(f"  python extract_brainharmonix.py --dataset {args.dataset} \\")
        print(f"      --harmonizer-ckpt {args.output_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    main()
