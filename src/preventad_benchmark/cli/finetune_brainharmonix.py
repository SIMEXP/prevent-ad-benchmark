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
    python finetune_brainharmonix.py --dataset data.arrow --target sex --task classification

    # Regression (e.g., age prediction)
    python finetune_brainharmonix.py --dataset data.arrow --target age --task regression

References:
    - BrainHarmony: https://github.com/hzlab/Brain-Harmony
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from preventad_benchmark.config import (
    BRAINHARMONIX_CHECKPOINTS,
    BRAINHARMONIX_POS_EMBED_PATHS,
)
from preventad_benchmark.models.brainharmonix.loaders import load_all_models
from preventad_benchmark.models.brainharmonix.datasets import BrainHarmonixDataset, FineTuneDataset
from preventad_benchmark.models.brainharmonix.models import (
    BrainHarmonixSelfSupervisedModel,
    BrainHarmonixSupervisedModel,
)
from preventad_benchmark.models.brainharmonix.fintuning_engines import (
    train_epoch_self_supervised,
    evaluate_self_supervised,
    train_epoch_supervised,
    evaluate_supervised,
)
from preventad_benchmark.models.brainharmonix.utils import save_checkpoint

# Default paths for CLI argument defaults
DEFAULT_GRADIENT_PATH = str(BRAINHARMONIX_POS_EMBED_PATHS["gradient"])
DEFAULT_GEO_HARM_PATH = str(BRAINHARMONIX_POS_EMBED_PATHS["geo_harm"])
DEFAULT_HARMONIZER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["harmonizer"])
DEFAULT_FMRI_ENCODER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["fmri_encoder"])
DEFAULT_T1_ENCODER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["t1_encoder"])

def load_models(args, device: torch.device) -> tuple:
    """Load all three model components using shared loaders."""
    return load_all_models(
        device=device,
        mode="train",
        harmonizer_ckpt=args.harmonizer_ckpt,
        fmri_ckpt=args.fmri_ckpt,
        t1_ckpt=args.t1_ckpt,
        gradient_path=args.gradient_path,
        geo_harm_path=args.geo_harm_path,
    )


def main():
    """Main function to parse arguments and run fine-tuning."""
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
        default=Path("outputs/finetune/brainharmonix"),
        help="Output directory (default: outputs/finetune/brainharmonix)",
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
            freeze_stage0_encoders=not args.unfreeze_harmonizer,
        )
        # Use lower learning rate for self-supervised
        if args.lr == 1e-4 and not args.unfreeze_harmonizer:  # Default was not changed
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
    metrics = []
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

            metrics.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            })

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
                metrics.append({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                })
            else:
                print(f", Val Loss={val_metrics['loss']:.4f}, Val MAE={val_metrics['mae']:.4f}")
                metric_key = "mae"
                is_best = val_metrics["mae"] < best_loss
                if is_best:
                    best_loss = val_metrics["mae"]
                metrics.append({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                })
        # Save best model
        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                args.output_dir / "harmonizer_checkpoint_best.pt",
                args.task,
                label_map=dataset.label_map if args.task == "classification" else None,
            )
            # save the fmri encoder and t1 encoder checkpoints as well for reproducibility
            if args.unfreeze_harmonizer:
                torch.save(model.fmri_encoder.state_dict(), args.output_dir / "fmri_encoder_checkpoint_best.pt")
                torch.save(model.t1_encoder.state_dict(), args.output_dir / "t1_encoder_checkpoint_best.pt")

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        val_metrics,
        args.output_dir / "harmonizer_checkpoint_final.pt",
        args.task,
        label_map=dataset.label_map if args.task == "classification" else None,
    )
    if args.unfreeze_harmonizer:
        torch.save(model.fmri_encoder.state_dict(), args.output_dir / "fmri_encoder_checkpoint_final.pt")
        torch.save(model.t1_encoder.state_dict(), args.output_dir / "t1_encoder_checkpoint_final.pt")

    # Save config
    config = {
        "task": args.task,
        "target": args.target,
        "best_metric_value": best_loss,
        "metric_key": metric_key if args.task != "self-supervised" else "loss",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "metrics": metrics,
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
        print("\nTo use the fine-tuned harmonizer with extract_brainharmonix.py:")
        print(f"  python extract_brainharmonix.py --dataset {args.dataset} \\")
        print(f"      --harmonizer-ckpt {args.output_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    main()
