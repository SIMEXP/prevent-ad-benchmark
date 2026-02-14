#!/usr/bin/env python3
"""Extract embeddings from BrainHarmonix models.

This script extracts embeddings from three BrainHarmonix model components:
1. fMRI encoder (Harmonix-F): Encodes fMRI time series → (N, 7200, 768)
2. T1 encoder (Harmonix-S): Encodes T1 structural images → (N, 1200, 768)
3. Harmonizer: Fuses fMRI + T1 into joint representation → (N, 129, 768)

The harmonizer output contains 129 tokens: 1 CLS token + 128 latent tokens that
compress the multimodal information from both fMRI and T1.

Usage:
    python extract_brainharmonix.py --dataset data/processed/dataset.arrow --output-dir outputs/

References:
    - BrainHarmony: https://github.com/hzlab/Brain-Harmony
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from preventad_benchmark.config import (
    BRAINHARMONIX_CHECKPOINTS,
    BRAINHARMONIX_POS_EMBED_PATHS,
)
from preventad_benchmark.models.brainharmonix.loaders import (
    load_fmri_encoder,
    load_harmonizer,
    load_t1_encoder,
)
from preventad_benchmark.models.brainharmonix.utils import BrainHarmonixDataset

# Default paths for CLI argument defaults
DEFAULT_GRADIENT_PATH = str(BRAINHARMONIX_POS_EMBED_PATHS["gradient"])
DEFAULT_GEO_HARM_PATH = str(BRAINHARMONIX_POS_EMBED_PATHS["geo_harm"])
DEFAULT_HARMONIZER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["harmonizer"])
DEFAULT_FMRI_ENCODER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["fmri_encoder"])
DEFAULT_T1_ENCODER_CKPT = str(BRAINHARMONIX_CHECKPOINTS["t1_encoder"])


def extract_embeddings(
    dataloader: torch.utils.data.DataLoader,
    fmri_encoder: torch.nn.Module,
    t1_encoder: torch.nn.Module,
    harmonizer: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract embeddings from all three model components.

    Args:
        dataloader: DataLoader yielding batches with 'fmri', 't1', 'attn_mask', 'patch_size'
        fmri_encoder: Loaded fMRI encoder (fp32)
        t1_encoder: Loaded T1 encoder (fp16)
        harmonizer: Loaded harmonizer (fp16)
        device: Device to run inference on

    Returns:
        Tuple of (fmri_embeddings, t1_embeddings, harmonizer_embeddings)
    """
    all_fmri = []
    all_t1 = []
    all_harmonizer = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # fMRI encoder uses SDPA (fp32), T1/harmonizer use flash_attention_2 (fp16)
            fmri = batch["fmri"].to(device)
            attn_mask = batch["attn_mask"].to(device).bool()
            patch_size = batch["patch_size"][0].item()
            t1 = batch["t1"].to(device).half()

            # fMRI embedding: (B, 7200, 768)
            fmri_embed = fmri_encoder(fmri, patch_size, attention_mask=attn_mask)
            all_fmri.append(fmri_embed.cpu())

            # T1 embedding: (B, 1200, 768)
            t1_embed = t1_encoder(t1)
            all_t1.append(t1_embed.cpu().float())

            # Harmonizer: fuses fMRI + T1 → (B, 129, 768)
            # 129 = 1 CLS token + 128 latent tokens
            combined_embed = torch.cat([fmri_embed.half(), t1_embed], dim=1)
            # Note: harmonizer internally pads attn_mask for T1 (1200) and latent tokens (128)
            # So we pass only the fMRI attention mask (7200)
            harmonizer_embed, _ = harmonizer.forward_encoder(combined_embed, attn_mask)
            all_harmonizer.append(harmonizer_embed.cpu().float())

    return (
        torch.cat(all_fmri, dim=0),
        torch.cat(all_t1, dim=0),
        torch.cat(all_harmonizer, dim=0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from BrainHarmonix models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python extract_brainharmonix.py --dataset data/processed/dataset.arrow

    # Custom output directory
    python extract_brainharmonix.py --dataset data.arrow --output-dir outputs/

    # Custom checkpoint paths
    python extract_brainharmonix.py --dataset data.arrow \\
        --harmonizer-ckpt models/harmonizer.pth \\
        --fmri-ckpt models/harmonix-f.pth \\
        --t1-ckpt models/harmonix-s.pth
        """,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to Arrow dataset with fMRI and T1 data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/brainharmonix"),
        help="Output directory for embeddings (default: outputs/embeddings/brainharmonix)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="brainharmonix",
        help="Prefix for output files (default: brainharmonix)",
    )
    parser.add_argument(
        "--harmonizer-ckpt",
        type=Path,
        default=Path(DEFAULT_HARMONIZER_CKPT),
        help=f"Path to harmonizer checkpoint (default: {DEFAULT_HARMONIZER_CKPT})",
    )
    parser.add_argument(
        "--fmri-ckpt",
        type=Path,
        default=Path(DEFAULT_FMRI_ENCODER_CKPT),
        help=f"Path to fMRI encoder checkpoint (default: {DEFAULT_FMRI_ENCODER_CKPT})",
    )
    parser.add_argument(
        "--t1-ckpt",
        type=Path,
        default=Path(DEFAULT_T1_ENCODER_CKPT),
        help=f"Path to T1 encoder checkpoint (default: {DEFAULT_T1_ENCODER_CKPT})",
    )
    parser.add_argument(
        "--gradient-path",
        type=Path,
        default=Path(DEFAULT_GRADIENT_PATH),
        help=f"Path to gradient mapping CSV (default: {DEFAULT_GRADIENT_PATH})",
    )
    parser.add_argument(
        "--geo-harm-path",
        type=Path,
        default=Path(DEFAULT_GEO_HARM_PATH),
        help=f"Path to geometric harmonics CSV (default: {DEFAULT_GEO_HARM_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--is-finetuned",
        action="store_true",
        help="Whether to use fine-tuned models (default: False)",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")

    # Load models
    print("\nLoading models...")
    harmonizer = load_harmonizer(args.harmonizer_ckpt, device, mode="inference")
    fmri_encoder = load_fmri_encoder(args.fmri_ckpt, args.gradient_path, args.geo_harm_path, device, is_finetuned=args.is_finetuned)
    t1_encoder = load_t1_encoder(args.t1_ckpt, device, mode="inference", is_finetuned=args.is_finetuned)

    print("Models loaded successfully")

    # Load dataset
    print("\nLoading dataset...")
    dataset = BrainHarmonixDataset(str(args.dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    participant_ids = [d["participant_id"] for d in dataset]
    # Extract embeddings
    print("\nExtracting embeddings...")
    fmri_embeds, t1_embeds, harmonizer_embeds = extract_embeddings(
        dataloader, fmri_encoder, t1_encoder, harmonizer, device
    )

    print(f"\nfMRI embeddings shape: {fmri_embeds.shape}")
    print(f"T1 embeddings shape: {t1_embeds.shape}")
    print(f"Harmonizer embeddings shape: {harmonizer_embeds.shape}")

    # Save embeddings as npz files
    embedding_path = args.output_dir / f"{args.output_prefix}.embeddings.npz"

    np.savez(
        embedding_path,
        fmri=fmri_embeds.numpy(),
        t1=t1_embeds.numpy(),
        harmonizer=harmonizer_embeds.numpy(),
        participant_ids=participant_ids
    )

    print("\nSaved embeddings to:")
    print(f"  - {embedding_path}")


if __name__ == "__main__":
    main()
