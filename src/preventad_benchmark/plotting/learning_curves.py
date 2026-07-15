"""Learning curve plotting utilities for BrainLM and BrainHarmony finetuning."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_brainharmony_curves(finetune_dir: Path) -> pd.DataFrame:
    """Load per-epoch train/val loss for all BrainHarmony splits and conditions.

    Expected layout::

        finetune_dir/
          {condition}/
            {split_idx}/
              config.json   ← has "metrics": [{epoch, train_loss, val_loss}, ...]

    Returns a long-form DataFrame with columns:
        condition, split, epoch, train_loss, val_loss
    """
    records = []
    finetune_dir = Path(finetune_dir)

    for cond_dir in sorted(finetune_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name  # e.g. "zscore.self-supervised"

        for split_dir in sorted(cond_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not split_dir.is_dir():
                continue
            config_file = split_dir / "config.json"
            if not config_file.exists():
                continue

            with open(config_file) as f:
                cfg = json.load(f)

            split_idx = int(split_dir.name)
            for entry in cfg.get("metrics", []):
                records.append(
                    {
                        "condition": condition,
                        "split": split_idx,
                        "epoch": entry["epoch"],
                        "train_loss": entry["train_loss"],
                        "val_loss": entry["val_loss"],
                    }
                )

    return pd.DataFrame(records)


def load_brainlm_curves(finetune_dir: Path) -> pd.DataFrame:
    """Load per-epoch training loss for all BrainLM splits and conditions.

    Expected layout::

        finetune_dir/
          {condition}/           e.g. "zscore_brainlm.650M.selfsupervised"
            split{N}/
              trainer_state.json ← log_history with per-step loss

    Steps are averaged within each integer epoch so the result aligns with
    BrainHarmony's epoch-level granularity.

    Returns a long-form DataFrame with columns:
        condition, split, epoch, train_loss
    (no val_loss — validation was not logged during BrainLM training)
    """
    records = []
    finetune_dir = Path(finetune_dir)

    for cond_dir in sorted(finetune_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name  # e.g. "zscore_brainlm.650M.selfsupervised"

        for split_dir in sorted(
            cond_dir.iterdir(),
            key=lambda p: int(m.group(1)) if (m := re.match(r"split(\d+)$", p.name)) else -1,
        ):
            if not split_dir.is_dir():
                continue
            state_file = split_dir / "trainer_state.json"
            if not state_file.exists():
                continue

            with open(state_file) as f:
                state = json.load(f)

            # Extract split index from directory name "split{N}"
            m = re.match(r"split(\d+)$", split_dir.name)
            if m is None:
                continue
            split_idx = int(m.group(1))

            # Collect per-step training losses (skip the final summary entry)
            step_rows = [
                e for e in state["log_history"]
                if "loss" in e and "eval_loss" not in e and "train_loss" not in e
            ]

            if not step_rows:
                continue

            # Average loss within each integer epoch
            df_steps = pd.DataFrame(step_rows)[["epoch", "loss"]]
            df_steps["epoch_int"] = df_steps["epoch"].apply(
                lambda e: int(np.ceil(e))  # ceil so epoch 0.01 → epoch 1
            )
            df_epoch = (
                df_steps.groupby("epoch_int")["loss"]
                .mean()
                .reset_index()
                .rename(columns={"epoch_int": "epoch", "loss": "train_loss"})
            )

            for _, row in df_epoch.iterrows():
                records.append(
                    {
                        "condition": condition,
                        "split": split_idx,
                        "epoch": int(row["epoch"]),
                        "train_loss": row["train_loss"],
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Condition label helpers
# ---------------------------------------------------------------------------

_BRAINHARMONY_LABELS = {
    "zscore.self-supervised": "z-score",
    "nozscore.self-supervised": "no z-score",
}

_BRAINLM_LABELS = {
    "zscore_brainlm.650M.selfsupervised": "z-score  ·  BrainLM atlas",
    "zscore_gigaconnectome.650M.selfsupervised": "z-score  ·  Giga atlas",
    "nozscore_brainlm.650M.selfsupervised": "no z-score  ·  BrainLM atlas",
    "nozscore_gigaconnectome.650M.selfsupervised": "no z-score  ·  Giga atlas",
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_mean_band(ax, df, x_col, y_col, label, color, linestyle="-"):
    """Plot mean ± 95 % CI across splits."""
    grouped = df.groupby(x_col)[y_col]
    mean = grouped.mean()
    lo = grouped.quantile(0.025)
    hi = grouped.quantile(0.975)
    ax.plot(mean.index, mean.values, linestyle=linestyle, color=color, label=label, linewidth=1.8)
    ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_brainharmony_curves(
    finetune_dir: Path,
    output_path: Path | None = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Plot BrainHarmony learning curves (train + val loss per epoch).

    One panel per condition, shaded band = 95 % CI across splits.
    """
    df = load_brainharmony_curves(finetune_dir)
    if df.empty:
        raise ValueError(f"No BrainHarmony data found in {finetune_dir}")

    conditions = sorted(df["condition"].unique())
    colors = {"train_loss": "#1f77b4", "val_loss": "#ff7f0e"}

    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        label = _BRAINHARMONY_LABELS.get(cond, cond)
        _plot_mean_band(ax, sub, "epoch", "train_loss",
                        label="Train loss", color=colors["train_loss"])
        _plot_mean_band(ax, sub, "epoch", "val_loss",
                        label="Val loss", color=colors["val_loss"], linestyle="--")
        ax.set_title(f"BrainHarmony\n{label}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Loss")
    fig.suptitle("BrainHarmony — finetuning learning curves", fontsize=12, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


def plot_brainlm_curves(
    finetune_dir: Path,
    output_path: Path | None = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot BrainLM training loss curves (no val loss available).

    Two panels: z-score vs no-z-score, two lines each (atlas variant).
    Shaded band = 95 % CI across splits.
    """
    df = load_brainlm_curves(finetune_dir)
    if df.empty:
        raise ValueError(f"No BrainLM data found in {finetune_dir}")

    # Group conditions into zscore / nozscore panels
    panel_groups = {
        "z-score": [c for c in df["condition"].unique() if c.startswith("zscore")],
        "no z-score": [c for c in df["condition"].unique() if c.startswith("nozscore")],
    }
    panel_groups = {k: v for k, v in panel_groups.items() if v}  # drop empty

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, len(panel_groups), figsize=figsize, sharey=True)
    if len(panel_groups) == 1:
        axes = [axes]

    for ax, (panel_name, conds) in zip(axes, panel_groups.items()):
        for color, cond in zip(palette, sorted(conds)):
            sub = df[df["condition"] == cond]
            label = _BRAINLM_LABELS.get(cond, cond).split("·")[-1].strip()
            _plot_mean_band(ax, sub, "epoch", "train_loss", label=label, color=color)
        ax.set_title(f"BrainLM — {panel_name}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.98, 0.97, "Training loss only\n(no val loss logged)",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="gray", style="italic",
        )

    axes[0].set_ylabel("Loss")
    fig.suptitle("BrainLM — finetuning learning curves", fontsize=12, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


def plot_combined_curves(
    brainharmony_finetune_dir: Path,
    brainlm_finetune_dir: Path,
    output_path: Path | None = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Single figure with BrainHarmony (top row) and BrainLM (bottom row)."""
    bh_df = load_brainharmony_curves(brainharmony_finetune_dir)
    bl_df = load_brainlm_curves(brainlm_finetune_dir)

    bh_conditions = sorted(bh_df["condition"].unique())
    bl_panel_groups = {
        "z-score": sorted(c for c in bl_df["condition"].unique() if c.startswith("zscore")),
        "no z-score": sorted(c for c in bl_df["condition"].unique() if c.startswith("nozscore")),
    }
    bl_panel_groups = {k: v for k, v in bl_panel_groups.items() if v}

    n_cols = max(len(bh_conditions), len(bl_panel_groups))
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)

    bh_colors = {"train_loss": "#1f77b4", "val_loss": "#ff7f0e"}
    bl_palette = ["#1f77b4", "#d62728"]

    # --- BrainHarmony (row 0) ---
    for col, cond in enumerate(bh_conditions):
        ax = axes[0][col]
        sub = bh_df[bh_df["condition"] == cond]
        label = _BRAINHARMONY_LABELS.get(cond, cond)
        _plot_mean_band(ax, sub, "epoch", "train_loss",
                        "Train loss", bh_colors["train_loss"])
        _plot_mean_band(ax, sub, "epoch", "val_loss",
                        "Val loss", bh_colors["val_loss"], linestyle="--")
        ax.set_title(f"BrainHarmony\n{label}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Loss")

    # Hide unused BrainHarmony panels
    for col in range(len(bh_conditions), n_cols):
        axes[0][col].set_visible(False)

    # --- BrainLM (row 1) ---
    for col, (panel_name, conds) in enumerate(bl_panel_groups.items()):
        ax = axes[1][col]
        for color, cond in zip(bl_palette, conds):
            sub = bl_df[bl_df["condition"] == cond]
            label = _BRAINLM_LABELS.get(cond, cond).split("·")[-1].strip()
            _plot_mean_band(ax, sub, "epoch", "train_loss", label, color)
        ax.set_title(f"BrainLM — {panel_name}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.98, 0.97, "Training loss only",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="gray", style="italic",
        )
        if col == 0:
            ax.set_ylabel("Loss")

    # Hide unused BrainLM panels
    for col in range(len(bl_panel_groups), n_cols):
        axes[1][col].set_visible(False)

    fig.suptitle("Finetuning learning curves", fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig
