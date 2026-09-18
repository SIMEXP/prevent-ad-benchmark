"""Per-target classification summary bar charts.

Ported from a standalone reference script (plot_classification_summary.py) that
read summary_classification.tsv and regex-parsed its `ACCURACY`/`PRECISION`
columns (formatted as "mean [low high]") to recover the CI bounds. Now that
plotting.utils.make_summary_table writes the mean and CI bounds as separate
numeric columns directly, that parsing step is gone -- everything else
(filtering, labeling, coloring, layout) is unchanged from the original.

Filtering:
  - drop SVM classifiers (keep LINEAR and DUMMY only)
  - for Variation == "baseline", keep only Atlas == "Schaefer400"

Labeling:
  - "linear" is dropped from labels (all non-dummy rows are the linear classifier)
  - dummy classifier is labeled "(chance level)"
  - for brainharmonix, T1 (mean) drops its nozscore/zscore qualifier (T1 isn't
    fMRI data, so normalization doesn't apply to it) and keeps whichever of the
    two duplicate rows has the higher value for the metric being plotted

Coloring (per metric panel, i.e. independently for accuracy and precision):
  - Dummy classifier bar: special color
  - Functional connectivity (baseline) bar: special color
  - Timeseries, top 75 PCs (baseline) bar: special color
  - everything else, relative to the dummy/FC reference values for that metric:
      - below the lower of {dummy, functional connectivity}: light grey
      - between them: darker grey
      - above the higher of {dummy, functional connectivity}: green

Rows are ranked by accuracy (descending) and that order is shared by both the
accuracy (top) and precision (bottom) panels.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLOR_DUMMY = "#4a3aa7"  # violet -- special: dummy classifier
COLOR_FC = "#2a78d6"  # blue -- special: functional connectivity (baseline)
COLOR_TIMESERIES = "#eda100"  # yellow -- special: timeseries (baseline)
COLOR_BELOW = "#d9d8d3"  # light grey -- below dummy & FC
COLOR_BETWEEN = "#8a8983"  # darker grey -- between dummy & FC
COLOR_ABOVE = "#1baf7a"  # green -- above dummy & FC
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"

NO_PRECISION_TARGETS = {"Sex", "Age (binary)"}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to LINEAR/DUMMY classifiers and Schaefer400-only baselines, and
    expose the metric/CI columns under the lowercase names the rest of this
    module expects.
    """
    df = df.copy()
    for metric in ("ACCURACY", "PRECISION"):
        key = metric.lower()
        df[key] = df[metric]
        df[f"{key}_low"] = df[f"{metric}_CI_LOW"]
        df[f"{key}_high"] = df[f"{metric}_CI_HIGH"]

    df = df[df["Classifier"] != "SVM"]
    df = df[~((df["Variation"] == "baseline") & (df["Atlas"] != "Schaefer400"))]
    return df


def label_for(row):
    if row["Feature"] == "dummy":
        return "Dummy classifier (chance level)"
    if row["Variation"] == "baseline" and row["Feature"] == "Functional connectivity":
        return "Functional connectivity (baseline)"
    if row["Variation"] == "baseline" and row["Feature"] == "Timeseries, top 75 PCs":
        return "Timeseries, top 75 PCs (baseline)"
    if row["Feature"] == "T1 (mean)":
        # T1 isn't fMRI data, so the fMRI z-score/no-zscore normalization
        # variants are duplicates -- normalization doesn't apply to it.
        return f"{row['Foundation Model']} · {row['Feature']}"
    return f"{row['Foundation Model']} · {row['Variation']} · {row['Feature']}"


def dedupe_by_label(model_sub, metric):
    """When rows collapse to the same label (e.g. T1 (mean) under different
    normalizations), keep only the one with the higher value for this metric."""
    labeled = model_sub.assign(_label=model_sub.apply(label_for, axis=1))
    keep_idx = labeled.groupby("_label")[metric].idxmax()
    return labeled.loc[keep_idx].drop(columns="_label").reset_index(drop=True)


def bucket_color(value, low, high):
    if value < low:
        return COLOR_BELOW
    if value > high:
        return COLOR_ABOVE
    return COLOR_BETWEEN


def tie_priority(row):
    """Lower sorts first (higher on chart) among rows tied on a metric value."""
    if row["Variation"] == "baseline" and row["Feature"] == "Functional connectivity":
        return 1
    if row["Feature"] == "dummy":
        return 2
    if row["Variation"] == "baseline" and row["Feature"] == "Timeseries, top 75 PCs":
        return 3
    return 0


def with_shared_baselines(model_sub, model, *special_rows):
    """Ensure the dummy / FC / timeseries reference rows are present, borrowing
    them from whichever foundation model they actually belong to if needed."""
    missing = [row for row in special_rows if row["Foundation Model"] != model]
    if not missing:
        return model_sub
    return pd.concat([model_sub, pd.DataFrame(missing)], ignore_index=True)


def colors_for_metric(sub, metric, dummy_val, fc_val):
    low, high = sorted((dummy_val, fc_val))
    colors = []
    for _, row in sub.iterrows():
        if row["Feature"] == "dummy":
            colors.append(COLOR_DUMMY)
        elif row["Variation"] == "baseline" and row["Feature"] == "Functional connectivity":
            colors.append(COLOR_FC)
        elif row["Variation"] == "baseline" and row["Feature"] == "Timeseries, top 75 PCs":
            colors.append(COLOR_TIMESERIES)
        else:
            colors.append(bucket_color(row[metric], low, high))
    return colors


def plot_classification_summary(df: pd.DataFrame, output_dir: Path):
    """Write one bar-chart figure per (target, foundation model) into output_dir.

    `df` is the DataFrame returned by plotting.utils.make_summary_table (or
    summary_classification.tsv read back in) -- must have the Foundation Model/
    Atlas/Variation/Feature/Target/Classifier columns plus ACCURACY/PRECISION
    and their _CI_LOW/_CI_HIGH counterparts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare(df)

    for raw_target, sub in df.groupby("Target"):
        target = raw_target.replace("β", "beta")

        dummy_row = sub[sub["Feature"] == "dummy"].iloc[0]
        fc_row = sub[
            (sub["Variation"] == "baseline") & (sub["Feature"] == "Functional connectivity")
        ].iloc[0]
        timeseries_row = sub[
            (sub["Variation"] == "baseline") & (sub["Feature"] == "Timeseries, top 75 PCs")
        ].iloc[0]

        models = sorted(sub["Foundation Model"].unique())
        safe_target = re.sub(r"[^\w]+", "_", target).strip("_").lower()

        metrics = (
            [("accuracy", dummy_row["accuracy"], fc_row["accuracy"], "Accuracy")]
            if raw_target in NO_PRECISION_TARGETS
            else [
                ("accuracy", dummy_row["accuracy"], fc_row["accuracy"], "Accuracy"),
                ("precision", dummy_row["precision"], fc_row["precision"], "Precision"),
            ]
        )

        for model in models:
            model_sub = with_shared_baselines(
                sub[sub["Foundation Model"] == model], model, dummy_row, fc_row, timeseries_row
            )
            n = len(model_sub)

            fig, axes = plt.subplots(
                1, len(metrics), figsize=(5.5 * len(metrics), max(0.28 * n, 4) + 1.5)
            )
            axes = [axes] if len(metrics) == 1 else axes
            fig.patch.set_alpha(0.0)

            for ax, (metric, dummy_val, fc_val, title) in zip(axes, metrics):
                deduped = dedupe_by_label(model_sub, metric)
                priority = deduped.apply(tie_priority, axis=1)
                ranked = (
                    deduped.assign(_prio=priority)
                    .sort_values([metric, "_prio"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                labels = [label_for(row) for _, row in ranked.iterrows()]
                values = ranked[metric].tolist()
                colors = colors_for_metric(ranked, metric, dummy_val, fc_val)

                is_dummy = ranked["Feature"] == "dummy"
                err_low = (ranked[metric] - ranked[f"{metric}_low"]).where(~is_dummy, 0.0)
                err_high = (ranked[f"{metric}_high"] - ranked[metric]).where(~is_dummy, 0.0)
                xerr = [err_low.tolist(), err_high.tolist()]

                y = range(len(ranked))
                ax.patch.set_alpha(0.0)
                ax.barh(
                    list(y),
                    values,
                    height=0.7,
                    color=colors,
                    xerr=xerr,
                    error_kw=dict(ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2.5, capthick=1),
                )
                ax.set_xlim(0, 1.08)
                ax.set_xlabel(title, color=TEXT_SECONDARY)
                ax.tick_params(colors=TEXT_SECONDARY)
                ax.grid(axis="x", color=GRID_COLOR, linewidth=1, zorder=0)
                ax.set_axisbelow(True)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                ax.spines["bottom"].set_color(GRID_COLOR)

                ax.invert_yaxis()  # highest score at top
                ax.set_yticks(list(y))
                ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=7.5)

            handles = [
                plt.Rectangle((0, 0), 1, 1, color=COLOR_FC),
                plt.Rectangle((0, 0), 1, 1, color=COLOR_DUMMY),
                plt.Rectangle((0, 0), 1, 1, color=COLOR_TIMESERIES),
                plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE),
                plt.Rectangle((0, 0), 1, 1, color=COLOR_BETWEEN),
                plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW),
            ]
            fig.legend(
                handles,
                [
                    "Functional connectivity (baseline)",
                    "Dummy classifier (chance level)",
                    "Timeseries, top 75 PCs (baseline)",
                    "Above dummy & FC",
                    "Between dummy & FC",
                    "Below dummy & FC",
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=3,
                frameon=False,
                labelcolor=TEXT_PRIMARY,
                fontsize=8.5,
            )

            fig.suptitle(f"{target} — {model}", color=TEXT_PRIMARY, fontsize=15, y=1.04)

            fig.tight_layout()
            safe_model = re.sub(r"[^\w]+", "_", model).strip("_").lower()
            out_path = output_dir / f"{safe_target}_{safe_model}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
            plt.close(fig)
            print(f"wrote {out_path}")
