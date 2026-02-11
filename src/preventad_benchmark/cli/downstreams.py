"""CLI entry point for downstream prediction experiments."""

import numpy as np
import torch
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure
from pathlib import Path

from preventad_benchmark.config import (
    A424_NPARCELS,
    BRAINHARMONIX_SCHAEFER_ROIS,
    BRAINHARMONIX_T1_TARGET_SHAPE,
    BRAINLM_TIMESERIES_LENGTH,
    EVALUATION_PCA_COMPONENTS,
    PATHS,
    PHENOTYPE_TARGETS,
)
from preventad_benchmark.models.brainharmonix.utils import crop_center
from preventad_benchmark.evaluation import (
    load_prediction_targets,
    run_downstream_experiment,
)


# Default embedding paths (relative to project root)
BRAINLM_BASELINE_FEAT = str(PATHS["outputs"] / "embeddings/brainlm.vitmae_650M.direct_transfer.gigaconnectome")
BRAINHARMONIX_DATASET = str(PATHS["processed"] / "dataset-preventad.fmri.zscored.gigaconnectome.schaefer400.arrow")
BRAINHARMONIX_EMB_DIR = str(PATHS["outputs"] / "embeddings/brainharmonix")
BRAINHARMONIX_FMRI_EMB = f"{BRAINHARMONIX_EMB_DIR}/brainharmonix.fmri_embeddings.pt"
BRAINHARMONIX_T1_EMB = f"{BRAINHARMONIX_EMB_DIR}/brainharmonix.t1_embeddings.pt"
BRAINHARMONIX_HARMONIZER_EMB = f"{BRAINHARMONIX_EMB_DIR}/brainharmonix.harmonizer_embeddings.pt"


def get_brainlm_baseline_data(timeseries_length=BRAINLM_TIMESERIES_LENGTH):
    """Load baseline features: flattened timeseries and functional connectivity."""
    features = load_from_disk(BRAINLM_BASELINE_FEAT)

    ts_flatten = [
        np.array(example).reshape(3, A424_NPARCELS, timeseries_length)[0].T.flatten()
        for example in features['padded_recording']
    ]

    correlation_baseline = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    ts = [
        np.array(example).reshape(3, A424_NPARCELS, timeseries_length)[0].T
        for example in features['padded_recording']
    ]
    fc = correlation_baseline.fit_transform(ts)
    return ts_flatten, fc


def brainlm_baseline_experiment() -> None:
    ts, fc = get_brainlm_baseline_data()
    labels = load_prediction_targets(BRAINLM_BASELINE_FEAT)
    output_dir = "outputs/downstreams/brainlm-baseline"

    # Timeseries and connectivity are high-dimensional -> use PCA
    for feat_data, feat_name in zip((ts, fc), ('timeseries', 'connectivity')):
        run_downstream_experiment(
            feat_data, labels, output_dir, feat_name,
            pca_components=EVALUATION_PCA_COMPONENTS,
        )


def brainlm_experiment(features_path: Path, feat_name: str) -> None:
    features = np.array(load_from_disk(features_path)[feat_name]).squeeze()
    labels = load_prediction_targets(features_path)
    output_dir = f"outputs/downstreams/{features_path.name}"
    prefix = feat_name.replace("_", "-")

    # Embeddings are low-dimensional -> no PCA
    run_downstream_experiment(features, labels, output_dir, prefix)


def brainharmonix_experiment(
    harmonizer_emb_path: str = None,
    output_dir: str = 'outputs/downstreams/brainharmonix/',
    fmri_emb_path: str = None,
    t1_emb_path: str = None,
) -> None:
    """Run downstream experiments with BrainHarmonix features."""
    features = {}

    fmri_path = fmri_emb_path or BRAINHARMONIX_FMRI_EMB
    if Path(fmri_path).exists():
        fmri_emb = torch.load(fmri_path, map_location="cpu", weights_only=True)
        features['fmri_mean'] = fmri_emb.mean(dim=1).numpy()

    t1_path = t1_emb_path or BRAINHARMONIX_T1_EMB
    if Path(t1_path).exists():
        t1_emb = torch.load(t1_path, map_location="cpu", weights_only=True)
        features['t1_mean'] = t1_emb.mean(dim=1).numpy()

    harmonizer_path = harmonizer_emb_path or BRAINHARMONIX_HARMONIZER_EMB
    harmonizer_emb = torch.load(harmonizer_path, map_location="cpu", weights_only=True)
    features['harmonizer_cls'] = harmonizer_emb[:, 0, :].numpy()
    features['harmonizer_latent_mean'] = harmonizer_emb[:, 1:, :].mean(dim=1).numpy()

    labels = load_prediction_targets(BRAINHARMONIX_DATASET)

    # Embeddings are 768-dim -> no PCA
    for feat_name, feat_data in features.items():
        print(f"Running experiments with {feat_name} features, shape: {feat_data.shape}")
        run_downstream_experiment(feat_data, labels, output_dir, feat_name)


def brainharmonix_baseline_experiment(
    output_dir: str = 'outputs/downstreams/brainharmonix-baseline/',
) -> None:
    """Run downstream experiments with BrainHarmonix baseline features.

    Extracts raw timeseries, functional connectivity, and T1 structural features
    from the BrainHarmonix Arrow dataset (Schaefer 400 parcellation).
    """
    dataset = load_from_disk(BRAINHARMONIX_DATASET)
    timeseries_length = BRAINLM_TIMESERIES_LENGTH  # 140

    # --- Timeseries features: flatten (T, 400) -> (400 * 140) ---
    ts_flatten = []
    ts_matrices = []
    for example in dataset:
        ts = np.array(example['raw_timeseries'], dtype=np.float32)
        # Crop to timeseries_length (take first 140 timepoints)
        ts = ts[:timeseries_length, :]
        ts_flatten.append(ts.T.flatten())  # (400, 140) -> 56000
        ts_matrices.append(ts)  # (140, 400) for connectivity

    # --- Connectivity features: correlation, vectorized ---
    correlation_measure = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    fc = correlation_measure.fit_transform(ts_matrices)

    # --- T1 features: load, z-score, center-crop, flatten ---
    t1_features = []
    for example in dataset:
        t1_filepath = example.get('t1_filepath')
        if t1_filepath and Path(t1_filepath).exists():
            t1_img = torch.load(t1_filepath, weights_only=True).numpy()
            # Z-score normalize on non-zero voxels
            mask = t1_img > 0
            if mask.sum() > 0:
                mean = t1_img[mask].mean()
                std = t1_img[mask].std()
                t1_img = (t1_img - mean) / (std + 1e-9)
                t1_img[~mask] = 0
            t1_cropped = crop_center(t1_img, BRAINHARMONIX_T1_TARGET_SHAPE)
            t1_features.append(t1_cropped.flatten())
        else:
            # Zero vector if T1 not available
            n_voxels = np.prod(BRAINHARMONIX_T1_TARGET_SHAPE)
            t1_features.append(np.zeros(n_voxels, dtype=np.float32))

    labels = load_prediction_targets(BRAINHARMONIX_DATASET)

    # Timeseries (56K features) -> PCA
    print("Running BrainHarmonix baseline: timeseries")
    run_downstream_experiment(
        ts_flatten, labels, output_dir, 'timeseries',
        pca_components=EVALUATION_PCA_COMPONENTS,
    )

    # Connectivity (79.8K features) -> no PCA
    print("Running BrainHarmonix baseline: connectivity")
    run_downstream_experiment(
        fc, labels, output_dir, 'connectivity',
    )

    # T1 (4.9M features) -> PCA
    print("Running BrainHarmonix baseline: t1")
    run_downstream_experiment(
        t1_features, labels, output_dir, 't1',
        pca_components=EVALUATION_PCA_COMPONENTS,
    )


def main():
    """CLI entry point for downstream experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run downstream prediction experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "brainlm-baseline", "brainharmonix-baseline", "brainlm", "brainharmonix", ],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--harmonizer-emb",
        type=str,
        default=None,
        help="Path to harmonizer embeddings .pt file (for brainharmonix experiment)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    if args.experiment in ["all", "brainlm-baseline"]:
        brainlm_baseline_experiment()

    if args.experiment in ["all", "brainharmonix-baseline"]:
        brainharmonix_baseline_experiment()

    if args.experiment in ["all", "brainharmonix"]:
        brainharmonix_experiment(
            harmonizer_emb_path=args.harmonizer_emb,
            output_dir=args.output_dir or 'outputs/downstreams/brainharmonix/',
        )

    if args.experiment in ["all", "brainlm"]:
        for p in Path("./outputs/").glob("*finetune*"):
            for feat_name in ['cls_token', 'cls_embedding']:
                brainlm_experiment(p, feat_name)

        for p in Path("./outputs/").glob("*direct*"):
            for feat_name in ['cls_token', 'cls_embedding']:
                brainlm_experiment(p, feat_name)


if __name__ == "__main__":
    main()
