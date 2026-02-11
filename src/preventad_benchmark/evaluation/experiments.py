"""Generic downstream experiment runner."""

from pathlib import Path
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure

import numpy as np
import pandas as pd

from preventad_benchmark.config import EVALUATION_TARGETS, TIMESERIES_LENGTH, EVALUATION_PCA_COMPONENTS
from preventad_benchmark.evaluation.pipelines import linear_pipeline, svm_pipeline
from preventad_benchmark.evaluation.targets import load_prediction_targets


def baseline_experiment(input_dir, output_dir):
    """Run downstream experiment with baseline features.
    Extracts raw timeseries, functional connectivity from the Arrow dataset.

    Args:
        input_dir: path of arrow dataset.
        output_dir: output path of experiment results.
    """
    # generate baseline features
    dataset = load_from_disk(input_dir)
    timeseries_length = TIMESERIES_LENGTH  # 140

    # Timeseries features: flatten
    ts_flatten = []
    ts_matrices = []
    for example in dataset:
        ts = np.array(example['raw_timeseries'], dtype=np.float32)
        # Crop to timeseries_length (take first 140 timepoints)
        ts = ts[:timeseries_length, :]
        ts_flatten.append(ts.T.flatten())  
        ts_matrices.append(ts) 

    correlation_measure = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    fc = correlation_measure.fit_transform(ts_matrices)
    # the loaded labels will have the same order as the feature
    labels = load_prediction_targets(input_dir)  
    
    # Timeseries -> PCA
    print("Running BrainHarmonix baseline: timeseries")
    run_downstream_experiment(
        ts_flatten, labels, output_dir, 'timeseries',
        pca_components=EVALUATION_PCA_COMPONENTS,
    )

    # Connectivity -> no PCA
    print("Running BrainHarmonix baseline: connectivity")
    run_downstream_experiment(
        fc, labels, output_dir, 'connectivity',
    )


def run_downstream_experiment(features, labels, output_dir, prefix, pca_components=None):
    """Run SVM + linear pipelines for all targets and save results.

    Args:
        features: (N, D) array of feature vectors.
        labels: dict mapping target name -> label array (from load_prediction_targets).
        output_dir: Directory to write result TSVs.
        prefix: Feature name prefix for output filenames (e.g. 'cls-token', 'connectivity').
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Pass EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = np.array(features)

    for target_name in EVALUATION_TARGETS:
        y = np.array(labels[target_name])

        # Filter out samples with NaN labels
        if isinstance(y[0], str):
            valid_mask = np.array([v != 'nan' for v in y])
        else:
            valid_mask = np.array([bool(not np.isnan(v)) for v in y])

        if not valid_mask.any():
            print(f"  Skipping {target_name}: all labels are NaN")
            continue
        X_valid = features[valid_mask]
        y_valid = y[valid_mask].tolist()

        # SVM pipeline
        svm_path = output_dir / f"x-{prefix}_y-{target_name}_svm_prediction.tsv"
        if svm_path.exists():
            print(f"{svm_path} exists, skip")
        else:
            print(f"  Running SVM for {prefix} -> {target_name}...")
            svm_scores = svm_pipeline(X_valid, y_valid, pca_components=pca_components)
            pd.DataFrame(svm_scores).to_csv(svm_path, sep="\t")

        # Linear pipeline
        linear_path = output_dir / f"x-{prefix}_y-{target_name}_linear_prediction.tsv"
        if linear_path.exists():
            print(f"{linear_path} exists, skip")
        else:
            print(f"  Running Linear for {prefix} -> {target_name}...")
            linear_scores = linear_pipeline(X_valid, y_valid, pca_components=pca_components)
            pd.DataFrame(linear_scores).to_csv(linear_path, sep="\t")
