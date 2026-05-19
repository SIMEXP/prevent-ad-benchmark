"""Generic downstream experiment runner."""

from pathlib import Path
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure

import numpy as np
import pandas as pd

from preventad_benchmark.config import EVALUATION_TARGETS, TIMESERIES_LENGTH, EVALUATION_PCA_COMPONENTS
from preventad_benchmark.evaluation.pipelines import baseline_pipeline, svm_fit_score, linear_fit_score, valid_samples
from preventad_benchmark.evaluation.targets import load_prediction_targets
from preventad_benchmark.plotting.utils import TARGET_NAMES


def run_baseline_experiment(input_dir, output_dir):
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
    print("Running baseline: timeseries")
    baseline_pipeline(
        ts_flatten, labels, output_dir, 'timeseries',
        pca_components=EVALUATION_PCA_COMPONENTS,
    )
    print("Running baseline: dummy classifier")
    baseline_pipeline(
        ts_flatten, labels, output_dir, 'dummy',
        pca_components=EVALUATION_PCA_COMPONENTS,
    )
    # Connectivity -> no PCA
    print("Running baseline: connectivity")
    baseline_pipeline(
        fc, labels, output_dir, 'connectivity'
    )


def run_foundation_model_experiment(train_features, train_labels, test_features, test_labels, prefix, pca_components=None):
    """Fit SVM + linear on test-set embeddings and score.

    Args:
        features: (N, D) array of feature vectors.
        labels: dict mapping target name -> label array (from load_prediction_targets).
        output_dir: Directory to write result TSVs.
        prefix: Feature name prefix for output filenames.
        pca_components: Number of PCA components. None skips PCA.
    """

    train_features = np.array(train_features)
    test_features = np.array(test_features)

    all_results = []
    for target_name in EVALUATION_TARGETS:
        x_train, y_train = valid_samples(train_features, train_labels, target_name)
        x_test, y_test = valid_samples(test_features, test_labels, target_name)

        if y_train is None or y_test is None:
            print(f"  Skipping {target_name}: all labels are NaN")
            continue

        print(f"  Running {prefix} -> {target_name}...")
        svm_scores = svm_fit_score(x_train, y_train, x_test, y_test, pca_components=pca_components)
        linear_scores = linear_fit_score(x_train, y_train, x_test, y_test, pca_components=pca_components)
        results = pd.DataFrame([svm_scores, linear_scores])
        results["Classifier"] = ["SVM", "Linear"]
        results["Target"] = TARGET_NAMES[target_name]
        all_results.append(results)
    return pd.concat(all_results).reset_index(drop=True)

