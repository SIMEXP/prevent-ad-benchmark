"""Reusable sklearn pipelines for downstream evaluation."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR

from preventad_benchmark.config import EVALUATION_N_SPLITS, EVALUATION_TARGETS


SCORING = {
    "classification": {"acc": "accuracy", "auc": "roc_auc", "f1": "f1", "precision": "precision"},
    "regression": {
        "nrmse": "neg_root_mean_squared_error",
        "nmae": "neg_mean_absolute_error",
        "r2": "r2",
    }

}

N_JOBS = -1  # Use all available CPU cores for parallel processing

def valid_samples(features, labels, target_name):
    """Filter out samples with NaN/None labels and return aligned features and labels."""
    y = np.array(labels[target_name])

    valid_mask = np.array([
        v is not None and v != 'nan' and not (isinstance(v, float) and np.isnan(v))
        for v in y
    ])

    if not valid_mask.any():
        return None, None

    x_valid = features[valid_mask]
    y_valid = y[valid_mask].tolist()
    return x_valid, y_valid


def _encode_labels(y):
    """Encode string labels to integers if needed. Returns (y, is_classification)."""
    if isinstance(y[0], str):
        le = LabelEncoder()
        return le.fit_transform(y), True, le
    return np.asarray(y), False, None


def _stratify_labels(y, is_clf):
    """Return discrete labels for StratifiedShuffleSplit."""
    if is_clf:
        return y
    # Bin continuous values into 5 quantile groups for stratification
    percentiles = np.percentile(y, [20, 40, 60, 80])
    return np.digitize(y, percentiles)


def _build_steps(estimator, pca_components=None):
    """Build pipeline steps: always with robust scalar, optionally PCA, then estimator."""
    steps = [("scaler", RobustScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components)))
    steps.append(("estimator", estimator))
    return steps


def svm_pipeline(x, y, n_splits=EVALUATION_N_SPLITS, pca_components=None):
    """Run SVM cross-validation (SVC for classification, SVR for regression).

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        n_splits: Number of cross-validation splits.
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Set to EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
    """
    y, is_clf, _= _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(SVC(C=1, class_weight="balanced"), pca_components))
        scoring = "classification"
    else:
        pipe = Pipeline(_build_steps(SVR(), pca_components))
        scoring ="regression"

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    strat_labels = _stratify_labels(y, is_clf)
    return cross_validate(pipe, x, y, cv=cv.split(x, strat_labels), scoring=SCORING[scoring], n_jobs=N_JOBS)


def linear_pipeline(x, y, n_splits=EVALUATION_N_SPLITS, pca_components=None):
    """Run linear cross-validation (LogisticRegression for classification, LinearRegression for regression).

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        n_splits: Number of cross-validation splits.
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Set to EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
    """
    y, is_clf, _= _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(LogisticRegression(), pca_components))
        scoring = "classification"
    else:
        pipe = Pipeline(_build_steps(LinearRegression(), pca_components))
        scoring ="regression"

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    strat_labels = _stratify_labels(y, is_clf)
    return cross_validate(pipe, x, y, cv=cv.split(x, strat_labels), scoring=SCORING[scoring], n_jobs=N_JOBS)

def dummy_pipeline(x, y, n_splits=EVALUATION_N_SPLITS, pca_components=None):
    """Run dummy cross-validation (DummyClassifier for classification, DummyRegressor for regression).

    Provides a chance-level baseline to verify that real models learn meaningful patterns.

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        n_splits: Number of cross-validation splits.
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Set to EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
    """
    y, is_clf, _ = _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(DummyClassifier(strategy="most_frequent"), pca_components))
        scoring = "classification"
    else:
        pipe = Pipeline(_build_steps(DummyRegressor(strategy="mean"), pca_components))
        scoring = "regression"

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    strat_labels = _stratify_labels(y, is_clf)
    return cross_validate(pipe, x, y, cv=cv.split(x, strat_labels), scoring=SCORING[scoring], n_jobs=N_JOBS)


def baseline_pipeline(features, labels, output_dir, prefix, pca_components=None):
    """Run SVM + linear pipelines for all targets and save results.

    Args:
        features: (N, D) array of feature vectors.
        labels: dict mapping target name -> label array (from load_prediction_targets).
        output_dir: Directory to write result TSVs.
        prefix: Feature name prefix for output filenames (e.g. 'dummy', 'cls-token', 'connectivity').
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Pass EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
        run_dummy: Run dummy classifier.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = np.array(features)

    for target_name in EVALUATION_TARGETS:
        x, y = valid_samples(features, labels, target_name)
        if prefix == 'dummy':
            # Dummy pipeline
            dummy_path = output_dir / f"x-{prefix}_y-{target_name}_dummy_prediction.tsv"
            if dummy_path.exists():
                print(f"{dummy_path} exists, skip")
            else:
                print(f"  Running dummy classifier for {target_name}...")
                dummy_scores = dummy_pipeline(x, y, pca_components=pca_components)
                pd.DataFrame(dummy_scores).to_csv(dummy_path, sep="\t")
            continue

        # SVM pipeline
        svm_path = output_dir / f"x-{prefix}_y-{target_name}_svm_prediction.tsv"
        if svm_path.exists():
            print(f"{svm_path} exists, skip")
        else:
            print(f"  Running SVM for {prefix} -> {target_name}...")
            svm_scores = svm_pipeline(x, y, pca_components=pca_components)
            pd.DataFrame(svm_scores).to_csv(svm_path, sep="\t")

        # Linear pipeline
        linear_path = output_dir / f"x-{prefix}_y-{target_name}_linear_prediction.tsv"
        if linear_path.exists():
            print(f"{linear_path} exists, skip")
        else:
            print(f"  Running Linear for {prefix} -> {target_name}...")
            linear_scores = linear_pipeline(x, y, pca_components=pca_components)
            pd.DataFrame(linear_scores).to_csv(linear_path, sep="\t")


def _score_predictions(y_true, y_pred, is_clf):
    """Compute evaluation metrics for a single train/test fit.

    Returns dict with accuracy/AUC/F1 for classification, or
    negative RMSE/MAE and R² for regression (sign convention matches sklearn).
    """
    if is_clf:
        return {
            "test_acc": accuracy_score(y_true, y_pred),
            "test_auc": roc_auc_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred),
        }
    return {
        "test_nrmse": -np.sqrt(mean_squared_error(y_true, y_pred)),
        "test_nmae": -mean_absolute_error(y_true, y_pred),
        "test_r2": r2_score(y_true, y_pred),
    }


def svm_fit_score(x_train, y_train, x_test, y_test, pca_components=None):
    """Fit SVM and score on one fold of data.

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        pca_components: Number of PCA components. None skips PCA.
    """
    y_train, is_clf, le = _encode_labels(y_train)

    if is_clf:
        pipe = Pipeline(_build_steps(SVC(C=1, class_weight="balanced"), pca_components))
        y_test = le.transform(y_test)
    else:
        pipe = Pipeline(_build_steps(SVR(), pca_components))
        y_test = np.asarray(y_test)

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return _score_predictions(y_test, y_pred, is_clf)


def dummy_fit_score(x_train, y_train, x_test, y_test, pca_components=None):
    """Fit dummy model and score on one fold of data.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels. String labels trigger classification, numeric triggers regression.
        x_test: Test feature matrix.
        y_test: Test labels.
        pca_components: Number of PCA components. None skips PCA.
    """
    y_train, is_clf, le = _encode_labels(y_train)

    if is_clf:
        pipe = Pipeline(_build_steps(DummyClassifier(strategy="most_frequent"), pca_components))
        y_test = le.transform(y_test)
    else:
        pipe = Pipeline(_build_steps(DummyRegressor(strategy="mean"), pca_components))
        y_test = np.asarray(y_test)

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return _score_predictions(y_test, y_pred, is_clf)


def linear_fit_score(x_train, y_train, x_test, y_test, pca_components=None):
    """Fit linear model and score on one fold of data.

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        pca_components: Number of PCA components. None skips PCA.
    """
    y_train, is_clf, le = _encode_labels(y_train)

    if is_clf:
        pipe = Pipeline(_build_steps(LogisticRegression(), pca_components))
        y_test = le.transform(y_test)
    else:
        pipe = Pipeline(_build_steps(LinearRegression(), pca_components))
        y_test = np.asarray(y_test)

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return _score_predictions(y_test, y_pred, is_clf)
