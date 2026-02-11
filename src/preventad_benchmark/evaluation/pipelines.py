"""Reusable sklearn pipelines for downstream evaluation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC, SVR

from preventad_benchmark.config import EVALUATION_N_SPLITS


def _encode_labels(y):
    """Encode string labels to integers if needed. Returns (y, is_classification)."""
    if isinstance(y[0], str):
        le = LabelEncoder()
        return le.fit_transform(y), True
    return np.asarray(y), False


def _classification_scoring():
    return {"acc": "accuracy", "auc": "roc_auc", "f1": "f1"}


def _regression_scoring():
    return {
        "nrmse": "neg_root_mean_squared_error",
        "nmae": "neg_mean_absolute_error",
        "r2": "r2",
    }


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
    y, is_clf = _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(SVC(C=1, class_weight="balanced"), pca_components))
        scoring = _classification_scoring()
        cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=1)
    else:
        pipe = Pipeline(_build_steps(SVR(), pca_components))
        scoring = _regression_scoring()
        cv = ShuffleSplit(n_splits=n_splits, random_state=1)

    return cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=-1)


def linear_pipeline(x, y, n_splits=EVALUATION_N_SPLITS, pca_components=None):
    """Run linear cross-validation (LogisticRegression for classification, LinearRegression for regression).

    Args:
        x: Feature matrix (N, D).
        y: Labels. String labels trigger classification, numeric triggers regression.
        n_splits: Number of cross-validation splits.
        pca_components: Number of PCA components. None skips PCA (use for embeddings).
            Set to EVALUATION_PCA_COMPONENTS for high-dimensional timeseries.
    """
    y, is_clf = _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(LogisticRegression(), pca_components))
        scoring = _classification_scoring()
        cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=1)
    else:
        pipe = Pipeline(_build_steps(LinearRegression(), pca_components))
        scoring = _regression_scoring()
        cv = ShuffleSplit(n_splits=n_splits, random_state=1)

    return cross_validate(pipe, x, y, cv=cv, scoring=scoring, n_jobs=-1)
