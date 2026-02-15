"""Reusable sklearn pipelines for downstream evaluation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC, SVR

from preventad_benchmark.config import EVALUATION_N_SPLITS


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
    y, is_clf, _= _encode_labels(y)

    if is_clf:
        pipe = Pipeline(_build_steps(SVC(C=1, class_weight="balanced"), pca_components))
        scoring = _classification_scoring()
    else:
        pipe = Pipeline(_build_steps(SVR(), pca_components))
        scoring = _regression_scoring()

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    strat_labels = _stratify_labels(y, is_clf)
    return cross_validate(pipe, x, y, cv=cv.split(x, strat_labels), scoring=scoring, n_jobs=-1)


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
        scoring = _classification_scoring()
    else:
        pipe = Pipeline(_build_steps(LinearRegression(), pca_components))
        scoring = _regression_scoring()

    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    strat_labels = _stratify_labels(y, is_clf)
    return cross_validate(pipe, x, y, cv=cv.split(x, strat_labels), scoring=scoring, n_jobs=-1)


def _score_predictions(y_true, y_pred, is_clf):
    """Compute metrics for a single fit."""
    if is_clf:
        return {
            "test_acc": accuracy_score(y_true, y_pred),
            "test_auc": roc_auc_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
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
