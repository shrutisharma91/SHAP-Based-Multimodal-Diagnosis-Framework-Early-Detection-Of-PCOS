"""Build clinical classifiers, cross-validate them, and fit SMOTE inside each training fold."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from xgboost import XGBClassifier

_XGB_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
    "gamma",
)
_RF_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features")
_METRICS = ("roc_auc", "f1", "recall", "precision")


def build_xgb(params: dict, y_train, seed: int) -> XGBClassifier:
    """XGBoost with scale_pos_weight equal to the negative/positive training ratio."""
    labels = np.asarray(y_train).astype(int)
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("build_xgb needs at least one positive training row")
    negatives = int((labels == 0).sum())
    chosen = {key: params[key] for key in _XGB_KEYS if key in params and params[key] is not None}
    return XGBClassifier(
        **chosen,
        scale_pos_weight=negatives / positives,
        random_state=int(seed),
        eval_metric="logloss",
        n_jobs=1,
    )


def build_rf(params: dict, seed: int) -> RandomForestClassifier:
    """Random forest using the supplied hyperparameters."""
    chosen = {key: params[key] for key in _RF_KEYS if key in params and params[key] is not None}
    return RandomForestClassifier(**chosen, random_state=int(seed), n_jobs=1)


def build_logreg(seed: int, class_weight: str | None = "balanced") -> LogisticRegression:
    """Logistic regression with balanced class weights."""
    return LogisticRegression(class_weight=class_weight, random_state=int(seed), max_iter=1000, solver="lbfgs")


def cross_validate(model, X, y, folds: int, seed: int) -> dict[str, dict[str, float]]:
    """Stratified CV. SMOTE is fit only on each training fold."""
    pipeline = _make_pipeline(clone(model), y, seed)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    scores = sklearn_cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring={metric: metric for metric in _METRICS},
        n_jobs=1,
    )
    summary: dict[str, dict[str, float]] = {}
    for metric in _METRICS:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[metric] = {"mean": float(values.mean()), "std": deviation}
    return summary


def fit_with_smote(model, X_tr, y_tr):
    """Fit a clone inside an imblearn Pipeline so SMOTE never sees held-out rows."""
    seed = int(getattr(model, "random_state", 0) or 0)
    pipeline = _make_pipeline(clone(model), y_tr, seed)
    pipeline.fit(X_tr, y_tr)
    return pipeline


def save_model(model, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path):
    return joblib.load(path)


def _make_pipeline(model, y, seed: int) -> Pipeline:
    labels = np.asarray(y).astype(int)
    counts = np.bincount(labels)
    present = counts[counts > 0]
    minority = int(present.min()) if present.size else 0
    if minority < 2:
        raise ValueError("SMOTE needs at least two rows in each class")
    sampler = SMOTE(random_state=int(seed), k_neighbors=min(5, minority - 1))
    return Pipeline([("smote", sampler), ("clf", model)])
