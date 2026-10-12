"""XGBoost model for clinical / biochemical features."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.utils import ensure_dir, get_logger

log = get_logger(__name__)


def build_clinical_model(params: dict, y_train: pd.Series, seed: int) -> XGBClassifier:
    # Kaggle PCOS set is ~33% positive -> rebalance
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    return XGBClassifier(
        **params,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def cross_validate(model: XGBClassifier, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {
        m: cross_val_score(model, X, y, cv=cv, scoring=m)
        for m in ("roc_auc", "f1", "recall")
    }
    summary = {m: {"mean": float(np.mean(s)), "std": float(np.std(s))} for m, s in scores.items()}
    for m, s in summary.items():
        log.info("CV %-8s %.4f ± %.4f", m, s["mean"], s["std"])
    return summary


def save_model(model: XGBClassifier, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    joblib.dump(model, path)


def load_model(path: str | Path) -> XGBClassifier:
    return joblib.load(path)
