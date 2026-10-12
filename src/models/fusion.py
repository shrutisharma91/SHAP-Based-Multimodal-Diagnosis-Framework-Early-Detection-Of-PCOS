"""Late fusion of clinical and ultrasound modalities.

Each unimodal model emits P(PCOS). Fusion combines the two probabilities:
  - "weighted": fixed convex combination
  - "stacking": logistic-regression meta-learner on logit-transformed probs
                (learned weights are themselves interpretable)

Missing modality handling: if one probability is None, the other is returned,
so the framework still works when a patient has no scan (or no lab panel).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils import ensure_dir

MODALITIES = ["clinical", "ultrasound"]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class LateFusion:
    def __init__(self, strategy: str = "stacking", weights=(0.6, 0.4)) -> None:
        if strategy not in {"stacking", "weighted"}:
            raise ValueError(strategy)
        self.strategy = strategy
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)
        self.meta: LogisticRegression | None = None

    def fit(self, p_clin: np.ndarray, p_img: np.ndarray, y: np.ndarray) -> "LateFusion":
        if self.strategy == "stacking":
            Z = np.column_stack([_logit(p_clin), _logit(p_img)])
            self.meta = LogisticRegression(class_weight="balanced").fit(Z, y)
        return self

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        """P: (N,2) array of [p_clinical, p_ultrasound] -> (N,) fused prob."""
        P = np.atleast_2d(P)
        if self.strategy == "weighted":
            return P @ self.weights
        return self.meta.predict_proba(_logit(P))[:, 1]

    def predict_single(self, p_clin: float | None, p_img: float | None) -> float:
        if p_clin is None and p_img is None:
            raise ValueError("At least one modality is required")
        if p_img is None:
            return float(p_clin)
        if p_clin is None:
            return float(p_img)
        return float(self.predict_proba(np.array([[p_clin, p_img]]))[0])

    def modality_weights(self) -> dict[str, float]:
        if self.strategy == "weighted":
            return dict(zip(MODALITIES, self.weights.tolist()))
        return dict(zip(MODALITIES, self.meta.coef_[0].tolist()))

    def save(self, path: str | Path) -> None:
        ensure_dir(Path(path).parent)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "LateFusion":
        return joblib.load(path)
