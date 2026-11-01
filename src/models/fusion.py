"""Late-fusion logistic regression that stays valid when a modality is missing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LateFusion:
    """Combine clinical and image probabilities. Either score may be absent."""

    def __init__(self, C: float = 1.0) -> None:
        self.model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, clinical_proba: np.ndarray, image_proba: np.ndarray, labels: np.ndarray) -> "LateFusion":
        self.model.fit(self._stack(clinical_proba, image_proba), labels)
        return self

    def predict_proba(self, clinical_proba: np.ndarray | None, image_proba: np.ndarray | None) -> np.ndarray:
        clinical, image = self._align(clinical_proba, image_proba)
        return self.model.predict_proba(self._stack(clinical, image))[:, 1]

    @staticmethod
    def _align(clinical_proba, image_proba):
        if clinical_proba is None and image_proba is None:
            raise ValueError("At least one modality probability is required")
        if clinical_proba is None:
            clinical_proba = np.full_like(image_proba, np.nan, dtype=float)
        if image_proba is None:
            image_proba = np.full_like(clinical_proba, np.nan, dtype=float)
        return np.asarray(clinical_proba, dtype=float), np.asarray(image_proba, dtype=float)

    @staticmethod
    def _stack(clinical_proba: np.ndarray, image_proba: np.ndarray) -> np.ndarray:
        clinical = np.nan_to_num(np.asarray(clinical_proba, dtype=float), nan=0.5)
        image = np.nan_to_num(np.asarray(image_proba, dtype=float), nan=0.5)
        missing_clinical = np.isnan(np.asarray(clinical_proba, dtype=float)).astype(float)
        missing_image = np.isnan(np.asarray(image_proba, dtype=float)).astype(float)
        return np.column_stack([clinical, image, missing_clinical, missing_image])
