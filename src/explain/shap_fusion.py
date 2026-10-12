"""Modality-level SHAP: how much did clinical vs ultrasound evidence
contribute to each fused diagnosis?"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.models.fusion import MODALITIES, LateFusion
from src.utils import ensure_dir


class FusionExplainer:
    def __init__(self, fusion: LateFusion, background: np.ndarray) -> None:
        bg = shap.kmeans(background, 10) if len(background) > 50 else background
        self.explainer = shap.KernelExplainer(fusion.predict_proba, bg)

    def explain(self, P: np.ndarray) -> np.ndarray:
        """P: (N,2) unimodal probs -> (N,2) SHAP contributions."""
        return np.asarray(self.explainer.shap_values(P, silent=True))

    def save_contribution_plot(self, sv: np.ndarray, out_path: str | Path) -> None:
        ensure_dir(Path(out_path).parent)
        share = np.abs(sv) / (np.abs(sv).sum(axis=1, keepdims=True) + 1e-12)
        mean_share = share.mean(axis=0)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(MODALITIES, mean_share, color=["#4C72B0", "#DD8452"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean share of |SHAP| in fused prediction")
        ax.set_title("Modality contribution")
        for i, v in enumerate(mean_share):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    @property
    def base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])
