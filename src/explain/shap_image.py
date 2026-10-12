"""SHAP for the ultrasound CNN using GradientExplainer (expected gradients).

Produces pixel-level attribution maps showing which regions of the scan
(e.g. peripheral follicle 'string of pearls', stromal area) drove the
PCOS prediction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.data.ultrasound import denormalize
from src.models.image_model import ProbWrapper
from src.utils import ensure_dir


class ImageExplainer:
    def __init__(self, model: torch.nn.Module, background: torch.Tensor, device) -> None:
        self.device = device
        self.model = ProbWrapper(model).to(device).eval()
        self.explainer = shap.GradientExplainer(self.model, background.to(device))

    def explain(self, x: torch.Tensor, nsamples: int = 200) -> np.ndarray:
        """Returns SHAP values shaped (N, H, W, 3)."""
        sv = self.explainer.shap_values(x.to(self.device), nsamples=nsamples)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 5:          # (N,C,H,W,outputs) in newer shap versions
            sv = sv[..., 0]
        return np.transpose(sv, (0, 2, 3, 1))

    @torch.no_grad()
    def probs(self, x: torch.Tensor) -> np.ndarray:
        return self.model(x.to(self.device)).squeeze(1).cpu().numpy()

    def save_overlays(self, x: torch.Tensor, sv: np.ndarray, out_dir: str | Path,
                      names: list[str] | None = None) -> list[Path]:
        """Side-by-side: original | SHAP heatmap (red=PCOS, blue=normal) | overlay."""
        out = ensure_dir(out_dir)
        imgs = denormalize(x).permute(0, 2, 3, 1).cpu().numpy()
        probs = self.probs(x)
        paths = []
        for i in range(len(imgs)):
            heat = sv[i].sum(axis=-1)
            lim = np.percentile(np.abs(heat), 99.5) or 1e-8
            gray = imgs[i].mean(axis=-1)

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(gray, cmap="gray")
            ax[0].set_title(f"Ultrasound  |  P(PCOS)={probs[i]:.2f}")
            im = ax[1].imshow(heat, cmap="bwr", vmin=-lim, vmax=lim)
            ax[1].set_title("SHAP attribution")
            fig.colorbar(im, ax=ax[1], fraction=0.046)
            ax[2].imshow(gray, cmap="gray")
            ax[2].imshow(np.clip(heat, 0, None), cmap="Reds", alpha=0.5, vmin=0, vmax=lim)
            ax[2].set_title("Evidence for PCOS")
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            name = names[i] if names else f"sample_{i}"
            p = out / f"ultrasound_shap_{name}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        return paths
