"""Gradient SHAP overlays for ultrasound images."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.data.ultrasound_loader import denormalize
from src.models.image_model import ProbWrapper


class ImageExplainer:
    """Attribute the sigmoid probability of a logit model with GradientExplainer."""

    def __init__(self, model, background_batch: torch.Tensor, device) -> None:
        self.device = device
        self.model = ProbWrapper(model).to(device)
        self.model.eval()
        background = background_batch.detach().to(device)
        self.explainer = shap.GradientExplainer(self.model, background)

    def explain(self, images: torch.Tensor, nsamples: int = 200) -> np.ndarray:
        """Return SHAP values as (N, H, W, 3)."""
        self.model.eval()
        values = self.explainer.shap_values(images.detach().to(self.device), nsamples=int(nsamples))
        return _as_nhwc(values)

    def probs(self, images: torch.Tensor) -> np.ndarray:
        """Sigmoid probabilities for a batch, one float per image."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(images.detach().to(self.device))
        return output.detach().cpu().numpy().reshape(-1)

    def save_overlays(self, images: torch.Tensor, shap_values: np.ndarray, out_dir, names: list[str]) -> list[Path]:
        """Save original, diverging heatmap, and red positive-attribution panels."""
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        batch = np.asarray(shap_values)
        if len(names) != len(batch):
            raise ValueError(f"Expected {len(batch)} names, received {len(names)}")
        written = []
        for index, name in enumerate(names):
            original = _to_hwc(images[index])
            heat = batch[index].mean(axis=-1)
            figure, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(original)
            axes[0].set_title("Original")
            limit = float(np.percentile(np.abs(heat), 99))
            limit = limit if limit > 0 else 1.0
            axes[1].imshow(heat, cmap="bwr", vmin=-limit, vmax=limit)
            axes[1].set_title("SHAP")
            axes[2].imshow(_positive_overlay(original, heat))
            axes[2].set_title("Positive")
            for axis in axes:
                axis.axis("off")
            figure.tight_layout()
            path = destination / f"{name}.png"
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            written.append(path)
        return written


def _as_nhwc(values) -> np.ndarray:
    array = values[0] if isinstance(values, list) else values
    array = np.asarray(array)
    if array.ndim == 5:
        array = array[0]
    if array.ndim == 4 and array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected SHAP values with shape (N, H, W, 3), received {array.shape}")
    return array


def _to_hwc(image: torch.Tensor) -> np.ndarray:
    restored = denormalize(image.detach().cpu()).numpy()
    return np.transpose(restored, (1, 2, 0))


def _positive_overlay(original: np.ndarray, heat: np.ndarray) -> np.ndarray:
    positive = np.clip(heat, 0, None)
    peak = float(positive.max())
    alpha = positive / peak if peak > 0 else positive
    red = np.zeros_like(original)
    red[..., 0] = 1.0
    blended = original * (1.0 - alpha[..., None]) + red * alpha[..., None]
    return np.clip(blended, 0.0, 1.0)
