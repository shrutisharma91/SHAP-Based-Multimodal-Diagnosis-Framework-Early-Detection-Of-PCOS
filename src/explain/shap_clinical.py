"""SHAP explanations for a fitted clinical model, including modality summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

_MODALITIES = ("symptoms", "hormones", "ultrasound", "vitals", "demographics")


class ClinicalExplainer:
    """Explain a fitted clinical classifier. Background rows should come from training data."""

    def __init__(self, model, background) -> None:
        frame = background if isinstance(background, pd.DataFrame) else pd.DataFrame(background)
        self.feature_names = [str(column) for column in frame.columns]
        self.explainer = shap.Explainer(_final_estimator(model), frame.to_numpy(dtype=float))

    def explain(self, features) -> shap.Explanation:
        frame = _aligned_frame(features, self.feature_names)
        explanation = self.explainer(frame.to_numpy(dtype=float))
        explanation.feature_names = self.feature_names
        explanation.data = frame.to_numpy(dtype=float)
        return _positive_explanation(explanation)

    def save_global_plots(self, explanation, out_dir, top_k: int = 15) -> list[Path]:
        """Write beeswarm, bar, and dependence plots for the three strongest features."""
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        values, data, names = _arrays(explanation)
        written = [
            _summary(values, data, names, destination / "beeswarm.png", top_k, "dot"),
            _summary(values, data, names, destination / "bar.png", top_k, "bar"),
        ]
        order = np.argsort(np.abs(values).mean(axis=0))[::-1][:3]
        for feature_index in order:
            feature_name = names[int(feature_index)]
            shap.dependence_plot(feature_name, values, data, feature_names=names, show=False)
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in feature_name)
            written.append(_save_current(destination / f"dependence_{safe_name}.png"))
        return written

    def save_patient_waterfall(self, explanation, idx: int, path) -> Path:
        shap.plots.waterfall(explanation[int(idx)], show=False)
        return _save_current(path)

    def patient_reasons(self, explanation, idx: int, top_k: int = 5) -> dict[str, list[dict]]:
        """Up to top_k features pushing risk up, and up to top_k pushing it down."""
        values, _, names = _arrays(explanation)
        row = values[int(idx)]
        order = np.argsort(row)
        decreases = [
            {"feature": names[int(feature)], "shap": float(row[int(feature)])}
            for feature in order
            if row[int(feature)] < 0
        ][:top_k]
        increases = [
            {"feature": names[int(feature)], "shap": float(row[int(feature)])}
            for feature in order[::-1]
            if row[int(feature)] > 0
        ][:top_k]
        return {"increases_risk": increases, "decreases_risk": decreases}

    def modality_breakdown(self, explanation, feature_to_modality_map: dict[str, str]) -> dict[str, float]:
        """Mean absolute SHAP value of the features assigned to each modality."""
        values, _, names = _arrays(explanation)
        breakdown = {}
        for modality in _MODALITIES:
            columns = [index for index, name in enumerate(names) if feature_to_modality_map.get(name) == modality]
            breakdown[modality] = float(np.abs(values[:, columns]).mean()) if columns else 0.0
        return breakdown


def _final_estimator(model):
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps"):
        return model.named_steps["clf"]
    return model


def _aligned_frame(features, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.loc[:, feature_names]
    return pd.DataFrame(np.asarray(features, dtype=float), columns=feature_names)


def _positive_explanation(explanation: shap.Explanation) -> shap.Explanation:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim < 3:
        return explanation
    class_index = values.shape[-1] - 1
    base = np.asarray(explanation.base_values, dtype=float)
    if base.ndim >= 2:
        base = base[..., class_index]
    data = None if explanation.data is None else np.asarray(explanation.data)
    return shap.Explanation(
        values=values[:, :, class_index],
        base_values=base,
        data=data,
        feature_names=explanation.feature_names,
    )


def _arrays(explanation) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    data = np.asarray(explanation.data, dtype=float)
    names = [str(name) for name in explanation.feature_names]
    return values, data, names


def _summary(values, data, names, path: Path, top_k: int, plot_type: str) -> Path:
    shap.summary_plot(values, data, feature_names=names, show=False, max_display=top_k, plot_type=plot_type)
    return _save_current(path)


def _save_current(path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight")
    plt.close()
    return destination
