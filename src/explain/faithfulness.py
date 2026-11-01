"""Faithfulness curves and SHAP stability checks for the clinical and image models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.explain.shap_clinical import ClinicalExplainer
from src.models.clinical_model import fit_with_smote

_ROTTERDAM = {
    "cycle_ri": "oligo_anovulation",
    "cycle_length_days": "oligo_anovulation",
    "hair_growth": "hyperandrogenism",
    "pimples": "hyperandrogenism",
    "hair_loss": "hyperandrogenism",
    "follicle_n_L": "polycystic_ovary_morphology",
    "follicle_n_R": "polycystic_ovary_morphology",
    "total_follicles": "polycystic_ovary_morphology",
    "mean_follicle_size": "polycystic_ovary_morphology",
    "follicle_size_l_mm": "polycystic_ovary_morphology",
    "follicle_size_r_mm": "polycystic_ovary_morphology",
}
_DROP_KS = (1, 3, 5, 10, 20)


def top_k_drop_curve(model, X_test, y_test, shap_values, ks=None, impute: str = "median") -> pd.DataFrame:
    """AUC after replacing the top-k |SHAP| features, with a random-k control."""
    frame = pd.DataFrame(X_test).reset_index(drop=True)
    values = _matrix(shap_values)
    labels = np.asarray(y_test).astype(int)
    if len(frame) != len(values):
        raise ValueError("SHAP rows must match X_test")
    fill = _fill_values(frame, impute)
    ranking = np.argsort(np.abs(values), axis=1)[:, ::-1]
    generator = np.random.default_rng(0)
    rows = []
    for k in _DROP_KS if ks is None else ks:
        width = min(int(k), frame.shape[1])
        chosen = ranking[:, :width]
        random_choice = np.vstack([generator.choice(frame.shape[1], size=width, replace=False) for _ in range(len(frame))])
        for strategy, columns in (("shap", chosen), ("random", random_choice)):
            masked = _replace(frame, columns, fill)
            rows.append({"k": int(k), "strategy": strategy, "auc": _auc(labels, _positive_proba(model, masked))})
    return pd.DataFrame(rows, columns=["k", "strategy", "auc"])


def stability_across_folds(model_builder, X, y, folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Spearman correlation of mean |SHAP| across stratified training folds."""
    frame = pd.DataFrame(X).reset_index(drop=True)
    labels = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    importances = []
    for train_idx, test_idx in splitter.split(frame, labels):
        fitted = fit_with_smote(model_builder(int(seed)), frame.iloc[train_idx], labels[train_idx])
        importances.append(_global_importance(fitted, frame.iloc[train_idx], frame.iloc[test_idx], seed))
    return _pairwise_spearman(importances)


def stability_across_seeds(model_builder, X, y, seeds=None) -> dict[str, float]:
    """Spearman correlation of mean |SHAP| across model seeds on the same rows."""
    frame = pd.DataFrame(X).reset_index(drop=True)
    labels = np.asarray(y).astype(int)
    importances = []
    for seed in (0, 1, 2, 3, 4) if seeds is None else seeds:
        fitted = fit_with_smote(model_builder(int(seed)), frame, labels)
        importances.append(_global_importance(fitted, frame, frame, int(seed)))
    return _pairwise_spearman(importances)


def rotterdam_alignment(global_importance_df: pd.DataFrame) -> float:
    """Fraction of the top-10 features that map to a Rotterdam criterion group."""
    frame = global_importance_df.copy()
    if "feature" not in frame.columns:
        frame = frame.rename_axis("feature").reset_index()
    importance = "importance" if "importance" in frame.columns else frame.select_dtypes("number").columns[0]
    ranked = frame.sort_values(importance, ascending=False).head(10)
    if ranked.empty:
        return 0.0
    return float(ranked["feature"].astype(str).isin(_ROTTERDAM).mean())


def patch_drop_curve(model, images: torch.Tensor, labels, shap_values, ks=None, patch: int = 8, device="cpu") -> pd.DataFrame:
    """AUC after masking the top-k 8x8 patches by mean |SHAP|, with a random-patch control."""
    batch = images.detach().cpu()
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"Expected SHAP image values (N, H, W, 3), received {values.shape}")
    height, width = batch.shape[-2:]
    if height % patch or width % patch:
        raise ValueError(f"Image size {(height, width)} is not divisible by patch {patch}")
    scores = _patch_scores(values, patch)
    ranking = np.argsort(scores, axis=1)[:, ::-1]
    generator = np.random.default_rng(0)
    targets = np.asarray(labels).astype(int)
    rows = []
    for k in _DROP_KS if ks is None else ks:
        width_k = min(int(k), scores.shape[1])
        chosen = ranking[:, :width_k]
        random_choice = np.vstack([generator.choice(scores.shape[1], size=width_k, replace=False) for _ in range(len(batch))])
        for strategy, patches in (("shap", chosen), ("random", random_choice)):
            masked = _mask_patches(batch, patches, patch)
            rows.append({"k": int(k), "strategy": strategy, "auc": _auc(targets, _image_probs(model, masked, device))})
    return pd.DataFrame(rows, columns=["k", "strategy", "auc"])


def drop_top_k(predict_fn, features: np.ndarray, shap_values: np.ndarray, k: int) -> float:
    """Mean probability drop after zeroing the k largest-|SHAP| features."""
    baseline = predict_fn(features)
    masked = features.copy()
    for row, values in enumerate(shap_values):
        ranking = np.argsort(np.abs(values))[::-1][:k]
        masked[row, ranking] = 0.0
    return float(np.mean(baseline - predict_fn(masked)))


def stability(runs: list[np.ndarray]) -> float:
    """Mean pairwise correlation of SHAP vectors across repeated runs."""
    if len(runs) < 2:
        return 1.0
    scores = []
    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1 :]:
            pair = np.vstack([left.ravel(), right.ravel()])
            if pair[0].std() == 0 or pair[1].std() == 0:
                scores.append(1.0 if np.allclose(pair[0], pair[1]) else 0.0)
            else:
                scores.append(float(np.corrcoef(pair)[0, 1]))
    return float(np.mean(scores))


def _global_importance(model, background: pd.DataFrame, features: pd.DataFrame, seed: int) -> np.ndarray:
    sample_n = min(100, len(background))
    sample = background.sample(n=sample_n, random_state=int(seed))
    explanation = ClinicalExplainer(model, sample).explain(features)
    return np.abs(_matrix(explanation.values)).mean(axis=0)


def _pairwise_spearman(importances: list[np.ndarray]) -> dict[str, float]:
    rhos = []
    for index, left in enumerate(importances):
        for right in importances[index + 1 :]:
            if np.std(left) == 0 or np.std(right) == 0:
                rhos.append(1.0 if np.allclose(left, right) else 0.0)
                continue
            result = spearmanr(left, right)
            rho = float(getattr(result, "statistic", result.correlation))
            if np.isfinite(rho):
                rhos.append(rho)
    if not rhos:
        return {"mean_rho": float("nan"), "std_rho": float("nan")}
    array = np.asarray(rhos, dtype=float)
    return {"mean_rho": float(array.mean()), "std_rho": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def _patch_scores(values: np.ndarray, patch: int) -> np.ndarray:
    count, height, width, _ = values.shape
    grid_h, grid_w = height // patch, width // patch
    blocks = values[:, : grid_h * patch, : grid_w * patch, :].reshape(count, grid_h, patch, grid_w, patch, 3)
    return np.abs(blocks).mean(axis=(2, 4, 5)).reshape(count, grid_h * grid_w)


def _mask_patches(images: torch.Tensor, chosen: np.ndarray, patch: int) -> torch.Tensor:
    masked = images.clone()
    grid_w = masked.shape[-1] // patch
    for row_index in range(masked.shape[0]):
        for patch_id in chosen[row_index]:
            top, left = divmod(int(patch_id), grid_w)
            masked[row_index, :, top * patch : (top + 1) * patch, left * patch : (left + 1) * patch] = 0
    return masked


def _image_probs(model, images: torch.Tensor, device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            logits = model(images[start : start + 16].to(device)).reshape(-1)
            outputs.append(torch.sigmoid(logits).cpu())
    return torch.cat(outputs).numpy() if outputs else np.empty(0, dtype=float)


def _fill_values(frame: pd.DataFrame, impute: str) -> np.ndarray:
    if impute == "zero":
        return np.zeros(frame.shape[1], dtype=float)
    if impute == "median":
        return frame.median(numeric_only=True).to_numpy(dtype=float)
    raise ValueError("impute must be 'median' or 'zero'")


def _replace(frame: pd.DataFrame, columns: np.ndarray, fill: np.ndarray) -> np.ndarray:
    masked = frame.to_numpy(dtype=float).copy()
    rows = np.arange(len(masked))[:, None]
    masked[rows, columns] = fill[columns]
    return masked


def _positive_proba(model, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    index = list(map(int, classes)).index(1) if 1 in set(map(int, classes)) else -1
    return probabilities[:, index]


def _matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))
