"""Evaluation metrics and plots — recall-focused, since a missed PCOS case
is the costlier error in an early-detection screener."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_dir


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "threshold": threshold,
    }


def best_threshold(y_true, y_prob, min_recall: float = 0.90) -> float:
    """Highest threshold that still achieves the target sensitivity."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= min_recall)[0]
    return float(np.clip(thr[ok[0]], 0, 1)) if len(ok) else 0.5


def plot_roc(curves: dict[str, tuple], out_path: str | Path) -> None:
    """curves: {name: (y_true, y_prob)} — overlay unimodal vs fused ROC."""
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(y_true, y_prob, out_path: str | Path, threshold: float = 0.5) -> None:
    ensure_dir(Path(out_path).parent)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Normal", "PCOS"], cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
