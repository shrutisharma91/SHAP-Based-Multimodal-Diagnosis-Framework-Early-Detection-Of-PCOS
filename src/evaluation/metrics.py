"""Thresholded classification metrics and ROC, confusion, and calibration plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y, p, threshold: float) -> dict[str, float]:
    """Score probabilities at one threshold. Ranking metrics ignore the threshold."""
    labels = np.asarray(y).astype(int)
    probabilities = np.asarray(p, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    negative_total = true_negative + false_positive
    specificity = true_negative / negative_total if negative_total else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": _ranking_score(roc_auc_score, labels, probabilities),
        "pr_auc": _ranking_score(average_precision_score, labels, probabilities),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def best_threshold(y, p, min_recall: float = 0.9) -> float:
    """Highest threshold whose recall is still at least min_recall."""
    labels = np.asarray(y).astype(int)
    probabilities = np.asarray(p, dtype=float)
    positives = labels == 1
    if not positives.any():
        return 0.5
    chosen = None
    for threshold in np.sort(np.unique(probabilities))[::-1]:
        recall = float((probabilities[positives] >= threshold).mean())
        if recall >= min_recall:
            chosen = float(threshold)
            break
    if chosen is not None:
        return chosen
    fallback = max(np.unique(probabilities), key=lambda threshold: (probabilities[positives] >= threshold).mean())
    return float(fallback)


def plot_roc(curves_dict: dict, path: str | Path) -> Path:
    """Plot ROC curves. curves_dict maps a name to an (fpr, tpr) pair."""
    integrate = getattr(np, "trapezoid", np.trapz)
    plt.figure()
    for name, curve in curves_dict.items():
        fpr, tpr = np.asarray(curve[0], dtype=float), np.asarray(curve[1], dtype=float)
        plt.plot(fpr, tpr, label=f"{name} (AUC={integrate(tpr, fpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    return _save(path)


def plot_confusion(y, p, path: str | Path, threshold: float) -> Path:
    """Save a 2x2 confusion matrix at the given threshold."""
    labels = np.asarray(y).astype(int)
    predictions = (np.asarray(p, dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots()
    axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(int(value)), ha="center", va="center")
    axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    axis.set_title(f"Confusion matrix at {threshold:.3f}")
    fig.tight_layout()
    return _save(path)


def plot_calibration(y, p, path: str | Path) -> Path:
    """Save a reliability curve against the diagonal."""
    labels = np.asarray(y).astype(int)
    probabilities = np.asarray(p, dtype=float)
    bins = min(8, max(2, int(np.unique(probabilities).size)))
    fraction_positive, mean_predicted = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    plt.figure()
    plt.plot(mean_predicted, fraction_positive, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="upper left")
    return _save(path)


def _ranking_score(metric, labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(metric(labels, probabilities))


def _save(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight")
    plt.close()
    return destination
