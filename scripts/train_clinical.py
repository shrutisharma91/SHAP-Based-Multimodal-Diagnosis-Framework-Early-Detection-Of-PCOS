"""Train clinical models for feature sets A-D. Fitting runs only from main()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from src.data.clinical_preprocess import feature_sets
from src.evaluation.metrics import best_threshold, compute_metrics, plot_calibration, plot_confusion, plot_roc
from src.explain.shap_clinical import ClinicalExplainer
from src.models.clinical_model import (
    build_logreg,
    build_rf,
    build_xgb,
    cross_validate,
    fit_with_smote,
    save_model,
)
from src.utils import ensure_dir, load_config, save_json, set_seed

_SETS = (
    ("A", "A_full"),
    ("B", "B_no_ultrasound"),
    ("C", "C_no_us_no_hcg"),
    ("D", "D_no_amh"),
)
_MODALITY = {
    "symptoms": (
        "cycle_ri",
        "cycle_length_days",
        "pregnant",
        "n_abortions",
        "weight_gain",
        "hair_growth",
        "skin_darkening",
        "hair_loss",
        "pimples",
        "fast_food",
        "reg_exercise",
    ),
    "hormones": ("beta_hcg_1", "beta_hcg_2", "fsh", "lh", "lh_fsh_ratio", "tsh", "amh", "prl", "vit_d3", "prg"),
    "ultrasound": (
        "total_follicles",
        "mean_follicle_size",
        "endometrium_mm",
        "follicle_n_L",
        "follicle_n_R",
        "follicle_size_l_mm",
        "follicle_size_r_mm",
    ),
    "vitals": ("pulse_bpm", "rr_breaths_min", "hb_g_dl", "bp_systolic", "bp_diastolic", "map_bp", "rbs"),
    "demographics": (
        "age_yrs",
        "weight_kg",
        "height_cm",
        "bmi",
        "marriage_yrs",
        "blood_group",
        "waist_inch",
        "hip_inch",
        "waist_hip_ratio",
    ),
}


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    set_seed(seed)
    folds = int(config["clinical"]["cv_folds"])
    target_name = config["clinical"]["target_col"]
    processed = _located(config["paths"]["processed_dir"])
    artifacts = _located(config["paths"]["artifacts_dir"])
    train_frame = pd.read_parquet(processed / "clinical_train.parquet")
    test_frame = pd.read_parquet(processed / "clinical_test.parquet")
    train_x, train_y = _split_frame(train_frame, target_name)
    test_x, test_y = _split_frame(test_frame, target_name)
    primary = _primary_mask(test_frame)
    modality_map = {feature: modality for modality, features in _MODALITY.items() for feature in features}
    columns = feature_sets(train_x)
    figures = ensure_dir(artifacts / "figures" / "clinical")
    models = ensure_dir(artifacts / "models")
    results: dict = {
        "note": "Set A keeps follicle features used in the Rotterdam criteria; set B drops them.",
        "sets": {},
    }
    comparison = []
    for set_id, set_key in _SETS:
        set_columns = columns[set_key]
        x_train = train_x.loc[:, set_columns]
        x_test = test_x.loc[:, set_columns]
        print(f"\n### {set_id} ({set_key}, {len(set_columns)} features)")
        print("| algo | roc_auc | f1 | recall | precision |")
        print("|---|---|---|---|---|")
        fitted = {}
        curves = {}
        model_reports = {}
        for algo in ("xgb", "rf", "logreg"):
            model = _build(algo, config, train_y, seed)
            cv_scores = cross_validate(model, x_train, train_y, folds, seed)
            _print_cv_row(algo, cv_scores)
            # Threshold uses training out-of-fold scores only, then is applied to test.
            oof = _oof_proba(model, x_train, train_y, folds, seed)
            threshold = best_threshold(train_y, oof, min_recall=0.9)
            final = fit_with_smote(clone(model), x_train, train_y)
            probabilities = _positive_proba(final, x_test)
            test_metrics = compute_metrics(test_y, probabilities, threshold)
            primary_metrics = compute_metrics(test_y[primary], probabilities[primary], threshold)
            save_model(final, models / f"clinical_{set_id}_{algo}.joblib")
            fitted[algo] = (final, probabilities, threshold, cv_scores, test_metrics, primary_metrics)
            curves[algo] = roc_curve(test_y, probabilities)[:2]
            model_reports[algo] = _report(cv_scores, threshold, test_metrics, primary_metrics)
        best_algo = max(fitted, key=lambda name: fitted[name][3]["roc_auc"]["mean"])
        best = fitted[best_algo]
        plot_roc(curves, figures / f"roc_{set_id}.png")
        plot_confusion(test_y, best[1], figures / f"confusion_{set_id}.png", best[2])
        plot_calibration(test_y, best[1], figures / f"calibration_{set_id}.png")
        background = x_train.sample(n=min(100, len(x_train)), random_state=seed)
        explainer = ClinicalExplainer(best[0], background)
        explanation = explainer.explain(x_test)
        shap_dir = ensure_dir(figures / set_id)
        explainer.save_global_plots(explanation, shap_dir, top_k=int(config["shap"]["clinical_top_k"]))
        explainer.save_patient_waterfall(explanation, 0, shap_dir / "waterfall.png")
        results["sets"][set_id] = {
            "feature_set": set_key,
            "features": set_columns,
            "best_algo": best_algo,
            "models": model_reports,
            "patient_reasons": explainer.patient_reasons(explanation, 0, top_k=5),
            "modality_breakdown": explainer.modality_breakdown(explanation, modality_map),
        }
        comparison.append((set_id, best_algo, best[3], best[4], best[5]))
    save_json(_clean(results), ensure_dir(artifacts / "reports") / "clinical_results.json")
    _print_comparison(comparison)
    print("Set A keeps follicle features used in the Rotterdam criteria; set B drops them.")


def _build(algo: str, config: dict, y_train, seed: int):
    if algo == "xgb":
        return build_xgb(config["clinical"]["xgb"], y_train, seed)
    if algo == "rf":
        return build_rf(config["clinical"]["rf"], seed)
    return build_logreg(seed)


def _oof_proba(model, features: pd.DataFrame, target, folds: int, seed: int) -> np.ndarray:
    labels = np.asarray(target).astype(int)
    matrix = features.to_numpy(dtype=float)
    scores = np.zeros(len(features), dtype=float)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    for train_idx, valid_idx in splitter.split(matrix, labels):
        fold_model = fit_with_smote(clone(model), matrix[train_idx], labels[train_idx])
        scores[valid_idx] = _positive_proba(fold_model, matrix[valid_idx])
    return scores


def _positive_proba(model, features) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    index = list(classes).index(1) if 1 in set(classes) else probabilities.shape[1] - 1
    return probabilities[:, index]


def _split_frame(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    drop = [column for column in (target, "is_synthetic_inferred", "patient_file_no", "sl_no") if column in frame.columns]
    features = frame.drop(columns=drop)
    return features, frame[target].to_numpy(dtype=int)


def _primary_mask(frame: pd.DataFrame) -> np.ndarray:
    if "is_synthetic_inferred" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return frame["is_synthetic_inferred"].to_numpy() == 0


def _report(cv_scores, threshold, test_metrics, primary_metrics) -> dict:
    return {
        "cv": cv_scores,
        "threshold": threshold,
        "test": test_metrics,
        "primary_test": primary_metrics,
    }


def _print_cv_row(algo: str, scores: dict) -> None:
    cells = [f"{scores[metric]['mean']:.3f} ± {scores[metric]['std']:.3f}" for metric in ("roc_auc", "f1", "recall", "precision")]
    print(f"| {algo} | " + " | ".join(cells) + " |")


def _print_comparison(rows: list[tuple]) -> None:
    print("\n| set | best | cv_auc | test_auc | test_recall | primary_auc | primary_recall |")
    print("|---|---|---|---|---|---|---|")
    for set_id, algo, cv_scores, test_metrics, primary_metrics in rows:
        print(
            f"| {set_id} | {algo} | {cv_scores['roc_auc']['mean']:.3f} | "
            f"{test_metrics['roc_auc']:.3f} | {test_metrics['recall']:.3f} | "
            f"{primary_metrics['roc_auc']:.3f} | {primary_metrics['recall']:.3f} |"
        )


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    return value


def _located(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


if __name__ == "__main__":
    main()
