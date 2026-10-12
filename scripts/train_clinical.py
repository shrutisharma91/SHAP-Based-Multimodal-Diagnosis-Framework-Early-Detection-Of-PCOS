"""Train + evaluate + explain the clinical model.

    python -m scripts.train_clinical --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.clinical import ClinicalPreprocessor, load_clinical, split_clinical
from src.evaluation.metrics import best_threshold, compute_metrics, plot_confusion, plot_roc
from src.explain.shap_clinical import ClinicalExplainer
from src.models.clinical_model import build_clinical_model, cross_validate, save_model
from src.utils import get_logger, load_config, save_json, set_seed

log = get_logger("train_clinical")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    c, art = cfg["clinical"], Path(cfg["paths"]["artifacts"])

    X, y, ids = load_clinical(cfg)
    X_tr, X_te, y_tr, y_te, _, ids_te = split_clinical(X, y, ids, c["test_size"], cfg["seed"])

    prep = ClinicalPreprocessor()
    X_tr = prep.fit_transform(X_tr)
    X_te = prep.transform(X_te)

    model = build_clinical_model(c["xgb"], y_tr, cfg["seed"])
    cv = cross_validate(model, X_tr, y_tr, c["cv_folds"], cfg["seed"])
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_te, p_te, min_recall=0.90)
    metrics = {"cv": cv, "test@0.5": compute_metrics(y_te, p_te), f"test@{thr:.3f}": compute_metrics(y_te, p_te, thr)}
    log.info("Test AUC %.4f | recall %.3f", metrics["test@0.5"]["roc_auc"], metrics["test@0.5"]["recall_sensitivity"])

    save_model(model, art / "models/clinical_xgb.joblib")
    prep.save(art / "models/clinical_preprocessor.joblib")
    save_json(metrics, art / "reports/clinical_metrics.json")
    plot_roc({"Clinical (XGBoost)": (y_te, p_te)}, art / "figures/clinical_roc.png")
    plot_confusion(y_te, p_te, art / "figures/clinical_confusion.png")

    # ---- SHAP ----
    explainer = ClinicalExplainer(model, background=X_tr.sample(min(100, len(X_tr)), random_state=cfg["seed"]))
    sv = explainer.explain(X_te)
    ClinicalExplainer.save_global_plots(sv, art / "figures/shap", cfg["shap"]["top_k"])
    ClinicalExplainer.global_importance(sv, cfg["shap"]["top_k"]).to_csv(
        art / "reports/clinical_feature_importance.csv", index=False)

    # local explanations for the 3 most confident PCOS predictions
    for i in p_te.argsort()[::-1][:3]:
        pid = ids_te.iloc[i]
        ClinicalExplainer.save_patient_waterfall(sv, int(i), art / f"figures/shap/patient_{pid}_waterfall.png")
        save_json({"patient_id": pid, "p_pcos": float(p_te[i]), **ClinicalExplainer.patient_reasons(sv, int(i))},
                  art / f"reports/patient_{pid}_clinical_explanation.json")
    log.info("Done. Artifacts in %s", art)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
