"""Fit the late-fusion layer on patients that have BOTH modalities.

Requires data/processed/paired_index.csv with columns:
    patient_id,image_path,label
where patient_id matches the clinical id column.

    python -m scripts.train_fusion --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.clinical import ClinicalPreprocessor, load_clinical
from src.data.ultrasound import load_image_tensor
from src.evaluation.metrics import compute_metrics, plot_roc
from src.explain.shap_fusion import FusionExplainer
from src.models.clinical_model import load_model
from src.models.fusion import LateFusion
from src.models.image_model import load_checkpoint
from src.utils import get_device, get_logger, load_config, save_json, set_seed

log = get_logger("train_fusion")


@torch.no_grad()
def image_probs(model, paths: list[str], img_size: int, device, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths), bs):
        x = torch.cat([load_image_tensor(p, img_size) for p in paths[i:i + bs]]).to(device)
        out.append(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    art, device, fc = Path(cfg["paths"]["artifacts"]), get_device(), cfg["fusion"]

    paired = pd.read_csv(cfg["paths"]["paired_csv"], dtype={"patient_id": str})
    X, _, ids = load_clinical(cfg)
    X.index = ids
    paired = paired[paired.patient_id.isin(X.index)].reset_index(drop=True)
    log.info("Paired patients: %d", len(paired))

    clin = load_model(art / "models/clinical_xgb.joblib")
    prep = ClinicalPreprocessor.load(art / "models/clinical_preprocessor.joblib")
    cnn = load_checkpoint(art / "models/ultrasound_cnn.pt", device)

    p_clin = clin.predict_proba(prep.transform(X.loc[paired.patient_id]))[:, 1]
    p_img = image_probs(cnn, paired.image_path.tolist(), cfg["image"]["img_size"], device)
    P, y = np.column_stack([p_clin, p_img]), paired.label.to_numpy()

    P_tr, P_te, y_tr, y_te = train_test_split(P, y, test_size=fc["test_size"], stratify=y, random_state=cfg["seed"])
    fusion = LateFusion(fc["strategy"], fc["weights"]).fit(P_tr[:, 0], P_tr[:, 1], y_tr)
    p_fused = fusion.predict_proba(P_te)

    report = {
        "clinical_only": compute_metrics(y_te, P_te[:, 0]),
        "ultrasound_only": compute_metrics(y_te, P_te[:, 1]),
        "fused": compute_metrics(y_te, p_fused),
        "modality_weights": fusion.modality_weights(),
    }
    for k in ("clinical_only", "ultrasound_only", "fused"):
        log.info("%-16s AUC %.4f  recall %.3f", k, report[k]["roc_auc"], report[k]["recall_sensitivity"])

    fusion.save(art / "models/fusion.joblib")
    save_json(report, art / "reports/fusion_metrics.json")
    plot_roc({"Clinical": (y_te, P_te[:, 0]), "Ultrasound": (y_te, P_te[:, 1]), "Fused": (y_te, p_fused)},
             art / "figures/fusion_roc_comparison.png")

    fx = FusionExplainer(fusion, P_tr)
    fx.save_contribution_plot(fx.explain(P_te), art / "figures/shap/fusion_modality_contribution.png")
    log.info("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
