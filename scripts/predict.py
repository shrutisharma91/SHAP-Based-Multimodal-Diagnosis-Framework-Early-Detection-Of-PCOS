"""Single-patient inference with a full explanation report.

    python -m scripts.predict --clinical sample_patient.json --image scan.jpg
    python -m scripts.predict --clinical sample_patient.json          # clinical only
    python -m scripts.predict --image scan.jpg                        # ultrasound only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.clinical import ClinicalPreprocessor, _engineer
from src.data.ultrasound import load_image_tensor
from src.explain.shap_clinical import ClinicalExplainer
from src.explain.shap_image import ImageExplainer
from src.models.clinical_model import load_model
from src.models.fusion import LateFusion
from src.models.image_model import load_checkpoint
from src.utils import ensure_dir, get_device, load_config, load_json, save_json

RISK_BANDS = [(0.3, "Low"), (0.7, "Moderate"), (1.01, "High")]


def risk_band(p: float) -> str:
    return next(label for cut, label in RISK_BANDS if p < cut)


def main(args) -> None:
    cfg = load_config(args.config)
    art, device = Path(cfg["paths"]["artifacts"]), get_device()
    out = ensure_dir(args.out)
    report: dict = {}
    p_clin = p_img = None

    if args.clinical:
        model = load_model(art / "models/clinical_xgb.joblib")
        prep = ClinicalPreprocessor.load(art / "models/clinical_preprocessor.joblib")
        row = _engineer(pd.DataFrame([load_json(args.clinical)]).apply(pd.to_numeric, errors="coerce"))
        X = prep.transform(row)
        p_clin = float(model.predict_proba(X)[0, 1])
        sv = ClinicalExplainer(model).explain(X)
        ClinicalExplainer.save_patient_waterfall(sv, 0, out / "clinical_waterfall.png")
        report["clinical"] = {"p_pcos": p_clin, **ClinicalExplainer.patient_reasons(sv, 0)}

    if args.image:
        cnn = load_checkpoint(art / "models/ultrasound_cnn.pt", device)
        x = load_image_tensor(args.image, cfg["image"]["img_size"])
        # Background: blurred/zero baseline is a reasonable stand-in at inference time
        bg = torch.zeros((8, *x.shape[1:]))
        ie = ImageExplainer(cnn, bg, device)
        p_img = float(ie.probs(x)[0])
        ie.save_overlays(x, ie.explain(x, cfg["shap"]["image_nsamples"]), out, [Path(args.image).stem])
        report["ultrasound"] = {"p_pcos": p_img}

    fusion_path = art / "models/fusion.joblib"
    if fusion_path.exists():
        p_final = LateFusion.load(fusion_path).predict_single(p_clin, p_img)
    else:
        p_final = float(np.mean([p for p in (p_clin, p_img) if p is not None]))

    report["final"] = {
        "p_pcos": round(p_final, 4),
        "risk_band": risk_band(p_final),
        "modalities_used": [m for m, p in (("clinical", p_clin), ("ultrasound", p_img)) if p is not None],
        "disclaimer": "Decision-support output only; not a diagnosis. Confirm with Rotterdam criteria.",
    }
    save_json(report, out / "report.json")
    print(f"P(PCOS) = {p_final:.3f}  [{risk_band(p_final)} risk]  -> {out / 'report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--clinical", help="JSON file of clinical feature values")
    ap.add_argument("--image", help="Ultrasound image path")
    ap.add_argument("--out", default="artifacts/predictions")
    a = ap.parse_args()
    if not (a.clinical or a.image):
        ap.error("Provide --clinical and/or --image")
    main(a)
