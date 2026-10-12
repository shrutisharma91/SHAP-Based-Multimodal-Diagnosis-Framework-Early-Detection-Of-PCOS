"""Smoke tests on synthetic data — no real dataset needed.

    pytest -q
"""
import numpy as np
import pandas as pd

from src.data.clinical import ClinicalPreprocessor, _engineer
from src.evaluation.metrics import compute_metrics
from src.explain.shap_clinical import ClinicalExplainer
from src.models.clinical_model import build_clinical_model
from src.models.fusion import LateFusion


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "Follicle No. (L)": rng.poisson(6 + 6 * y),
        "Follicle No. (R)": rng.poisson(6 + 6 * y),
        "LH(mIU/mL)": rng.normal(5 + 4 * y, 1.5),
        "FSH(mIU/mL)": rng.normal(6, 1.0, n).clip(1),
        "AMH(ng/mL)": rng.normal(3 + 4 * y, 1.5),
        "BMI": rng.normal(23 + 3 * y, 3),
    })
    X.loc[::17, "AMH(ng/mL)"] = np.nan
    return _engineer(X), pd.Series(y)


def test_clinical_model_and_shap():
    X, y = _synthetic()
    Xp = ClinicalPreprocessor().fit_transform(X)
    assert not Xp.isna().any().any()

    model = build_clinical_model({"n_estimators": 50, "max_depth": 3}, y, 0).fit(Xp, y)
    p = model.predict_proba(Xp)[:, 1]
    assert compute_metrics(y, p)["roc_auc"] > 0.8

    sv = ClinicalExplainer(model, background=Xp.sample(50, random_state=0)).explain(Xp.iloc[:10])
    assert sv.values.shape == (10, Xp.shape[1])
    reasons = ClinicalExplainer.patient_reasons(sv, 0)
    assert set(reasons) == {"increases_risk", "decreases_risk"}


def test_fusion_strategies_and_missing_modality():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p1 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 300), 0.01, 0.99)
    p2 = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.2, 300), 0.01, 0.99)

    for strategy in ("stacking", "weighted"):
        f = LateFusion(strategy).fit(p1, p2, y)
        fused = f.predict_proba(np.column_stack([p1, p2]))
        assert fused.shape == (300,) and ((0 <= fused) & (fused <= 1)).all()

    f = LateFusion("stacking").fit(p1, p2, y)
    assert f.predict_single(0.8, None) == 0.8
    assert f.predict_single(None, 0.2) == 0.2
