"""Synthetic checks for the clinical preprocessor, model, SHAP, and faithfulness drop."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.clinical_preprocess import ClinicalPreprocessor, add_engineered_features
from src.explain.faithfulness import top_k_drop_curve
from src.explain.shap_clinical import ClinicalExplainer
from src.models.clinical_model import build_xgb

ENGINEERED = ("total_follicles", "mean_follicle_size", "lh_fsh_ratio", "waist_hip_ratio", "map_bp")


def test_preprocessor_shape_and_no_nans():
    features, _ = _synthetic_frame()
    assert set(ENGINEERED).issubset(features.columns)
    train = features.iloc[:200]
    held = features.iloc[200:]
    preprocessor = ClinicalPreprocessor().fit(train)
    transformed = preprocessor.transform(held)
    assert transformed.shape == (len(held), len(preprocessor.feature_columns_))
    assert int(transformed.isna().to_numpy().sum()) == 0


def test_xgboost_auc_on_synthetic():
    model, x_test, y_test, _ = _fitted()
    scores = model.predict_proba(x_test.to_numpy(dtype=float))[:, 1]
    assert roc_auc_score(y_test, scores) > 0.8


def test_clinical_explainer_shape():
    _, x_test, _, values = _fitted()
    assert values.shape == (len(x_test), x_test.shape[1])


def test_top_k_drop_falls_faster_than_random():
    model, x_test, y_test, values = _fitted()
    curve = top_k_drop_curve(model, x_test, y_test, values, ks=(1,))
    shap_auc = curve.loc[curve["strategy"] == "shap", "auc"].iloc[0]
    random_auc = curve.loc[curve["strategy"] == "random", "auc"].iloc[0]
    assert shap_auc < random_auc


@lru_cache(maxsize=1)
def _fitted():
    features, labels = _synthetic_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=0, stratify=labels
    )
    preprocessor = ClinicalPreprocessor().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    model = build_xgb({"n_estimators": 50}, y_train, seed=0)
    model.fit(x_train.to_numpy(dtype=float), y_train)
    background = x_train.sample(n=80, random_state=0)
    values = np.asarray(ClinicalExplainer(model, background).explain(x_test).values)
    return model, x_test, y_test, values


def _synthetic_frame(n: int = 280, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[n // 2 :] = 1
    rng.shuffle(labels)
    amh = np.where(labels == 1, rng.normal(12.0, 0.3, n), rng.normal(2.0, 0.3, n))
    columns = {
        "amh": amh,
        "age_yrs": rng.normal(30, 4, n),
        "fsh": rng.uniform(3, 10, n),
        "lh": rng.uniform(4, 12, n),
        "follicle_n_L": rng.uniform(2, 14, n),
        "follicle_n_R": rng.uniform(2, 14, n),
        "follicle_size_l_mm": rng.uniform(3, 8, n),
        "follicle_size_r_mm": rng.uniform(3, 8, n),
        "waist_inch": rng.uniform(28, 40, n),
        "hip_inch": rng.uniform(34, 46, n),
        "bp_systolic": rng.normal(118, 8, n),
        "bp_diastolic": rng.normal(76, 6, n),
    }
    for index in range(6):
        columns[f"noise_{index}"] = rng.normal(size=n)
    frame = add_engineered_features(pd.DataFrame(columns))
    frame.loc[frame.index[:4], "age_yrs"] = np.nan
    frame.loc[frame.index[-4:], "age_yrs"] = np.nan
    return frame, pd.Series(labels, name="PCOS")
