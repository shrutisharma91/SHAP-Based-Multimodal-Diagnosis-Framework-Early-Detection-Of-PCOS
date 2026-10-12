"""Clinical / tabular data loading and preprocessing.

Designed around the public Kaggle "PCOS dataset" (541 patients, 44 features:
hormonal, metabolic, anthropometric, ultrasound follicle counts) but works
with any CSV/XLSX that has a binary target column.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, get_logger

log = get_logger(__name__)


def _read_raw(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name or 0)
    else:
        df = pd.read_csv(path)
    # Kaggle columns carry trailing spaces, e.g. "Cycle(R/I) "
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features clinicians actually use for PCOS (Rotterdam criteria)."""
    out = df.copy()
    fl, fr = "Follicle No. (L)", "Follicle No. (R)"
    if fl in out and fr in out:
        out["Total Follicles"] = out[fl] + out[fr]
        out["Follicle Asymmetry"] = (out[fl] - out[fr]).abs()

    lh, fsh = "LH(mIU/mL)", "FSH(mIU/mL)"
    if lh in out and fsh in out:
        out["LH/FSH"] = out[lh] / out[fsh].replace(0, np.nan)

    waist, hip = "Waist(inch)", "Hip(inch)"
    if waist in out and hip in out and "Waist:Hip Ratio" not in out:
        out["Waist:Hip Ratio"] = out[waist] / out[hip].replace(0, np.nan)
    return out


def load_clinical(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, patient_ids) with all features numeric, NaNs preserved."""
    c = cfg["clinical"]
    df = _read_raw(cfg["paths"]["clinical_data"], c.get("sheet_name"))

    drop = [col for col in c.get("drop_cols", []) if col in df.columns]
    df = df.drop(columns=drop)
    df = df.dropna(subset=[c["target"]])

    ids = df[c["id_col"]].astype(str) if c["id_col"] in df else pd.Series(df.index.astype(str))
    y = df[c["target"]].astype(int)
    X = df.drop(columns=[col for col in (c["target"], c["id_col"]) if col in df])

    # Some numeric columns ship as strings ("1.99." / "a") -> coerce, bad values become NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.5]  # drop near-empty columns
    X = _engineer(X)
    X = X.replace([np.inf, -np.inf], np.nan)

    log.info("Clinical data: %d patients, %d features, PCOS prevalence %.1f%%",
             len(X), X.shape[1], 100 * y.mean())
    return X.reset_index(drop=True), y.reset_index(drop=True), ids.reset_index(drop=True)


class ClinicalPreprocessor:
    """Median imputation fitted on train only; keeps a DataFrame interface so
    SHAP plots retain real feature names."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame) -> "ClinicalPreprocessor":
        self.columns = list(X.columns)
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.columns)
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        ensure_dir(Path(path).parent)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ClinicalPreprocessor":
        return joblib.load(path)


def split_clinical(X, y, ids, test_size: float, seed: int):
    return train_test_split(X, y, ids, test_size=test_size, stratify=y, random_state=seed)
