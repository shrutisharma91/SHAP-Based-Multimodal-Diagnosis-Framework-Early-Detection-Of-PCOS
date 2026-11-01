"""Engineer clinical features and fit a train-only winsorise, log, impute, and scale pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HORMONE_COLUMNS = ("beta_hcg_1", "beta_hcg_2", "fsh", "lh", "tsh", "amh", "prl", "vit_d3", "prg")
ULTRASOUND_COLUMNS = ("total_follicles", "mean_follicle_size", "endometrium_mm", "follicle_n_L", "follicle_n_R")
HCG_COLUMNS = ("pregnant", "beta_hcg_1", "beta_hcg_2")
SKEW_THRESHOLD = 1.0


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add follicle, LH/FSH, waist-hip, and mean-arterial-pressure features when inputs exist."""
    frame = features.copy()
    if {"follicle_n_L", "follicle_n_R"}.issubset(frame.columns):
        frame["total_follicles"] = frame["follicle_n_L"] + frame["follicle_n_R"]
    size_columns = [column for column in ("follicle_size_l_mm", "follicle_size_r_mm") if column in frame.columns]
    if size_columns:
        frame["mean_follicle_size"] = frame[size_columns].mean(axis=1)
    if {"lh", "fsh"}.issubset(frame.columns):
        ratio = frame["lh"] / frame["fsh"].replace(0, np.nan)
        frame["lh_fsh_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    if {"waist_inch", "hip_inch"}.issubset(frame.columns):
        ratio = frame["waist_inch"] / frame["hip_inch"].replace(0, np.nan)
        frame["waist_hip_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    if {"bp_systolic", "bp_diastolic"}.issubset(frame.columns):
        frame["map_bp"] = (frame["bp_systolic"] + 2.0 * frame["bp_diastolic"]) / 3.0
    return frame


def feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    """Return the four nested clinical feature sets."""
    columns = list(features.columns)
    without_ultrasound = _without(columns, ULTRASOUND_COLUMNS)
    without_hcg = _without(without_ultrasound, HCG_COLUMNS)
    without_amh = _without(without_hcg, ("amh",))
    return {
        "A_full": columns,
        "B_no_ultrasound": without_ultrasound,
        "C_no_us_no_hcg": without_hcg,
        "D_no_amh": without_amh,
    }


class ClinicalPreprocessor:
    """Winsorise, log1p skewed hormones, median-impute, and scale using train statistics."""

    def __init__(self, skew_threshold: float = SKEW_THRESHOLD) -> None:
        self.skew_threshold = skew_threshold
        self.feature_columns_: list[str] = []
        self.binary_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.log_columns_: list[str] = []
        self.winsor_bounds_: dict[str, tuple[float, float]] = {}
        self.medians_: dict[str, float] = {}
        self.scaler_: StandardScaler | None = None

    def fit(self, features: pd.DataFrame) -> "ClinicalPreprocessor":
        frame = features.copy()
        self.feature_columns_ = list(frame.columns)
        self.binary_columns_ = [column for column in frame.columns if _is_binary(frame[column])]
        self.numeric_columns_ = [column for column in frame.columns if column not in self.binary_columns_]
        clipped = frame.copy()
        self.winsor_bounds_ = {}
        for column in self.numeric_columns_:
            values = pd.to_numeric(frame[column], errors="coerce")
            lower, upper = _percentile_bounds(values)
            self.winsor_bounds_[column] = (lower, upper)
            clipped[column] = values.clip(lower, upper)
        self.log_columns_ = []
        for column in self.numeric_columns_:
            if column not in HORMONE_COLUMNS:
                continue
            observed = clipped[column].dropna()
            if len(observed) < 3 or float(observed.min()) < 0 or float(observed.skew()) <= self.skew_threshold:
                continue
            self.log_columns_.append(column)
            clipped[column] = np.log1p(clipped[column].clip(lower=0))
        self.medians_ = {}
        for column in self.numeric_columns_:
            median = float(np.nanmedian(clipped[column].to_numpy(dtype=float)))
            self.medians_[column] = 0.0 if np.isnan(median) else median
            clipped[column] = clipped[column].fillna(self.medians_[column])
        self.scaler_ = StandardScaler()
        if self.numeric_columns_:
            self.scaler_.fit(clipped[self.numeric_columns_])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns_ if column not in features.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        frame = features.loc[:, self.feature_columns_].copy()
        for column in self.numeric_columns_:
            lower, upper = self.winsor_bounds_[column]
            values = pd.to_numeric(frame[column], errors="coerce").astype(float).clip(lower, upper)
            if column in self.log_columns_:
                values = np.log1p(values.clip(lower=0))
            frame[column] = values.fillna(self.medians_[column])
        if self.numeric_columns_ and self.scaler_ is not None:
            scaled = self.scaler_.transform(frame[self.numeric_columns_])
            for index, column in enumerate(self.numeric_columns_):
                frame[column] = scaled[:, index]
        return frame

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ClinicalPreprocessor":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, loaded {type(loaded).__name__}")
        return loaded


def _without(columns: list[str], dropped: tuple[str, ...]) -> list[str]:
    banned = set(dropped)
    return [column for column in columns if column not in banned]


def _is_binary(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    unique = set(np.round(values, 6))
    return unique.issubset({0.0, 1.0})


def _percentile_bounds(values: pd.Series) -> tuple[float, float]:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0
    lower, upper = np.percentile(finite, [1, 99])
    return float(lower), float(upper)
