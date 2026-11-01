"""Load the Kaggle clinical CSV pair or a preprocessed xlsx into X, y, and meta."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WITHOUT_INFERTILITY = "pcos_data_without_infertility.csv"
INFERTILITY = "pcos_infertility.csv"
TRAIN_SHEET = "model_train_scaled"
TEST_SHEET = "model_test_scaled"
ID_COLUMNS = ("patient_file_no", "sl_no")
DROP_FEATURES = {"fsh_lh"}
SPARSE_LIMIT = 0.5

_ALIASES = {
    "sl. no": "sl_no",
    "sl no": "sl_no",
    "patient file no.": "patient_file_no",
    "patient file no": "patient_file_no",
    "pcos (y/n)": "PCOS",
    "pcos": "PCOS",
    "age (yrs)": "age_yrs",
    "weight (kg)": "weight_kg",
    "height(cm)": "height_cm",
    "bmi": "bmi",
    "blood group": "blood_group",
    "pulse rate(bpm)": "pulse_bpm",
    "rr (breaths/min)": "rr_breaths_min",
    "hb(g/dl)": "hb_g_dl",
    "cycle(r/i)": "cycle_ri",
    "cycle length(days)": "cycle_length_days",
    "marraige status (yrs)": "marriage_yrs",
    "marriage status (yrs)": "marriage_yrs",
    "pregnant(y/n)": "pregnant",
    "pregnant": "pregnant",
    "no. of aborptions": "n_abortions",
    "no. of abortions": "n_abortions",
    "i beta-hcg(miu/ml)": "beta_hcg_1",
    "ii beta-hcg(miu/ml)": "beta_hcg_2",
    "beta_hcg_1": "beta_hcg_1",
    "beta_hcg_2": "beta_hcg_2",
    "fsh(miu/ml)": "fsh",
    "lh(miu/ml)": "lh",
    "fsh/lh": "fsh_lh",
    "hip(inch)": "hip_inch",
    "waist(inch)": "waist_inch",
    "waist:hip ratio": "waist_hip_ratio",
    "waist_hip_ratio": "waist_hip_ratio",
    "tsh (miu/l)": "tsh",
    "amh(ng/ml)": "amh",
    "amh": "amh",
    "prl(ng/ml)": "prl",
    "vit d3 (ng/ml)": "vit_d3",
    "prg(ng/ml)": "prg",
    "rbs(mg/dl)": "rbs",
    "weight gain(y/n)": "weight_gain",
    "hair growth(y/n)": "hair_growth",
    "skin darkening (y/n)": "skin_darkening",
    "hair loss(y/n)": "hair_loss",
    "pimples(y/n)": "pimples",
    "fast food (y/n)": "fast_food",
    "reg.exercise(y/n)": "reg_exercise",
    "bp _systolic (mmhg)": "bp_systolic",
    "bp _diastolic (mmhg)": "bp_diastolic",
    "bp_systolic": "bp_systolic",
    "bp_diastolic": "bp_diastolic",
    "follicle no. (l)": "follicle_n_L",
    "follicle no. (r)": "follicle_n_R",
    "follicle_n_l": "follicle_n_L",
    "follicle_n_r": "follicle_n_R",
    "avg. f size (l) (mm)": "follicle_size_l_mm",
    "avg. f size (r) (mm)": "follicle_size_r_mm",
    "follicle_size_l_mm": "follicle_size_l_mm",
    "follicle_size_r_mm": "follicle_size_r_mm",
    "endometrium (mm)": "endometrium_mm",
    "endometrium_mm": "endometrium_mm",
    "is_synthetic_inferred": "is_synthetic_inferred",
    "is_synthetic": "is_synthetic_inferred",
    "synthetic": "is_synthetic_inferred",
}


def detect_source(path: str | Path) -> str:
    """Return 'xlsx' or 'csv_pair' for a clinical file or directory."""
    location = Path(path)
    if location.is_file():
        if location.suffix.lower() == ".xlsx":
            return "xlsx"
        raise ValueError(f"Unsupported clinical file: {location.name}")
    if not location.is_dir():
        raise FileNotFoundError(f"Clinical path does not exist: {location}")
    workbooks = sorted(location.glob("*.xlsx"))
    csv_names = {item.name.lower() for item in location.glob("*.csv")}
    has_pair = WITHOUT_INFERTILITY in csv_names and INFERTILITY in csv_names
    if workbooks and has_pair:
        raise ValueError(f"Both an xlsx workbook and the Kaggle CSV pair are in {location}. Leave one source.")
    if len(workbooks) > 1:
        names = ", ".join(item.name for item in workbooks)
        raise ValueError(f"Multiple xlsx workbooks in {location}: {names}")
    if len(workbooks) == 1:
        return "xlsx"
    if has_pair:
        return "csv_pair"
    raise FileNotFoundError(
        f"No clinical source in {location}. Expected one xlsx or {WITHOUT_INFERTILITY} and {INFERTILITY}."
    )


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features, the PCOS target, and meta with is_synthetic_inferred."""
    location = _resolve(cfg["paths"]["clinical_raw_dir"])
    target = cfg["clinical"]["target_col"]
    if detect_source(location) == "xlsx":
        workbook = location if location.is_file() else sorted(location.glob("*.xlsx"))[0]
        frame, flags = _load_xlsx(workbook)
    else:
        frame = _load_csv_pair(location)
        flags = pd.Series(0, index=frame.index, dtype=int)
    return _finalize(frame, flags, target)


def _load_csv_pair(directory: Path) -> pd.DataFrame:
    files = {item.name.lower(): item for item in directory.glob("*.csv")}
    left = _prepare_table(_read_csv(files[WITHOUT_INFERTILITY]))
    right = _prepare_table(_read_csv(files[INFERTILITY]))
    key = "patient_file_no"
    if key not in left.columns or key not in right.columns:
        raise KeyError("Both clinical CSVs need a Patient File No. column.")
    left[key] = pd.to_numeric(left[key], errors="coerce")
    right[key] = pd.to_numeric(right[key], errors="coerce")
    right = right.drop_duplicates(subset=[key], keep="first")
    merged = left.merge(right, on=key, how="left", suffixes=("", "_dup"))
    duplicate_cols = [column for column in merged.columns if column.endswith("_dup")]
    return merged.drop(columns=duplicate_cols)


def _load_xlsx(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    workbook = pd.ExcelFile(path)
    sheets = {name.lower(): name for name in workbook.sheet_names}
    missing = [name for name in (TRAIN_SHEET, TEST_SHEET) if name not in sheets]
    if missing:
        raise ValueError(f"{path.name} is missing {missing}. Found sheets: {workbook.sheet_names}")
    train = _prepare_table(pd.read_excel(workbook, sheets[TRAIN_SHEET]))
    test = _prepare_table(pd.read_excel(workbook, sheets[TEST_SHEET]))
    frame = pd.concat([train, test], ignore_index=True)
    smote_names = [name for lower, name in sheets.items() if "smote" in lower]
    if len(smote_names) > 1:
        raise ValueError(f"Multiple SMOTE sheets in {path.name}: {smote_names}")
    smote = _prepare_table(pd.read_excel(workbook, smote_names[0])) if smote_names else None
    return _synthetic_flags(frame, smote)


def _synthetic_flags(frame: pd.DataFrame, smote: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    frame = frame.copy()
    if "is_synthetic_inferred" in frame.columns:
        flags = _as_flag(frame["is_synthetic_inferred"])
        return frame.drop(columns=["is_synthetic_inferred"]), flags
    flags = pd.Series(0, index=frame.index, dtype=int)
    if smote is None or smote.empty:
        return frame, flags
    excluded = set(ID_COLUMNS) | {"PCOS", "is_synthetic_inferred", "fsh_lh"}
    shared = [column for column in frame.columns if column in smote.columns and column not in excluded]
    if not shared:
        return frame, flags
    observed = _signature(frame, shared)
    synthetic = set(_signature(smote, shared))
    matched = observed.isin(synthetic)
    if float(matched.mean()) > 0.5:
        return frame, flags
    return frame, matched.astype(int)


def _finalize(frame: pd.DataFrame, flags: pd.Series, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} not found. Columns: {list(frame.columns)}")
    working = frame.copy()
    y = _coerce_target(working[target])
    features = working.drop(columns=[target])
    meta = pd.DataFrame({"is_synthetic_inferred": flags.reindex(working.index).fillna(0).astype(int).to_numpy()})
    for column in ID_COLUMNS:
        if column in features.columns:
            meta[column] = features[column].to_numpy()
            features = features.drop(columns=[column])
    features = features.drop(columns=[column for column in DROP_FEATURES if column in features.columns])
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.loc[:, features.isna().mean() <= SPARSE_LIMIT]
    valid = y.notna()
    features = features.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if not y.empty and (y % 1 == 0).all():
        y = y.astype(int)
    y.name = target
    meta = meta.loc[valid].reset_index(drop=True)
    return features, y, meta


def _prepare_table(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [_canonical(str(column).replace("\n", " ").strip()) for column in renamed.columns]
    renamed = renamed.loc[:, ~renamed.columns.duplicated()]
    return renamed.dropna(how="all")


def _canonical(header: str) -> str:
    key = " ".join(header.replace("\ufeff", "").replace("\xa0", " ").split()).lower()
    if key in _ALIASES:
        return _ALIASES[key]
    cleaned = "".join(char if char.isalnum() else "_" for char in key)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _signature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    rounded = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").round(5)

    def _cell(value: object) -> str:
        if pd.isna(value):
            return "na"
        return f"{float(value):.5f}"

    text = rounded.apply(lambda column: column.map(_cell))
    return text.agg("|".join, axis=1)


def _as_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _coerce_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0})
    values = numeric.fillna(labels)
    whole = values.dropna()
    if not whole.empty and (whole % 1 == 0).all():
        return values.round().astype("Int64")
    return values


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path}") from last_error


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    from_cwd = Path.cwd() / candidate
    if from_cwd.exists():
        return from_cwd
    return Path(__file__).resolve().parents[2] / candidate
