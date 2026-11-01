"""Fit the clinical preprocessor on a stratified train split and write processed tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clinical_loader import load_raw
from src.data.clinical_preprocess import ClinicalPreprocessor, add_engineered_features, feature_sets
from src.utils import ensure_dir, load_config, set_seed


def main() -> None:
    config = load_config()
    set_seed(int(config["seed"]))
    target_name = config["clinical"]["target_col"]
    features, target, meta = load_raw(config)
    features = add_engineered_features(features)
    splits = train_test_split(
        features,
        target,
        meta,
        test_size=float(config["clinical"]["test_size"]),
        random_state=int(config["seed"]),
        stratify=target,
    )
    x_train, x_test, y_train, y_test, meta_train, meta_test = splits
    preprocessor = ClinicalPreprocessor().fit(x_train)
    train_table = _pack(preprocessor.transform(x_train), y_train, meta_train, target_name)
    test_table = _pack(preprocessor.transform(x_test), y_test, meta_test, target_name)
    train_primary = train_table.loc[train_table["is_synthetic_inferred"] == 0].reset_index(drop=True)
    test_primary = test_table.loc[test_table["is_synthetic_inferred"] == 0].reset_index(drop=True)
    processed = ensure_dir(_project_path(config["paths"]["processed_dir"]))
    models = ensure_dir(_project_path(config["paths"]["artifacts_dir"]) / "models")
    written = [
        _write(train_table, processed / "clinical_train.parquet"),
        _write(test_table, processed / "clinical_test.parquet"),
        _write(train_primary, processed / "clinical_train_primary.parquet"),
        _write(test_primary, processed / "clinical_test_primary.parquet"),
        preprocessor.save(models / "clinical_preprocessor.joblib"),
    ]
    print(
        "rows: "
        f"total={len(target)} train={len(train_table)} test={len(test_table)} "
        f"primary_train={len(train_primary)} primary_test={len(test_primary)}"
    )
    balances = {
        "train": y_train,
        "test": y_test,
        "primary_train": train_primary[target_name],
        "primary_test": test_primary[target_name],
    }
    for name, labels in balances.items():
        counts = pd.Series(labels).value_counts().sort_index()
        text = ", ".join(f"{label}={int(count)}" for label, count in counts.items())
        print(f"class balance {name}: {text}")
    for name, columns in feature_sets(x_train).items():
        print(f"feature count {name}: {len(columns)}")
    for path in written:
        print(f"wrote {path}")


def _pack(features: pd.DataFrame, target: pd.Series, meta: pd.DataFrame, target_name: str) -> pd.DataFrame:
    table = features.reset_index(drop=True).copy()
    table[target_name] = pd.Series(target).reset_index(drop=True).to_numpy()
    table["is_synthetic_inferred"] = pd.Series(meta["is_synthetic_inferred"]).reset_index(drop=True).to_numpy()
    return table


def _write(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def _project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


if __name__ == "__main__":
    main()
