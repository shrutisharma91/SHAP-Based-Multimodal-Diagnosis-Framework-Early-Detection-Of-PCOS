"""Run clinical and image faithfulness checks. Execution starts only from main()."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.clinical_preprocess import feature_sets
from src.data.ultrasound_loader import UltrasoundDataset, build_transforms
from src.explain.faithfulness import (
    patch_drop_curve,
    rotterdam_alignment,
    stability_across_folds,
    stability_across_seeds,
    top_k_drop_curve,
)
from src.explain.shap_clinical import ClinicalExplainer
from src.explain.shap_image import ImageExplainer
from src.models.clinical_model import build_logreg, build_rf, build_xgb, load_model
from src.models.image_model import load_checkpoint
from src.utils import ensure_dir, get_device, load_config, load_json, save_json, set_seed
from scripts.train_image import _split_train_groups


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    set_seed(seed)
    artifacts = _located(config["paths"]["artifacts_dir"])
    figures = ensure_dir(artifacts / "figures" / "faithfulness")
    clinical = _clinical_report(config, artifacts, seed)
    image = _image_report(config, artifacts, seed)
    report = {"clinical": clinical, "image": image}
    save_json(_clean(report), ensure_dir(artifacts / "reports") / "faithfulness_report.json")
    _plot_curve(pd.DataFrame(clinical["drop_curve"]), figures / "clinical_drop_curve.png", "Clinical top-k drop")
    _plot_curve(pd.DataFrame(image["drop_curve"]), figures / "image_patch_drop_curve.png", "Image 8x8 patch drop")
    _print_summary(report)


def _clinical_report(config: dict, artifacts: Path, seed: int) -> dict:
    set_id, set_key, algo = _selected_clinical(artifacts)
    processed = _located(config["paths"]["processed_dir"])
    train_frame = pd.read_parquet(processed / "clinical_train.parquet")
    test_frame = pd.read_parquet(processed / "clinical_test.parquet")
    train_x, train_y = _xy(train_frame, config["clinical"]["target_col"])
    test_x, test_y = _xy(test_frame, config["clinical"]["target_col"])
    columns = feature_sets(train_x)[set_key]
    x_train, x_test = train_x.loc[:, columns], test_x.loc[:, columns]
    model = load_model(artifacts / "models" / f"clinical_{set_id}_{algo}.joblib")
    background = x_train.sample(n=min(100, len(x_train)), random_state=seed)
    explanation = ClinicalExplainer(model, background).explain(x_test)
    values = np.asarray(explanation.values, dtype=float)
    curve = top_k_drop_curve(model, x_test, test_y, values, impute="median")
    importance = pd.DataFrame({"feature": columns, "importance": np.abs(values).mean(axis=0)})

    def builder(model_seed: int):
        if algo == "rf":
            return build_rf(config["clinical"]["rf"], model_seed)
        if algo == "logreg":
            return build_logreg(model_seed)
        return build_xgb(config["clinical"]["xgb"], train_y, model_seed)

    return {
        "model": f"clinical_{set_id}_{algo}.joblib",
        "feature_set": set_key,
        "impute": "median",
        "drop_curve": curve.to_dict(orient="records"),
        "stability_folds": stability_across_folds(builder, x_train, train_y, folds=int(config["clinical"]["cv_folds"]), seed=seed),
        "stability_seeds": stability_across_seeds(builder, x_train, train_y),
        "rotterdam_alignment": rotterdam_alignment(importance),
        "top_features": importance.sort_values("importance", ascending=False).head(10).to_dict(orient="records"),
    }


def _image_report(config: dict, artifacts: Path, seed: int) -> dict:
    device = get_device()
    manifest = pd.read_csv(Path.cwd() / "data" / "interim" / "ultrasound_manifest.csv")
    train_side = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    fit_frame, _ = _split_train_groups(train_side, seed)
    img_size = int(config["image"]["img_size"])
    transform = build_transforms(img_size, train=False)
    fit_data = UltrasoundDataset(fit_frame, transform)
    test_data = UltrasoundDataset(test_frame, transform)
    generator = np.random.default_rng(seed)
    background_n = min(int(config["shap"]["image_background"]), len(fit_data))
    background_idx = generator.choice(len(fit_data), size=background_n, replace=False)
    background = torch.stack([fit_data[int(index)][0] for index in background_idx])
    images = torch.stack([test_data[index][0] for index in range(len(test_data))])
    labels = np.asarray(test_data.labels, dtype=int)
    model = load_checkpoint(artifacts / "models" / "ultrasound_cnn.pt", device)
    explainer = ImageExplainer(model, background, device)
    parts = []
    batch_size = int(config["image"]["batch_size"])
    nsamples = int(config["shap"]["image_nsamples"])
    for start in range(0, len(images), batch_size):
        parts.append(explainer.explain(images[start : start + batch_size], nsamples=nsamples))
    curve = patch_drop_curve(model, images, labels, np.concatenate(parts, axis=0), patch=8, device=device)
    return {"model": "ultrasound_cnn.pt", "patch": 8, "drop_curve": curve.to_dict(orient="records")}


def _selected_clinical(artifacts: Path) -> tuple[str, str, str]:
    set_id, algo = "A", "xgb"
    report_path = artifacts / "reports" / "clinical_results.json"
    if report_path.exists():
        chosen = load_json(report_path).get("sets", {}).get("A", {})
        algo = chosen.get("best_algo", algo)
    return set_id, "A_full", algo


def _xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    drop = [column for column in (target, "is_synthetic_inferred", "patient_file_no", "sl_no") if column in frame.columns]
    return frame.drop(columns=drop), frame[target].to_numpy(dtype=int)


def _plot_curve(frame: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure()
    for strategy, group in frame.groupby("strategy"):
        ordered = group.sort_values("k")
        plt.plot(ordered["k"], ordered["auc"], marker="o", label=str(strategy))
    plt.xlabel("k")
    plt.ylabel("AUC")
    plt.title(title)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _print_summary(report: dict) -> None:
    clinical, image = report["clinical"], report["image"]
    print("| check | value |")
    print("|---|---|")
    print(f"| clinical model | {clinical['model']} |")
    print(f"| fold stability mean rho | {clinical['stability_folds']['mean_rho']:.3f} |")
    print(f"| seed stability mean rho | {clinical['stability_seeds']['mean_rho']:.3f} |")
    print(f"| Rotterdam top-10 fraction | {clinical['rotterdam_alignment']:.3f} |")
    print("\n| k | clinical shap AUC | clinical random AUC | image shap AUC | image random AUC |")
    print("|---|---|---|---|---|")
    clinical_curve = pd.DataFrame(clinical["drop_curve"])
    image_curve = pd.DataFrame(image["drop_curve"])
    for k in sorted(clinical_curve["k"].unique()):
        print(
            f"| {int(k)} | {_auc_at(clinical_curve, k, 'shap')} | {_auc_at(clinical_curve, k, 'random')} | "
            f"{_auc_at(image_curve, k, 'shap')} | {_auc_at(image_curve, k, 'random')} |"
        )


def _auc_at(frame: pd.DataFrame, k: int, strategy: str) -> str:
    values = frame.loc[(frame["k"] == k) & (frame["strategy"] == strategy), "auc"]
    return f"{float(values.iloc[0]):.3f}" if not values.empty else ""


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
