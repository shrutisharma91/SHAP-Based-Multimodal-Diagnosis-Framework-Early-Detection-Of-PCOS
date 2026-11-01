"""Fine-tune the ultrasound CNN. The training loop runs only from main()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.data.ultrasound_loader import UltrasoundDataset, build_transforms
from src.evaluation.metrics import best_threshold, compute_metrics
from src.explain.shap_image import ImageExplainer
from src.models.image_model import build_image_model, evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from src.utils import ensure_dir, load_config, save_json, set_seed


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    set_seed(seed)
    image_config = config["image"]
    shap_config = config["shap"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(Path.cwd() / "data" / "interim" / "ultrasound_manifest.csv")
    train_side = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    if train_side.empty or test_frame.empty:
        raise FileNotFoundError("The ultrasound manifest needs both a train split and a held-out test split.")
    fit_frame, val_frame = _split_train_groups(train_side, seed)
    img_size = int(image_config["img_size"])
    batch_size = int(image_config["batch_size"])
    workers = int(image_config["num_workers"])
    fit_loader = _loader(fit_frame, img_size, batch_size, workers, train=True)
    val_loader = _loader(val_frame, img_size, batch_size, workers, train=False)
    test_loader = _loader(test_frame, img_size, batch_size, workers, train=False)
    model = build_image_model(image_config["backbone"], pretrained=True).to(device)
    labels = fit_frame["label"].to_numpy(dtype=int)
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("The group-aware training split has no positive images.")
    pos_weight = torch.tensor([(labels == 0).sum() / positives], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(image_config["lr"]), weight_decay=float(image_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(image_config["epochs"]))
    artifacts = _located(config["paths"]["artifacts_dir"])
    checkpoint_path = ensure_dir(artifacts / "models") / "ultrasound_cnn.pt"
    best_auc = -1.0
    best_epoch = 0
    patience_left = 5
    epochs_ran = 0
    for epoch in range(1, int(image_config["epochs"]) + 1):
        loss = train_one_epoch(model, fit_loader, optimizer, criterion, device)
        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auc = _auc(val_labels, val_probs)
        scheduler.step()
        epochs_ran = epoch
        print(f"epoch {epoch} loss={loss:.4f} val_auc={val_auc:.4f}")
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_left = 5
            save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved because validation AUC was undefined.")
    model = load_checkpoint(checkpoint_path, device)
    val_probs, val_labels = evaluate(model, val_loader, device)
    threshold = best_threshold(val_labels, val_probs, min_recall=0.9)
    test_probs, test_labels = evaluate(model, test_loader, device)
    report = {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "epochs_ran": epochs_ran,
        "threshold": threshold,
        "pos_weight": float(pos_weight.item()),
        "n_train": int(len(fit_frame)),
        "n_val": int(len(val_frame)),
        "n_test": int(len(test_frame)),
        "test": compute_metrics(test_labels, test_probs, threshold),
    }
    save_json(report, ensure_dir(artifacts / "reports") / "image_results.json")
    _write_shap(model, fit_frame, test_frame, img_size, device, shap_config, artifacts / "figures" / "image", seed)
    print(
        f"test auc={report['test']['roc_auc']:.4f} recall={report['test']['recall']:.4f} "
        f"threshold={threshold:.4f} checkpoint={checkpoint_path}"
    )


def _split_train_groups(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every group_id on one side of an approximately 80/20 training/validation cut."""
    indices = np.arange(len(frame))
    labels = frame["label"].to_numpy()
    groups = frame["group_id"].to_numpy()
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_idx, val_idx = next(splitter.split(indices, labels, groups))
    except ValueError:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        fit_idx, val_idx = next(splitter.split(indices, labels, groups))
    return frame.iloc[fit_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)


def _loader(frame: pd.DataFrame, img_size: int, batch_size: int, workers: int, train: bool) -> DataLoader:
    dataset = UltrasoundDataset(frame, build_transforms(img_size, train=train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers)


def _write_shap(model, fit_frame, test_frame, img_size, device, shap_config, out_dir, seed) -> None:
    background_n = min(int(shap_config["image_background"]), len(fit_frame))
    sample_n = min(int(shap_config["image_samples"]), len(test_frame))
    generator = np.random.default_rng(seed)
    fit_dataset = UltrasoundDataset(fit_frame, build_transforms(img_size, train=False))
    test_dataset = UltrasoundDataset(test_frame, build_transforms(img_size, train=False))
    background_idx = generator.choice(len(fit_dataset), size=background_n, replace=False)
    test_idx = generator.choice(len(test_dataset), size=sample_n, replace=False)
    background = torch.stack([fit_dataset[int(index)][0] for index in background_idx])
    images = torch.stack([test_dataset[int(index)][0] for index in test_idx])
    names = [Path(test_frame.iloc[int(index)]["image_path"]).stem for index in test_idx]
    explainer = ImageExplainer(model, background, device)
    values = explainer.explain(images, nsamples=int(shap_config["image_nsamples"]))
    explainer.save_overlays(images, values, out_dir, names)


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def _located(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


if __name__ == "__main__":
    main()
