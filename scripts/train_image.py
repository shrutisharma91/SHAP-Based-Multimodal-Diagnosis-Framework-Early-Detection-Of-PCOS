"""Train + evaluate + explain the ultrasound CNN.

    python -m scripts.train_image --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.ultrasound import UltrasoundDataset, build_transforms, index_folder, load_image_tensor
from src.evaluation.metrics import compute_metrics, plot_confusion, plot_roc
from src.explain.shap_image import ImageExplainer
from src.models.image_model import build_image_model, predict_proba, save_checkpoint, train_one_epoch
from src.utils import ensure_dir, get_device, get_logger, load_config, save_json, set_seed

log = get_logger("train_image")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    ic, art, device = cfg["image"], Path(cfg["paths"]["artifacts"]), get_device()
    ensure_dir(art / "models")

    df = index_folder(cfg["paths"]["ultrasound_dir"], ic["positive_class"])
    # NOTE: if your dataset has several frames per patient, split by patient id
    # (GroupShuffleSplit) instead — otherwise leakage inflates accuracy.
    tr_df, va_df = train_test_split(df, test_size=ic["val_split"], stratify=df.label, random_state=cfg["seed"])
    log.info("Images: train=%d val=%d (PCOS=%d)", len(tr_df), len(va_df), int(df.label.sum()))

    tr_dl = DataLoader(UltrasoundDataset(tr_df, build_transforms(ic["img_size"], True)),
                       batch_size=ic["batch_size"], shuffle=True, num_workers=ic["num_workers"])
    va_ds = UltrasoundDataset(va_df, build_transforms(ic["img_size"], False))
    va_dl = DataLoader(va_ds, batch_size=ic["batch_size"], num_workers=ic["num_workers"])

    model = build_image_model(ic["backbone"]).to(device)
    pos = tr_df.label.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(tr_df) - pos) / max(pos, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=ic["lr"], weight_decay=ic["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ic["epochs"])

    best_auc, ckpt = -1.0, art / "models/ultrasound_cnn.pt"
    for epoch in range(1, ic["epochs"] + 1):
        loss = train_one_epoch(model, tr_dl, optimizer, criterion, device)
        scheduler.step()
        p, y = predict_proba(model, va_dl, device)
        m = compute_metrics(y.numpy(), p.numpy())
        log.info("epoch %02d | loss %.4f | val AUC %.4f | recall %.3f", epoch, loss, m["roc_auc"], m["recall_sensitivity"])
        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            save_checkpoint(model, ic["backbone"], ckpt)

    # ---- final eval with best weights ----
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    p, y = predict_proba(model, va_dl, device)
    save_json(compute_metrics(y.numpy(), p.numpy()), art / "reports/ultrasound_metrics.json")
    plot_roc({"Ultrasound (CNN)": (y.numpy(), p.numpy())}, art / "figures/ultrasound_roc.png")
    plot_confusion(y.numpy(), p.numpy(), art / "figures/ultrasound_confusion.png")

    # ---- SHAP on a few validation scans ----
    sc = cfg["shap"]
    bg_dl = DataLoader(UltrasoundDataset(tr_df.sample(sc["image_background"], random_state=cfg["seed"]),
                                         build_transforms(ic["img_size"], False)), batch_size=sc["image_background"])
    background, _ = next(iter(bg_dl))
    sample_df = va_df.sample(min(sc["image_samples"], len(va_df)), random_state=cfg["seed"])
    x = torch.cat([load_image_tensor(pth, ic["img_size"]) for pth in sample_df.image_path])

    explainer = ImageExplainer(model, background, device)
    sv = explainer.explain(x, nsamples=sc["image_nsamples"])
    names = [Path(pth).stem for pth in sample_df.image_path]
    explainer.save_overlays(x, sv, art / "figures/shap", names)
    log.info("Done. Best val AUC %.4f", best_auc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
