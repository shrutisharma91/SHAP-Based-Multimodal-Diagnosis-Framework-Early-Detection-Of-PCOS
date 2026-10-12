"""Ovarian ultrasound image datasets.

Expected layout (matches the public Kaggle "PCOS detection using ultrasound
images" dataset):

    data/raw/ultrasound/
        infected/      *.jpg   -> label 1 (PCOS)
        notinfected/   *.jpg   -> label 0
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # ultrasound is single-channel
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) normalized -> [0,1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def index_folder(root: str | Path, positive_class: str) -> pd.DataFrame:
    """Scan class subfolders -> DataFrame[image_path, label]."""
    rows = []
    for cls_dir in sorted(Path(root).iterdir()):
        if not cls_dir.is_dir():
            continue
        label = int(cls_dir.name == positive_class)
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"image_path": str(p), "label": label})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    return pd.DataFrame(rows)


class UltrasoundDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), torch.tensor(self.labels[i])


def load_image_tensor(path: str | Path, img_size: int) -> torch.Tensor:
    """Single image -> (1,3,H,W) tensor for inference."""
    img = Image.open(path).convert("L")
    return build_transforms(img_size, train=False)(img).unsqueeze(0)
