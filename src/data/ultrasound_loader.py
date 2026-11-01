"""Torch dataset and ImageNet-normalized transforms for deduplicated ultrasound manifests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Grayscale ultrasound to 3 channels, resize, and apply ImageNet normalization."""
    steps: list = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
    ]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class UltrasoundDataset(Dataset):
    """Read manifest rows and return (image tensor, label)."""

    def __init__(self, manifest_df, transform: transforms.Compose | None = None) -> None:
        self.paths = [str(path) for path in manifest_df["image_path"].tolist()]
        self.labels = [int(label) for label in manifest_df["label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            converted = image.convert("RGB")
            tensor = self.transform(converted) if self.transform is not None else transforms.ToTensor()(converted)
        return tensor, self.labels[index]


def load_image_tensor(path: str | Path, img_size: int) -> torch.Tensor:
    """Load one ultrasound image with the evaluation transform, shape (3, H, W)."""
    with Image.open(path) as image:
        return build_transforms(img_size, train=False)(image.convert("RGB"))


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization for a CHW or NCHW tensor."""
    restored = tensor.detach().clone().float()
    channel_axis = 1 if restored.ndim == 4 else 0
    view = [1] * restored.ndim
    view[channel_axis] = 3
    mean = restored.new_tensor(IMAGENET_MEAN).view(*view)
    std = restored.new_tensor(IMAGENET_STD).view(*view)
    return (restored * std + mean).clamp(0, 1)
