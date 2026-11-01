"""EfficientNet-B0 ultrasound classifier. The network returns one logit per image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_image_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    """ImageNet-initialized EfficientNet-B0 with dropout before a single logit."""
    if backbone != "efficientnet_b0":
        raise ValueError(f"Unsupported backbone '{backbone}'. Expected efficientnet_b0.")
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
    return model


class ProbWrapper(nn.Module):
    """Expose sigmoid probabilities for explanation code. Training still uses logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(inputs).reshape(-1))


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """One training pass. Returns the mean loss weighted by batch size."""
    model.train()
    total = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        targets = labels.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images).reshape(-1), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(targets.shape[0])
        seen += int(targets.shape[0])
    return total / seen if seen else 0.0


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the logit model and return sigmoid probabilities with integer labels."""
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).reshape(-1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy())
    if not probabilities:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    return np.concatenate(probabilities), np.concatenate(targets).astype(int)


def save_checkpoint(model, optimizer, epoch: int, val_auc: float, path) -> Path:
    """Save weights, optimizer state, epoch, and validation AUC."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": "efficientnet_b0",
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "epoch": int(epoch),
            "val_auc": float(val_auc),
        },
        destination,
    )
    return destination


def load_checkpoint(path, device) -> nn.Module:
    """Rebuild the logit model, load weights, and switch it to eval mode."""
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model = build_image_model(payload.get("backbone", "efficientnet_b0"), pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model
