"""CNN for ovarian ultrasound classification (transfer learning)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.utils import get_logger

log = get_logger(__name__)


def build_image_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    """Returns a model emitting a single logit (PCOS vs normal)."""
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 1))
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m


class ProbWrapper(nn.Module):
    """Logit -> probability, shape (N,1). SHAP explains this output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).squeeze(1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def save_checkpoint(model: nn.Module, backbone: str, path) -> None:
    torch.save({"backbone": backbone, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = build_image_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
