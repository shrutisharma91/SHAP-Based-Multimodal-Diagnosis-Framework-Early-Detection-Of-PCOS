"""Synthetic checks for the ultrasound model, dataset, and hash grouping."""

from __future__ import annotations

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.ultrasound_dedupe import _group_ids
from src.data.ultrasound_loader import UltrasoundDataset, build_transforms
from src.models.image_model import build_image_model


def test_image_model_forward_shape():
    model = build_image_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 224, 224))
    assert tuple(output.shape) == (2, 1)


def test_ultrasound_dataset_item(tmp_path):
    path = tmp_path / "scan.jpg"
    Image.new("L", (40, 30), color=90).save(path)
    manifest = pd.DataFrame({"image_path": [str(path)], "label": [1]})
    dataset = UltrasoundDataset(manifest, build_transforms(224, train=False))
    tensor, label = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(label, int)
    assert tensor.shape == (3, 224, 224)
    assert label == 1


def test_union_find_merges_near_duplicate_hashes():
    near_a = _hash(())
    near_b = _hash((0,))
    near_c = _hash((0, 1))
    far = _hash(tuple(range(100, 120)))
    groups = _group_ids([near_a, near_b, near_c, far], threshold=1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def _hash(on_bits: tuple[int, ...]) -> imagehash.ImageHash:
    bits = np.zeros(256, dtype=bool)
    bits[list(on_bits)] = True
    return imagehash.ImageHash(bits.reshape(16, 16))
