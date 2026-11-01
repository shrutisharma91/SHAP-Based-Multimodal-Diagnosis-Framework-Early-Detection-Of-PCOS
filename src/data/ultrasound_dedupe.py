"""Group ultrasound images by perceptual hash and remove cross-split near-duplicates."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

PHASH_HASH_SIZE = 16
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
MANIFEST_COLUMNS = ["image_path", "split", "label", "phash", "group_id"]
LABELS = (("notinfected", 0), ("infected", 1))
LOGGER = logging.getLogger(__name__)


def compute_phash(path: str | Path) -> imagehash.ImageHash:
    """Return a 16x16 perceptual hash. imagehash stores an 8x8 hash when hash_size is 8."""
    with Image.open(path) as image:
        return imagehash.phash(image, hash_size=PHASH_HASH_SIZE)


def build_manifest(root: str | Path, phash_threshold: int) -> pd.DataFrame:
    """Hash every jpg under train/test and assign transitive near-duplicate group ids."""
    threshold = int(phash_threshold)
    if threshold < 0:
        raise ValueError("phash_threshold must be >= 0")
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Ultrasound root does not exist: {root_path}")
    records: list[dict[str, object]] = []
    hashes: list[imagehash.ImageHash] = []
    for split in ("train", "test"):
        for label_name, label in LABELS:
            folder = root_path / split / label_name
            if not folder.is_dir():
                continue
            for image_path in sorted(folder.iterdir(), key=lambda item: item.name):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                digest = compute_phash(image_path)
                hashes.append(digest)
                records.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "split": split,
                        "label": label,
                        "phash": str(digest),
                    }
                )
    manifest = pd.DataFrame(records, columns=["image_path", "split", "label", "phash"])
    if manifest.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest["group_id"] = _group_ids(hashes, threshold)
    return manifest.loc[:, MANIFEST_COLUMNS]


def resolve_leakage(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep one image per group on the split that contains more of that group."""
    if manifest.empty:
        resolved = manifest.reindex(columns=MANIFEST_COLUMNS).copy()
        resolved.attrs["leaks_fixed"] = 0
        return resolved
    kept_rows: list[pd.Series] = []
    leaked = 0
    conflicts = 0
    for _, group in manifest.groupby("group_id", sort=True):
        if group["label"].nunique() > 1:
            conflicts += 1
        counts = group["split"].value_counts()
        selected = group
        if counts.size > 1:
            top = int(counts.max())
            leaders = [split for split, count in counts.items() if int(count) == top]
            ordered = _by_filename(group)
            winner = str(leaders[0] if len(leaders) == 1 else ordered.iloc[0]["split"])
            leaked += int((group["split"] != winner).sum())
            selected = group.loc[group["split"] == winner]
        kept_rows.append(_by_filename(selected).iloc[0])
    resolved = pd.DataFrame(kept_rows).loc[:, MANIFEST_COLUMNS]
    resolved = resolved.sort_values(["split", "image_path"], kind="mergesort").reset_index(drop=True)
    resolved.attrs["leaks_fixed"] = leaked
    LOGGER.info("Dropped %s ultrasound images from the minority side of cross-split groups", leaked)
    if conflicts:
        warnings.warn(
            f"{conflicts} near-duplicate groups contain both labels; the retained image keeps its own label",
            stacklevel=2,
        )
    return resolved


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def _group_ids(hashes: list[imagehash.ImageHash], threshold: int) -> list[int]:
    bits = np.stack([np.asarray(item.hash, dtype=bool).reshape(-1) for item in hashes])
    union = _UnionFind(len(hashes))
    for index in range(len(hashes)):
        distances = np.count_nonzero(bits[index + 1 :] != bits[index], axis=1)
        for offset in np.flatnonzero(distances <= threshold):
            union.union(index, index + 1 + int(offset))
    mapping: dict[int, int] = {}
    groups: list[int] = []
    for index in range(len(hashes)):
        root = union.find(index)
        if root not in mapping:
            mapping[root] = len(mapping)
        groups.append(mapping[root])
    return groups


def _by_filename(frame: pd.DataFrame) -> pd.DataFrame:
    filenames = frame["image_path"].map(lambda value: Path(str(value)).name)
    return frame.assign(_filename=filenames).sort_values(["_filename", "image_path"], kind="mergesort").drop(columns="_filename")
