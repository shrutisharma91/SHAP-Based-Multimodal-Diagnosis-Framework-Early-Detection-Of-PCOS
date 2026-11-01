"""Write a leakage-free ultrasound manifest. Does not train a model."""

from __future__ import annotations

from pathlib import Path

from src.data.ultrasound_dedupe import build_manifest, resolve_leakage
from src.utils import load_config


def main() -> None:
    config = load_config()
    root = _resolve(config["paths"]["ultrasound_root"])
    manifest = build_manifest(root, int(config["dedupe"]["phash_threshold"]))
    resolved = resolve_leakage(manifest)
    destination = Path.cwd() / "data" / "interim" / "ultrasound_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(destination, index=False)
    raw_count = len(manifest)
    after_exact = int(manifest["phash"].nunique()) if raw_count else 0
    print(f"raw count: {raw_count}")
    print(f"after exact dedupe: {after_exact}")
    print(f"after near-dup dedupe: {len(resolved)}")
    print(f"leaks fixed: {int(resolved.attrs.get('leaks_fixed', 0))}")
    print(f"wrote {destination}")


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


if __name__ == "__main__":
    main()
