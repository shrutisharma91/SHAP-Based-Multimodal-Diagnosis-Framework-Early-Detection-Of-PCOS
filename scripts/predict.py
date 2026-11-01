"""CLI prediction: --clinical patient.json and/or --image scan.jpg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a clinical JSON and/or an ultrasound image.")
    parser.add_argument("--clinical", type=Path, help="Patient feature JSON")
    parser.add_argument("--image", type=Path, help="Ultrasound image path")
    args = parser.parse_args()
    if args.clinical is None and args.image is None:
        parser.error("provide --clinical and/or --image")
    result = {}
    if args.clinical is not None:
        result["clinical"] = json.loads(args.clinical.read_text(encoding="utf-8"))
    if args.image is not None:
        result["image"] = str(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
