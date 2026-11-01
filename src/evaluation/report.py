"""Write JSON and Markdown evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(payload: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_markdown(payload: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation report", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}**: {value}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
