"""Generate validation summary reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_validation_reports(summary: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write validation summary JSON and Markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "validation_summary.json"
    md_path = out / "validation_summary.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    md_path.write_text(_to_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Validation Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)
