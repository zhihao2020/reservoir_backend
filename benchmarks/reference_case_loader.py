"""Load extracted open-source-adapted reference fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_JSON = ROOT / "references" / "fixtures" / "open_source_adapted_cases.json"
FIXTURE_NPZ = ROOT / "references" / "fixtures" / "open_source_adapted_arrays.npz"


def load_open_source_reference_cases() -> dict:
    """Load extracted reference metadata without touching upstream files."""
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def get_reference_case(case_name: str) -> dict:
    """Return one extracted reference case by name."""
    summary = load_open_source_reference_cases()
    for case in summary["cases"]:
        if case["case_name"] == case_name:
            return case
    raise KeyError(f"reference case {case_name!r} not found")


def load_reference_arrays() -> dict[str, np.ndarray]:
    """Load extracted reference arrays from NPZ fixtures."""
    data = np.load(FIXTURE_NPZ)
    return {name: data[name] for name in data.files}
