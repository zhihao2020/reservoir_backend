#!/usr/bin/env python3
"""Verify the example package (offline path).

Loads ``example/case.yaml``, runs ``run_pipeline``, and checks that the
expected outputs exist and are sane:

    python scripts/verify_examples.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.lab_case import load_lab_case
from src.programs.pipeline import run_pipeline, write_output

CASE = REPO / "example" / "case.yaml"


def _check_case() -> tuple[bool, str]:
    case = load_lab_case(CASE)
    n_cells = case.nx * case.ny * case.nz
    fields = run_pipeline(case)
    ok = (
        fields.p.shape == (case.times.size, n_cells)
        and np.isfinite(fields.p).any()
        and np.isfinite(fields.k).any()
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output(Path(tmp), case, fields)
        files = sorted(p.name for p in Path(tmp).iterdir())
    want = {"fields.npz", "fields_field.npz", "summary.json", "validation.json", "results.json"}
    rel = "example/case.yaml"
    return ok and want <= set(files), (
        f"{rel}: {n_cells} cells, {len(case.probes)} probes, {len(case.wells)} wells, "
        f"{case.times.size} times -> p{fields.p.shape} k/phi finite, wrote {len(files)} files"
    )


def main() -> int:
    try:
        ok, msg = _check_case()
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, f"example/case.yaml: ERROR {exc!r}"
    print(("  OK  " if ok else " FAIL ") + msg)
    print(f"\n{'all examples verified' if ok else '1 case(s) failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
