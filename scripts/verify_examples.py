#!/usr/bin/env python3
"""Verify the example library end-to-end (offline path).

Runs every fast case through ``run_pipeline`` and checks that the expected
outputs exist and are sane; the 60^3 ``online`` case is parsed only (running it
is slow). One command confirms the examples are complete and ready for联调:

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

EXAMPLES = REPO / "examples"

# (relative case.yaml, run the pipeline?)
CASES = [
    ("small/case.yaml", True),
    ("twod/case.yaml", True),
    ("model_compare/total_mobility.yaml", True),
    ("model_compare/three_phase.yaml", True),
    ("offline/case.yaml", True),
    ("online/case.yaml", False),  # 60^3 — parse only
]


def _check_case(rel: str, run: bool) -> tuple[bool, str]:
    case = load_lab_case(EXAMPLES / rel)
    n_cells = case.nx * case.ny * case.nz
    if not run:
        return True, f"loaded {rel}: {n_cells} cells, {len(case.probes)} probes, {len(case.wells)} wells, {case.times.size} times"
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
    return ok and want <= set(files), (
        f"{rel}: {n_cells} cells, {case.times.size} times -> p{fields.p.shape} "
        f"k/phi finite, wrote {len(files)} files"
    )


def main() -> int:
    failures = 0
    for rel, run in CASES:
        try:
            ok, msg = _check_case(rel, run)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{rel}: ERROR {exc!r}"
        print(("  OK  " if ok else " FAIL ") + msg)
        if not ok:
            failures += 1
    print(f"\n{'all examples verified' if failures == 0 else f'{failures} case(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
