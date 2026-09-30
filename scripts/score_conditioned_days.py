#!/usr/bin/env python3
"""Score probe-conditioned saturations at GEM day 30 and day 360.

Day 30 runs the compositional transport. Day 360 reuses that transport prior
(the year run showed it stalls and does not keep evolving) and applies the
day-360 probe saturations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gap_analysis_variants as gap
from src.core.lab_case import load_lab_case
from src.programs import pipeline
from src.programs.pipeline import _condition_saturations, run_mesh, run_pipeline

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
SR3 = ROOT / "results" / "shailoil_variants" / VARIANT / f"shailoil_{VARIANT}.sr3"


def _enable_forward(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("model: none", "model: compositional", 1)
    if "freeze_pressure:" not in text:
        text = text.replace(
            "model: compositional\n",
            "model: compositional\n  freeze_pressure: true\n",
            1,
        )
    path.write_text(text, encoding="utf-8")


def _report(label: str, snap: dict, sg: np.ndarray, probe_cells: np.ndarray, probe_sg: np.ndarray) -> None:
    layers = sg.reshape(gap.NZ, gap.NY, gap.NX)
    finite = np.isfinite(probe_sg)
    probe_err = float(np.sqrt(np.mean((sg[probe_cells][finite] - probe_sg[finite]) ** 2)))
    print(
        f"{label}: sg_rmse {gap._rmse(sg, snap['sg']):.4f} "
        f"bottom {layers[:3].mean():.3f} (gem {snap['sg'].reshape(gap.NZ, gap.NY, gap.NX)[:3].mean():.3f}) "
        f"top {layers[-3:].mean():.3f} (gem {snap['sg'].reshape(gap.NZ, gap.NY, gap.NX)[-3:].mean():.3f}) "
        f"n_sg>0.1 {(sg > 0.1).sum()} (gem {(snap['sg'] > 0.1).sum()}) "
        f"probe_sg_rmse {probe_err:.4f}",
        flush=True,
    )


def main() -> None:
    gap.TARGETS = (30.0, 360.0)
    snaps, wells, series, layers = gap._snaps(SR3)
    by_day = {snap["day"]: snap for snap in snaps}
    print(f"variant {VARIANT} {SR3.name}", flush=True)
    dest = gap.OUT / f"conditioned_day30_{VARIANT}"
    gap.write_case(dest, [by_day[30.0]], wells, [series[0]], layers)
    _enable_forward(dest / "case.yaml")
    saved: dict = {}
    real = pipeline._condition_saturations

    def _capture(case, mesh, sw, so, sg):
        saved["sg"] = np.array(sg, copy=True)
        saved["sw"] = np.array(sw, copy=True)
        saved["so"] = np.array(so, copy=True)
        saved["mesh"] = mesh
        return real(case, mesh, sw, so, sg)

    pipeline._condition_saturations = _capture
    case = load_lab_case(dest / "case.yaml")
    print("running day-30 transport + probe correction", flush=True)
    fields = run_pipeline(case)
    pipeline._condition_saturations = real
    mesh = run_mesh(case)
    cells = np.asarray(mesh.probes.cell, dtype=np.int64)
    _report("day 30", by_day[30.0], fields.sg[0], cells, case.sg[0])
    hold = fields.diagnostics
    print(
        f"holdout sg rmse conditioned {hold.get('sg_holdout_rmse')} "
        f"prior {hold.get('sg_prior_holdout_rmse')}",
        flush=True,
    )

    # Day 360 probe values, same stalled transport prior.
    snap = by_day[360.0]
    probe_sg = snap["sg"][cells]
    probe_so = snap["so"][cells]
    probe_sw = snap["sw"][cells]

    class _Day360:
        sw = probe_sw[None, :]
        so = probe_so[None, :]
        sg = probe_sg[None, :]

    csw, cso, csg = _condition_saturations(
        _Day360(), mesh, saved["sw"].copy(), saved["so"].copy(), saved["sg"].copy(),
    )
    _report("day 360", snap, csg[0], cells, probe_sg)


if __name__ == "__main__":
    main()
