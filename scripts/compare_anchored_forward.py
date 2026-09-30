#!/usr/bin/env python3
"""Compare pressure-anchored compositional transport with one GEM year.

Truth is the base 15x15x15 SR3 at 30, 60, ..., 360 days (a 360-day year,
one report a month). Pressure is kriged from the 27 bottom probes. Saturations
come from the frozen-pressure compositional forward. If a month is missing
SG, PRES, or Z(5), the base deck is rerun for those twelve report times.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import gap_analysis_variants as gap
from src.core.lab_case import load_lab_case
from src.programs.pipeline import run_pipeline

SR3 = ROOT / "results" / "shailoil_variants" / "base" / "shailoil_base.sr3"
DECK = ROOT / "examples" / "shailoil_variants" / "shailoil_base.dat"
OUT = ROOT / "results" / "gap_analysis"
MONTHS = tuple(float(day) for day in range(30, 361, 30))
NEED = ("SG", "PRES", "Z(5)")
GEM_EXE = Path(r"D:\Tool\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe")


def _months_present(sr3: Path) -> list[float]:
    import h5py

    missing = []
    with h5py.File(sr3, "r") as f:
        tt = f["General/MasterTimeTable"][:]
        tmap = {int(r["Index"]): float(r["Offset in days"]) for r in tt}
        sp = f["SpatialProperties"]
        found = []
        for key in sp:
            if not key.isdigit() or "SG" not in sp[key]:
                continue
            found.append((tmap[int(key)], int(key)))
        for day in MONTHS:
            near = min(found, key=lambda row: abs(row[0] - day))
            if abs(near[0] - day) > 1.0e-3:
                missing.append(day)
                continue
            group = sp[f"{near[1]:06d}"]
            if any(name not in group for name in NEED):
                missing.append(day)
    if missing:
        raise RuntimeError(f"missing monthly grid fields at days {missing}")
    return list(MONTHS)


def _yearly_deck(src: Path, dest: Path) -> None:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("*TIME"))
    # Keep the early *DTMAX switch, then report only the twelve month ends.
    body = ["*DTMAX 1.0"]
    body.extend(f"*TIME {int(day)}" for day in MONTHS)
    body.append("*STOP")
    dest.write_text("\n".join(lines[:start] + body) + "\n", encoding="utf-8")


def _regenerate(sr3: Path) -> None:
    import subprocess

    if not GEM_EXE.is_file():
        raise RuntimeError(f"monthly fields missing and GEM executable not found: {GEM_EXE}")
    work = OUT / "gem_year"
    work.mkdir(parents=True, exist_ok=True)
    deck = work / "shailoil_base.dat"
    _yearly_deck(DECK, deck)
    for stale in work.glob("shailoil_base.*"):
        if stale.suffix != ".dat":
            stale.unlink()
    log_path = work / "run.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            [str(GEM_EXE), "-f", deck.name],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"GEM year rerun failed with code {proc.returncode}; see {log_path}")
    produced = work / "shailoil_base.sr3"
    if not produced.is_file():
        raise RuntimeError(f"GEM year rerun did not write {produced}")
    sr3.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(sr3)


def _case_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("model: none", "model: compositional", 1)
    if "freeze_pressure:" not in text:
        text = text.replace("model: compositional\n", "model: compositional\n  freeze_pressure: true\n", 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        _months_present(SR3)
        sr3 = SR3
        regenerated = False
    except RuntimeError as exc:
        print(exc, flush=True)
        print("regenerating one GEM year", flush=True)
        _regenerate(SR3)
        _months_present(SR3)
        sr3 = SR3
        regenerated = True
    gap.TARGETS = MONTHS
    snaps, wells, series, layers = gap._snaps(sr3)
    dest = OUT / "anchored_year_case"
    gap.write_case(dest, snaps, wells, series, layers)
    _case_text(dest / "case.yaml")
    case = load_lab_case(dest / "case.yaml")
    if case.forward_model != "compositional" or not case.freeze_pressure or not case.k_homogeneous:
        raise RuntimeError(
            f"case flags forward={case.forward_model} freeze={case.freeze_pressure} "
            f"k_homogeneous={case.k_homogeneous}"
        )
    print(f"forward {len(MONTHS)} months on {case.nx}x{case.ny}x{case.nz}", flush=True)
    t0 = time.perf_counter()
    fields = run_pipeline(case)
    elapsed = time.perf_counter() - t0
    rows = []
    for i, snap in enumerate(snaps):
        co2 = None if fields.co2 is None else fields.co2[i]
        row = gap.plume_row(
            snap["sg"], fields.sg[i], snap["p"], fields.p[i], snap["so"], fields.so[i], snap["z"], co2,
        )
        row["day"] = snap["day"]
        rows.append(row)
        print(
            f"day {snap['day']:.0f} sg_rmse {row['sg_rmse']:.4f} "
            f"n_sg {row['n_sg_gt_0.1_truth']}->{row['n_sg_gt_0.1_recon']} "
            f"zco2_corr {row.get('co2_vs_zco2_corr')}",
            flush=True,
        )
    report = {
        "grid": [case.nx, case.ny, case.nz],
        "days": list(MONTHS),
        "regenerated": regenerated,
        "elapsed_s": round(elapsed, 3),
        "forward_model": case.forward_model,
        "freeze_pressure": case.freeze_pressure,
        "co2_from": fields.diagnostics.get("co2_from"),
        "overall": {
            "sg_rmse": gap._agg(rows, "sg_rmse"),
            "sg_corr": gap._agg(rows, "sg_corr"),
            "n_sg_gt_0.1_truth": gap._agg(rows, "n_sg_gt_0.1_truth"),
            "n_sg_gt_0.1_recon": gap._agg(rows, "n_sg_gt_0.1_recon"),
            "zco2_mean_truth": gap._agg(rows, "zco2_mean_truth"),
            "co2_mean_recon": gap._agg(rows, "co2_mean_recon"),
            "co2_vs_zco2_corr": gap._agg(rows, "co2_vs_zco2_corr"),
            "sg_top_truth": gap._agg(rows, "sg_top_truth"),
            "sg_top_recon": gap._agg(rows, "sg_top_recon"),
            "sg_bottom_truth": gap._agg(rows, "sg_bottom_truth"),
            "sg_bottom_recon": gap._agg(rows, "sg_bottom_recon"),
        },
        "per_time": rows,
    }
    dest_json = OUT / "anchored_year.json"
    dest_json.write_text(json.dumps(gap._json_ready(report), indent=2), encoding="utf-8")
    print("wrote", dest_json, "elapsed_s", report["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
