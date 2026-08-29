"""Run CMG IMEX faulted-channel case and validate shape discovery.

Usage:
  python validation/black_oil/cmg_fault_3d/run_imex_and_validate.py --synthetic
  python validation/black_oil/cmg_fault_3d/run_imex_and_validate.py --execute
  python validation/black_oil/cmg_fault_3d/run_imex_and_validate.py --from-out mxspr006_fault.out
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    AxisAlignedBounds,
    BoundaryConditions,
    SensorSample,
    WellPoint,
    build_faulted_channel_twin,
    build_mesh,
    mask_overlap,
    run_shape_discovery,
)

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_fault.dat"
TRUTH = HERE / "truth_fault.json"
IMEX_EXE = Path(r"D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe")


def load_truth_mask() -> tuple[np.ndarray, dict]:
    meta = json.loads(TRUTH.read_text(encoding="utf-8"))
    nx, ny, nz = meta["grid"]["nx"], meta["grid"]["ny"], meta["grid"]["nz"]
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in meta["channel_blocks_ijk"]:
        mask[k - 1, j - 1, i - 1] = True
    return mask, meta


def ft_to_m(x: float) -> float:
    return float(x) * 0.3048


def psi_to_pa(p: float) -> float:
    return float(p) * 6894.757293168


def _well_k(meta: dict, well_key: str) -> int:
    w = meta["wells"][well_key]
    if "k_perfs" in w and w["k_perfs"]:
        ks = sorted(int(x) for x in w["k_perfs"])
        return ks[len(ks) // 2]
    return int(w["k"])


def mesh_from_truth(meta: dict):
    g = meta["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    di, dj = ft_to_m(g["di_ft"]), ft_to_m(g["dj_ft"])
    dk = np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float)
    lx, ly, lz = nx * di, ny * dj, float(np.sum(dk))
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    dx = np.full(nx, di, dtype=float)
    dy = np.full(ny, dj, dtype=float)
    z_edges = np.concatenate([[0.0], np.cumsum(dk)])
    wells_meta = meta["wells"]

    def cell_center(i: int, j: int, k: int) -> tuple[float, float, float]:
        return ((i - 0.5) * di, (j - 0.5) * dj, 0.5 * (z_edges[k - 1] + z_edges[k]))

    ik, pk = _well_k(meta, "INJ"), _well_k(meta, "PROD")
    ix, iy, iz = cell_center(wells_meta["INJ"]["i"], wells_meta["INJ"]["j"], ik)
    px, py, pz = cell_center(wells_meta["PROD"]["i"], wells_meta["PROD"]["j"], pk)
    wells = [WellPoint("INJ", ix, iy, iz), WellPoint("PROD", px, py, pz)]
    return build_mesh(bounds, dx, dy, dk, wells=wells)


def parse_sw_grids_from_out(out_path: Path, *, nx: int, ny: int, nz: int):
    text = out_path.read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, np.ndarray] = {}
    chunks = re.split(r"(?=Time\s*=\s*[0-9.]+)", text)
    for ch in chunks:
        mt = re.match(r"Time\s*=\s*([0-9.]+)", ch)
        if not mt:
            continue
        mtitle = re.search(r"(Oil|Gas|Water) Saturation \(fraction\)", ch[:800])
        if not mtitle or mtitle.group(1) != "Water":
            continue
        time = float(mt.group(1))
        sw = np.full((nz, ny, nx), np.nan)
        for kplane in re.finditer(r"Plane K\s*=\s*(\d+)(.*?)(?=Plane K\s*=|\Z)", ch, re.S):
            k = int(kplane.group(1))
            if not (1 <= k <= nz):
                continue
            body = kplane.group(2)
            if "All values are" in body[:200]:
                mval = re.search(r"All values are\s+([0-9.]+)", body)
                if mval:
                    sw[k - 1, :, :] = float(mval.group(1))
                continue
            for jline in re.finditer(r"J=\s*(\d+)\s+(.+)", body):
                j = int(jline.group(1))
                if not (1 <= j <= ny):
                    continue
                vals = re.findall(r"([0-9]+\.[0-9]+)", jline.group(2))
                for i, v in enumerate(vals[:nx]):
                    sw[k - 1, j - 1, i] = float(v)
        if np.isfinite(sw).sum() > 0:
            by_t[time] = sw
    return sorted(by_t.items())


def parse_well_bhp_from_out(out_path: Path) -> dict[float, tuple[float, float]]:
    text = out_path.read_text(encoding="latin-1", errors="ignore")
    by_t: dict[float, tuple[float, float]] = {}
    for m in re.finditer(
        r"Bottom Hole\s+psi\s+\+\s+([0-9.E+-]+)\s+\+\s+([0-9.E+-]+)", text
    ):
        start = max(0, m.start() - 800)
        window = text[start : m.start()]
        times = list(re.finditer(r"Time\s*=\s*([0-9.]+)", window))
        if not times:
            continue
        t = float(times[-1].group(1))
        by_t[t] = (float(m.group(1)), float(m.group(2)))
    return by_t


def samples_from_out(meta: dict, out_path: Path) -> list[SensorSample]:
    g = meta["grid"]
    sw_series = parse_sw_grids_from_out(
        out_path, nx=int(g["nx"]), ny=int(g["ny"]), nz=int(g["nz"])
    )
    bhp_by_t = parse_well_bhp_from_out(out_path)
    samples_from_out._last_sw_series = sw_series  # type: ignore[attr-defined]
    if len(sw_series) < 2:
        return []
    wi, wp = meta["wells"]["INJ"], meta["wells"]["PROD"]
    ik, pk = _well_k(meta, "INJ"), _well_k(meta, "PROD")
    bhp_times = sorted(bhp_by_t.keys())

    def nearest_bhp(t: float) -> tuple[float, float]:
        if not bhp_times:
            return 4500.0, 2000.0
        nearest = min(bhp_times, key=lambda x: abs(x - t))
        return bhp_by_t[nearest]

    samples: list[SensorSample] = []
    for t, sw in sw_series:
        sw_inj = float(sw[ik - 1, wi["j"] - 1, wi["i"] - 1])
        sw_prod = float(sw[pk - 1, wp["j"] - 1, wp["i"] - 1])
        if not np.isfinite(sw_inj):
            sw_inj = 0.8
        if not np.isfinite(sw_prod):
            sw_prod = 0.2
        bhp_inj, bhp_prod = nearest_bhp(t)
        p_inj, p_prod = psi_to_pa(bhp_inj), psi_to_pa(bhp_prod)
        samples.append(
            SensorSample(
                time=float(t),
                well_pressure={"INJ": p_inj, "PROD": p_prod},
                well_saturation={
                    "INJ": (sw_inj, max(0.0, 1.0 - sw_inj), 0.0),
                    "PROD": (sw_prod, max(0.0, 1.0 - sw_prod), 0.0),
                },
                boundary=BoundaryConditions(),
            )
        )
    return samples


def run_imex_direct() -> dict:
    if not IMEX_EXE.is_file():
        return {"ok": False, "message": f"IMEX exe not found: {IMEX_EXE}"}
    try:
        proc = subprocess.run(
            [str(IMEX_EXE), "-f", DAT.name],
            cwd=str(HERE),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-1500:],
            "out_exists": (HERE / "mxspr006_fault.out").is_file(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}


def validate_on_cmg_geometry() -> dict:
    truth_mask, meta = load_truth_mask()
    mesh = mesh_from_truth(meta)
    out_path = HERE / "mxspr006_fault.out"
    if not out_path.is_file():
        return {"mode": "cmg_fault_geometry", "ok": False, "message": "missing .out"}
    samples = samples_from_out(meta, out_path)
    if len(samples) < 2:
        return {"mode": "cmg_fault_geometry", "ok": False, "message": "failed to parse samples"}

    result = run_shape_discovery(
        mesh,
        samples,
        permeability_prior_m2=4.0e-14,
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
    )
    metrics = mask_overlap(result.active_mask, truth_mask)

    footprint_metrics = None
    sw_series = getattr(samples_from_out, "_last_sw_series", None)
    if sw_series and len(sw_series) >= 2:
        dsw = np.nan_to_num(np.abs(sw_series[-1][1] - sw_series[0][1]), nan=0.0)
        thr = float(np.quantile(dsw, 0.75))
        footprint = dsw >= max(thr, 1.0e-3)
        footprint_metrics = mask_overlap(footprint, truth_mask)

    # fault plane enrichment: high |grad p| or low flow across seal j=1:6
    fault_meta = meta.get("fault") or {}

    report = {
        "mode": "cmg_fault_geometry",
        "ok": True,
        "n_samples": len(samples),
        "grid": meta["grid"],
        "fault": fault_meta,
        "sample_times": [s.time for s in samples],
        "well_sw_last": {
            "INJ": samples[-1].well_saturation["INJ"][0],
            "PROD": samples[-1].well_saturation["PROD"][0],
        },
        "indicator_stats": result.indicator_stats,
        "refine_stats": result.refine_stats,
        "overlap_discovery_vs_channel": metrics,
        "overlap_cmg_dsw_footprint_vs_channel": footprint_metrics,
        "notes": result.notes,
    }
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(HERE / "discovered_active_mask.npy", result.active_mask.astype(np.uint8))
    np.save(HERE / "truth_active_mask.npy", truth_mask.astype(np.uint8))
    return report


def validate_synthetic() -> dict:
    twin = build_faulted_channel_twin(nx=12, ny=10, nz=4, n_times=4)
    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=1.0e-13,
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
    )
    metrics = mask_overlap(result.active_mask, twin.true_channel_mask)
    # discovery should not light up the fault barrier as "channel"
    fault_false_pos = None
    if twin.true_fault_mask is not None and np.any(twin.true_fault_mask):
        fault_false_pos = float(
            np.mean(result.active_mask[twin.true_fault_mask].astype(float))
        )
    report = {
        "mode": "synthetic_faulted_channel_twin",
        "indicator_stats": result.indicator_stats,
        "refine_stats": result.refine_stats,
        "overlap": metrics,
        "fault_active_fraction": fault_false_pos,
        "notes": result.notes,
        "fine_cells": int(result.refine_stats.get("fine_cells", 0)),
    }
    (HERE / "synthetic_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CMG faulted channel validation")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--from-out", type=str, default=None)
    args = parser.parse_args(argv)

    reports = []
    if args.synthetic or (not args.execute and args.from_out is None):
        reports.append(validate_synthetic())
    if args.execute:
        imex = run_imex_direct()
        (HERE / "imex_run_status.json").write_text(json.dumps(imex, indent=2), encoding="utf-8")
        reports.append({"imex_run": imex})
    if args.from_out or args.execute or not args.synthetic:
        reports.append(validate_on_cmg_geometry())

    print(json.dumps(reports, indent=2))
    for r in reports:
        if r.get("mode") == "synthetic_faulted_channel_twin":
            if r["overlap"]["dice"] < 0.08:
                return 2
            # barrier cells should not be mostly classified active
            faf = r.get("fault_active_fraction")
            if faf is not None and faf > 0.85:
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
