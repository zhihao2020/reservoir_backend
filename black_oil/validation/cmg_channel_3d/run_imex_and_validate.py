"""Run CMG IMEX channel case (optional) and validate shape discovery.

Usage:
  # algorithm-only twin (no CMG license needed):
  python validation/cmg_channel_3d/run_imex_and_validate.py --synthetic

  # after IMEX produces mxspr006_channel.out (or with --execute):
  python validation/cmg_channel_3d/run_imex_and_validate.py --from-out mxspr006_channel.out

  # attempt controlled IMEX run then validate:
  python validation/cmg_channel_3d/run_imex_and_validate.py --execute
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
    build_channel_twin,
    build_mesh,
    mask_overlap,
    run_shape_discovery,
)

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_channel.dat"
TRUTH = HERE / "truth_channel.json"
CMG_HOME = Path(r"D:\Tool\CMG")
IMEX_EXE = CMG_HOME / "IMEX" / "2024.20" / "Win_x64" / "EXE" / "mx202420.exe"
SCRIPTS = Path(r"C:\Users\xuzhihao\.grok\skills\cmg-suite\scripts")


def load_truth_mask() -> tuple[np.ndarray, dict]:
    meta = json.loads(TRUTH.read_text(encoding="utf-8"))
    nx, ny, nz = meta["grid"]["nx"], meta["grid"]["ny"], meta["grid"]["nz"]
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in meta["channel_blocks_ijk"]:
        # CMG IJK 1-based → 0-based; field layout (k,j,i)
        mask[k - 1, j - 1, i - 1] = True
    return mask, meta


def ft_to_m(x: float) -> float:
    return float(x) * 0.3048


def psi_to_pa(p: float) -> float:
    return float(p) * 6894.757293168


def _well_k(meta: dict, well_key: str) -> int:
    """Representative completion layer (middle of k_perfs, or legacy single k)."""
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
    if dk.size != nz:
        dk = np.full(nz, float(np.sum(dk) / nz))
    lx, ly, lz = nx * di, ny * dj, float(np.sum(dk))
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    # Orthogonal discovery mesh: same IJK counts as CMG VARI (DTOP undulation
    # is carried in truth mask / structure metadata, not in TPFA z-nodes yet).
    dx = np.full(nx, di, dtype=float)
    dy = np.full(ny, dj, dtype=float)
    wells_meta = meta["wells"]
    z_edges = np.concatenate([[0.0], np.cumsum(dk)])

    def cell_center(i: int, j: int, k: int) -> tuple[float, float, float]:
        return (
            (i - 0.5) * di,
            (j - 0.5) * dj,
            0.5 * (z_edges[k - 1] + z_edges[k]),
        )

    ik, pk = _well_k(meta, "INJ"), _well_k(meta, "PROD")
    ix, iy, iz = cell_center(wells_meta["INJ"]["i"], wells_meta["INJ"]["j"], ik)
    px, py, pz = cell_center(wells_meta["PROD"]["i"], wells_meta["PROD"]["j"], pk)
    wells = [WellPoint("INJ", ix, iy, iz), WellPoint("PROD", px, py, pz)]
    return build_mesh(bounds, dx, dy, dk, wells=wells)


def parse_sw_grids_from_out(
    out_path: Path, *, nx: int = 7, ny: int = 7, nz: int = 3
) -> list[tuple[float, np.ndarray]]:
    """Parse multi-time water-saturation grids from IMEX .out text."""
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
    """Parse injector/producer BHP (psi) keyed by report time (days)."""
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


def samples_from_out_or_proxy(meta: dict, out_path: Path | None) -> list[SensorSample]:
    """Build multi-time SensorSample list from CMG .out or synthetic proxy."""
    if out_path is not None and out_path.is_file():
        g = meta["grid"]
        sw_series = parse_sw_grids_from_out(
            out_path, nx=int(g["nx"]), ny=int(g["ny"]), nz=int(g["nz"])
        )
        bhp_by_t = parse_well_bhp_from_out(out_path)
        if len(sw_series) >= 2:
            samples: list[SensorSample] = []
            wi = meta["wells"]["INJ"]
            wp = meta["wells"]["PROD"]
            ik, pk = _well_k(meta, "INJ"), _well_k(meta, "PROD")
            bhp_times = sorted(bhp_by_t.keys())

            def nearest_bhp(t: float) -> tuple[float, float]:
                if not bhp_times:
                    return 4500.0, 2000.0
                nearest = min(bhp_times, key=lambda x: abs(x - t))
                return bhp_by_t[nearest]

            for t, sw in sw_series:
                # CMG IJK 1-based → (k-1, j-1, i-1); use mid completion layer
                sw_inj = float(sw[ik - 1, wi["j"] - 1, wi["i"] - 1])
                sw_prod = float(sw[pk - 1, wp["j"] - 1, wp["i"] - 1])
                if not np.isfinite(sw_inj):
                    sw_inj = 0.8
                if not np.isfinite(sw_prod):
                    sw_prod = 0.2
                bhp_inj, bhp_prod = nearest_bhp(t)
                p_inj = psi_to_pa(bhp_inj)
                p_prod = psi_to_pa(bhp_prod)
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
            samples_from_out_or_proxy._last_sw_series = sw_series  # type: ignore[attr-defined]
            return samples

    # Proxy multi-time well data consistent with channel waterflood trend
    times = [0.0, 180.0, 365.0, 545.0, 730.0]
    samples = []
    for i, t in enumerate(times):
        p_inj = psi_to_pa(4500.0 + 100.0 * i)
        p_prod = psi_to_pa(2000.0 - 50.0 * i)
        sw_prod = 0.22 + 0.08 * i
        samples.append(
            SensorSample(
                time=t,
                well_pressure={"INJ": p_inj, "PROD": p_prod},
                well_saturation={
                    "INJ": (0.85, 0.15, 0.0),
                    "PROD": (sw_prod, 1.0 - sw_prod, 0.0),
                },
                boundary=BoundaryConditions(),
            )
        )
    return samples


def run_imex_direct() -> dict:
    """Launch IMEX executable on the local channel case."""
    if not IMEX_EXE.is_file():
        return {"ok": False, "message": f"IMEX exe not found: {IMEX_EXE}"}
    if not DAT.is_file():
        return {"ok": False, "message": f"DAT not found: {DAT}"}
    # IMEX typically run with dat as cwd
    cmd = [str(IMEX_EXE), "-f", DAT.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "out_exists": (HERE / "mxspr006_channel.out").is_file(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}


def validate_on_cmg_geometry() -> dict:
    truth_mask, meta = load_truth_mask()
    mesh = mesh_from_truth(meta)
    out_path = HERE / "mxspr006_channel.out"
    samples = samples_from_out_or_proxy(meta, out_path if out_path.is_file() else None)
    result = run_shape_discovery(
        mesh,
        samples,
        permeability_prior_m2=5.0e-14,  # ~50 md order of magnitude prior
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
    )
    # map truth to mesh shape (same nx ny nz)
    if truth_mask.shape != result.active_mask.shape:
        tmask = np.zeros(mesh.grid.shape, dtype=bool)
        for i, j, k in meta["channel_blocks_ijk"]:
            if i <= mesh.grid.nx and j <= mesh.grid.ny and k <= mesh.grid.nz:
                tmask[k - 1, j - 1, i - 1] = True
        truth_mask = tmask
    metrics = mask_overlap(result.active_mask, truth_mask)

    # CMG water footprint enrichment: cells with large ΔSw should align with channel
    footprint_metrics = None
    sw_series = getattr(samples_from_out_or_proxy, "_last_sw_series", None)
    if sw_series and len(sw_series) >= 2:
        dsw = np.abs(sw_series[-1][1] - sw_series[0][1])
        dsw = np.nan_to_num(dsw, nan=0.0)
        thr = float(np.quantile(dsw, 0.75))
        footprint = dsw >= max(thr, 1.0e-3)
        footprint_metrics = mask_overlap(footprint, truth_mask)

    structure = meta.get("structure") or {}
    report = {
        "mode": "cmg_geometry_undulating",
        "n_samples": len(samples),
        "used_out": out_path.is_file(),
        "grid": meta["grid"],
        "structure_relief_ft": structure.get("relief_ft"),
        "structure_type": structure.get("type"),
        "sample_times": [s.time for s in samples],
        "well_sw_last": {
            "INJ": samples[-1].well_saturation["INJ"][0] if samples else None,
            "PROD": samples[-1].well_saturation["PROD"][0] if samples else None,
        },
        "indicator_stats": result.indicator_stats,
        "refine_stats": result.refine_stats,
        "overlap_discovery_vs_channel": metrics,
        "overlap_cmg_dsw_footprint_vs_channel": footprint_metrics,
        "notes": result.notes,
    }
    (HERE / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.save(HERE / "discovered_active_mask.npy", result.active_mask.astype(np.uint8))
    np.save(HERE / "truth_active_mask.npy", truth_mask.astype(np.uint8))
    return report


def validate_synthetic() -> dict:
    twin = build_channel_twin(nx=10, ny=8, nz=4, n_times=4)
    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=1.0e-13,
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
    )
    metrics = mask_overlap(result.active_mask, twin.true_channel_mask)
    report = {
        "mode": "synthetic_channel_twin",
        "indicator_stats": result.indicator_stats,
        "refine_stats": result.refine_stats,
        "overlap": metrics,
        "notes": result.notes,
        "fine_cells": int(result.refine_stats.get("fine_cells", 0)),
    }
    (HERE / "synthetic_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CMG channel twin validation")
    parser.add_argument("--synthetic", action="store_true", help="run in-process synthetic twin")
    parser.add_argument("--execute", action="store_true", help="run IMEX on local DAT")
    parser.add_argument("--from-out", type=str, default=None, help="path to IMEX .out")
    args = parser.parse_args(argv)

    reports = []
    if args.synthetic or (not args.execute and args.from_out is None):
        reports.append(validate_synthetic())

    if args.execute:
        imex = run_imex_direct()
        (HERE / "imex_run_status.json").write_text(json.dumps(imex, indent=2), encoding="utf-8")
        reports.append({"imex_run": imex})

    if args.from_out or args.execute or not args.synthetic:
        # always try CMG geometry path (proxy samples if no .out)
        reports.append(validate_on_cmg_geometry())

    print(json.dumps(reports, indent=2))
    # soft pass: synthetic dice > 0.12
    for r in reports:
        if r.get("mode") == "synthetic_channel_twin":
            if r["overlap"]["dice"] < 0.12:
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
