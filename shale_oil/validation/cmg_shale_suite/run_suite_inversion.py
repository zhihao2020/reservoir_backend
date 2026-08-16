"""Offline S1–S5 shale inversion vs IMEX PRES (ruler only; no CMG at invert time).

Usage (repo root):
  python shale_oil/validation/cmg_shale_suite/run_suite_inversion.py
  python shale_oil/validation/cmg_shale_suite/run_suite_inversion.py --case S1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
CMG_IO = ROOT / "black_oil" / "validation"
if str(CMG_IO) not in sys.path:
    sys.path.insert(0, str(CMG_IO))

from build_shale_suite import CASE_DIR, CASES  # noqa: E402
from cmg_io.grid_parse import ft_to_m, parse_grid_series  # noqa: E402
from reservoir_backend.pipeline import (  # noqa: E402
    AxisAlignedBounds,
    BoundaryConditions,
    SensorSample,
    WellPoint,
    build_mesh,
    invert_rock,
    place_uniform_probes,
)
from reservoir_backend.pipeline.frac_param import decode_frac_theta  # noqa: E402

VAL = HERE.parent


def _mask(blocks, nx, ny, nz) -> np.ndarray:
    m = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in blocks:
        if 1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz:
            m[k - 1, j - 1, i - 1] = True
    return m


def _perf_step(n_frac_planes: int) -> int:
    """CMOST-style: denser stages → denser known completions."""
    return 2 if int(n_frac_planes) >= 8 else 3


def invert_case(case: str) -> dict:
    case = str(case).upper()
    case_dir = VAL / CASE_DIR[case]
    truth = json.loads((case_dir / f"truth_{case.lower()}.json").read_text(encoding="utf-8"))
    out_path = case_dir / f"mxshale_{case.lower()}.out"
    if not out_path.is_file():
        return {"case": case, "ok": False, "error": "missing IMEX .out"}

    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    di, dj = ft_to_m(g["di_ft"]), ft_to_m(g["dj_ft"])
    dk = np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float)
    bounds = AxisAlignedBounds(0.0, nx * di, 0.0, ny * dj, 0.0, float(np.sum(dk)))
    z_edges = np.concatenate([[0.0], np.cumsum(dk)])

    def center(i: int, j: int, k: int):
        return ((i - 0.5) * di, (j - 0.5) * dj, 0.5 * (z_edges[k - 1] + z_edges[k]))

    step = _perf_step(len(truth.get("frac_i_planes") or []))
    wells: list[WellPoint] = []
    open_from: dict[str, float] = {}
    for w in truth["wells"]:
        jw, kw = int(w["j"]), int(w["k"])
        tag = str(w["name"])
        open_from[tag] = float(w.get("open_from_day") or 0.0)
        for i in range(int(w["i0"]), int(w["i1"]) + 1, step):
            x, y, z = center(i, jw, kw)
            wells.append(WellPoint(f"{tag}_{i:02d}", x, y, z, role="producer"))
    base = build_mesh(bounds, np.full(nx, di), np.full(ny, dj), dk, wells=wells)
    specs = place_uniform_probes(base, n_p=6, n_s=0)
    wells.extend(WellPoint(**s.as_well_point_kwargs()) for s in specs)
    mesh = build_mesh(bounds, np.full(nx, di), np.full(ny, dj), dk, wells=wells)

    p_series = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    if len(p_series) < 2:
        return {"case": case, "ok": False, "error": "too few PRES times"}
    idx = np.unique(np.linspace(0, len(p_series) - 1, 5).astype(int))
    picks = [p_series[int(i)] for i in idx]
    n_prod = max(sum(1 for r in mesh.well_role.values() if r == "producer"), 1)
    q_each = -2.0e-6 / n_prod
    samples = []
    for t, p_psi in picks:
        p = np.asarray(p_psi, dtype=float) * 6894.757293168
        wp, wr = {}, {}
        for name, role in mesh.well_role.items():
            if name not in mesh.well_cell_id:
                continue
            i, j, k = mesh.grid.ijk(mesh.well_cell_id[name])
            val = float(p[k, j, i])
            prefix = name.rsplit("_", 1)[0]
            opened = float(t) >= float(open_from.get(prefix, 0.0))
            if role == "observer_p" and np.isfinite(val):
                wp[name] = val
            if role == "producer" and opened and np.isfinite(val):
                wp[name] = val
                wr[name] = q_each
        samples.append(
            SensorSample(
                time=float(t),
                well_pressure=wp,
                well_saturation={},
                boundary=BoundaryConditions(),
                well_rate=wr,
            )
        )

    k_prior = float(truth["matrix_perm_md"]["kx_geo"]) * 9.869233e-16
    auto = invert_rock(
        mesh,
        samples,
        permeability_prior_m2=k_prior,
        porosity_prior=0.08,
        viscosity_pa_s=2.0e-3,
        ne=8,
        n_assimilations=1,
        n_k_iterations=1,
        seed=5,
    )
    frac = _mask(truth["high_k_blocks_ijk"], nx, ny, nz)
    mat = ~frac
    r = float(np.mean(auto.k_mean[frac]) / max(float(np.mean(auto.k_mean[mat])), 1.0e-30))
    _t_last, p_last = picks[-1]
    p_pa = np.asarray(p_last, dtype=float) * 6894.757293168
    dp_true = float(np.nanmean(p_pa[mat]) - np.nanmean(p_pa[frac]))
    p_inv = auto.history[-1].pressure
    dp_inv = float(np.mean(p_inv[mat]) - np.mean(p_inv[frac]))
    eng = (
        decode_frac_theta(mesh, auto.theta_mean)
        if auto.theta_mean is not None
        else {}
    )
    js = np.where(frac.any(axis=(0, 2)))[0]
    truth_xf_m = 0.5 * (float(js.max() - js.min()) + 1) * float(dj) if js.size else float("nan")
    truth_fcd = float(truth["frac_perm_md"]) * 9.869233e-16 * float(di)
    return {
        "case": case,
        "ok": True,
        "analog": True,
        "n_times": len(samples),
        "n_wells": len(truth["wells"]),
        "frac_theta": any("frac θ" in n for n in auto.notes),
        "k_frac_over_matrix": r,
        "inv_x_f_m": eng.get("x_f_m"),
        "truth_x_f_m": truth_xf_m,
        "inv_F_cd_m3": eng.get("F_cd_m3"),
        "truth_F_cd_m3": truth_fcd,
        "inv_n_frac": eng.get("n_frac"),
        "truth_n_frac_planes": len(truth.get("frac_i_planes") or []),
        "imex_dp_matrix_minus_frac_Pa": dp_true,
        "inv_dp_matrix_minus_frac_Pa": dp_inv,
        "dp_sign_match": bool(dp_true * dp_inv > 0.0),
        "dp_ratio": float(dp_inv / dp_true) if abs(dp_true) > 1.0 else None,
        "notes": auto.notes[:12],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1–S5 inversion vs IMEX")
    parser.add_argument("--case", default="all", help="S1..S5 or all")
    args = parser.parse_args(argv)
    wanted = CASES if str(args.case).lower() == "all" else (str(args.case).upper(),)
    reports = []
    for case in wanted:
        print(f"=== {case} ===", flush=True)
        rec = invert_case(case)
        print(json.dumps(rec, indent=2), flush=True)
        reports.append(rec)
    dest = HERE / (
        "s1_inversion_report.json" if wanted == ("S1",) else "suite_inversion_report.json"
    )
    if wanted == ("S1",):
        dest.write_text(json.dumps(reports[0], indent=2), encoding="utf-8")
    else:
        dest.write_text(json.dumps({"cases": reports}, indent=2), encoding="utf-8")
        s1 = next((r for r in reports if r.get("case") == "S1"), None)
        if s1:
            (HERE / "s1_inversion_report.json").write_text(
                json.dumps(s1, indent=2), encoding="utf-8"
            )
    print(f"wrote {dest}", flush=True)
    return 0 if all(r.get("ok") for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
