"""CMG virtual-probe accuracy study: sweep N and layout strategy.

Samples virtual exclusive probes from existing IMEX .out full-grid PRES/SW
(does **not** add wells to the CMG deck or re-run IMEX).

Usage (repo root):
  python black_oil/validation/cmg_probe_study/run_probe_study.py
  python black_oil/validation/cmg_probe_study/run_probe_study.py --cases channel --n-list 0,4,8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    build_mesh,
    place_uniform_probes,
    recommend_probes,
    run_time_series,
    split_n_probes,
)
from reservoir_backend.pipeline.probe_design import field_variance_over_time  # noqa: E402
VAL = Path(__file__).resolve().parents[1]
if str(VAL) not in sys.path:
    sys.path.insert(0, str(VAL))
from cmg_io.grid_parse import (  # noqa: E402
    ft_to_m,
    parse_bhp,
    parse_grid_series,
    parse_surface_rates_m3s,
    psi_to_pa,
)

HERE = Path(__file__).resolve().parent
CHANNEL = VAL / "cmg_channel_3d"
FAULT = VAL / "cmg_fault_3d"
CHANNEL_FINE = VAL / "cmg_channel_fine"
FIVESPOT = VAL / "cmg_fivespot"


def _mid_k(w: dict) -> int:
    if "k_perfs" in w and w["k_perfs"]:
        ks = sorted(int(x) for x in w["k_perfs"])
        return ks[len(ks) // 2]
    return int(w.get("k", 1))


def mesh_with_probes(meta: dict, probe_well_points: list[WellPoint]):
    g = meta["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    di, dj = ft_to_m(g["di_ft"]), ft_to_m(g["dj_ft"])
    dk = np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float)
    lx, ly, lz = nx * di, ny * dj, float(np.sum(dk))
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    dx = np.full(nx, di)
    dy = np.full(ny, dj)
    z_edges = np.concatenate([[0.0], np.cumsum(dk)])

    def center(i: int, j: int, k: int):
        return (
            (i - 0.5) * di,
            (j - 0.5) * dj,
            0.5 * (z_edges[k - 1] + z_edges[k]),
        )

    wells: list[WellPoint] = []
    first_inj_k = 1
    first_prod_k = 1
    for name, spec in meta["wells"].items():
        role = str(spec.get("role") or "").lower()
        if not role:
            role = "injector" if "inj" in name.lower() else "producer"
        kk = _mid_k(spec)
        x, y, z = center(int(spec["i"]), int(spec["j"]), kk)
        wells.append(WellPoint(name, x, y, z, role=role))
        if role == "injector" and first_inj_k == 1:
            first_inj_k = kk
        if role == "producer" and first_prod_k == 1:
            first_prod_k = kk
    wells.extend(probe_well_points)
    return build_mesh(bounds, dx, dy, dk, wells=wells), first_inj_k, first_prod_k


def truth_mask(meta: dict) -> np.ndarray:
    g = meta["grid"]
    mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)
    key = "channel_blocks_ijk"
    if key not in meta and "fault_blocks_ijk" in meta:
        # fault case may only have channel list under channel_blocks_ijk still
        pass
    blocks = meta.get("channel_blocks_ijk") or meta.get("high_k_blocks_ijk") or []
    for i, j, k in blocks:
        mask[k - 1, j - 1, i - 1] = True
    return mask


def rel_l2(a, b) -> float:
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=float), nan=0.0)
    den = float(np.linalg.norm(b.ravel())) + 1.0e-30
    return float(np.linalg.norm((a - b).ravel()) / den)


def dice_masks(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.sum(a & b))
    s = float(np.sum(a) + np.sum(b))
    return 0.0 if s <= 0 else 2.0 * inter / s


def build_samples(
    meta: dict,
    mesh,
    out_path: Path,
    ik: int,
    pk: int,
    sw_series,
    p_series_pa,
) -> list[SensorSample]:
    bhp = parse_bhp(out_path)
    bhp_times = sorted(bhp.keys())
    rates = parse_surface_rates_m3s(out_path)
    rate_times = sorted(rates.keys())
    wi, wp = meta["wells"]["INJ"], meta["wells"]["PROD"]

    # map time → fields
    sw_by_t = {float(t): sw for t, sw in sw_series}
    p_by_t = {float(t): p for t, p in p_series_pa}
    times = sorted(set(sw_by_t) | set(p_by_t))

    def nearest(t, keys):
        return min(keys, key=lambda x: abs(x - t)) if keys else None

    samples = []
    for t in times:
        ts = nearest(t, list(sw_by_t.keys()))
        tp = nearest(t, list(p_by_t.keys()))
        if ts is None or tp is None:
            continue
        sw = sw_by_t[ts]
        pres = p_by_t[tp]

        well_pressure: dict[str, float] = {}
        well_sat: dict[str, tuple[float, float, float]] = {}
        well_rate: dict[str, float] = {}
        for name, role in mesh.well_role.items():
            if name not in mesh.well_cell_id:
                continue
            cid = mesh.well_cell_id[name]
            ii, jj, kk = mesh.grid.ijk(cid)
            if role in ("injector", "producer", "observer_p"):
                val = float(pres[kk, jj, ii])
                if np.isfinite(val):
                    # p_series_pa is already Pa
                    well_pressure[name] = val
            if role in ("injector", "producer", "observer_s"):
                s = float(sw[kk, jj, ii])
                if not np.isfinite(s):
                    s = 0.8 if role == "injector" else 0.25
                well_sat[name] = (s, max(0.0, 1.0 - s), 0.0)
            spec = (meta.get("wells") or {}).get(name, {})
            if role == "injector":
                q = spec.get("rate_m3s")
                well_rate[name] = float(q) if q is not None else 5000.0 * 0.158987 / 86400.0
            elif role == "producer":
                q = spec.get("rate_m3s")
                well_rate[name] = float(q) if q is not None else -2500.0 * 0.158987 / 86400.0

        tr = nearest(t, rate_times)
        if tr is not None and "INJ" in well_rate and "PROD" in well_rate:
            wr = rates[tr]
            well_rate["INJ"] = float(wr["INJ"])
            well_rate["PROD"] = float(wr["PROD"])

        samples.append(
            SensorSample(
                time=float(t),
                well_pressure=well_pressure,
                well_saturation=well_sat,
                boundary=BoundaryConditions(),
                well_rate=well_rate,
            )
        )
    return samples


def run_one(
    case_name: str,
    case_dir: Path,
    *,
    n_total: int,
    layout: str,
) -> dict:
    truth_files = list(case_dir.glob("truth_*.json"))
    if not truth_files:
        return {"case": case_name, "ok": False, "error": "no truth json"}
    meta = json.loads(truth_files[0].read_text(encoding="utf-8"))
    out_path = next(case_dir.glob("*.out"), None)
    if out_path is None or not out_path.is_file():
        return {"case": case_name, "ok": False, "error": "missing .out"}

    g = meta["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    sw_series = parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)
    p_series_psi = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    if len(sw_series) < 2 or len(p_series_psi) < 1:
        return {
            "case": case_name,
            "ok": False,
            "error": f"insufficient grids sw={len(sw_series)} p={len(p_series_psi)}",
        }
    p_series_pa = [(t, np.asarray(arr, dtype=float) * 6894.757293168) for t, arr in p_series_psi]

    # base mesh for design (wells only)
    base_mesh, ik, pk = mesh_with_probes(meta, [])
    n_p, n_s = split_n_probes(n_total)
    # keep probe budget modest on coarse CMG grids (avoid over-pinning)
    max_n = max(0, min(48, base_mesh.n_cells // 8))
    if n_p + n_s > max_n:
        scale = max_n / max(n_p + n_s, 1)
        n_p = int(n_p * scale)
        n_s = int(n_s * scale)
        n_p, n_s = split_n_probes(n_p + n_s)

    if n_p + n_s == 0:
        probe_pts: list[WellPoint] = []
        layout_used = "wells_only"
    elif layout == "uniform":
        specs = place_uniform_probes(base_mesh, n_p, n_s)
        probe_pts = [WellPoint(**s.as_well_point_kwargs()) for s in specs]
        layout_used = "uniform"
    else:
        var_p = field_variance_over_time(p_series_pa)
        var_s = field_variance_over_time(sw_series)
        specs = recommend_probes(
            base_mesh,
            n_p=n_p,
            n_s=n_s,
            mode="hybrid",
            prior_var_p=var_p,
            prior_var_s=var_s,
            seed=7,
        )
        probe_pts = [WellPoint(**s.as_well_point_kwargs()) for s in specs]
        layout_used = "adaptive"

    mesh, ik, pk = mesh_with_probes(meta, probe_pts)
    samples = build_samples(meta, mesh, out_path, ik, pk, sw_series, p_series_pa)
    if len(samples) < 2:
        return {"case": case_name, "ok": False, "error": "few samples"}

    k_bg = float(meta.get("background_perm_md", {}).get("kx", 50.0))
    k_prior = k_bg * 9.869233e-16
    # ES-MDA whenever ≥2 pressure hard sensors (wells and/or observer_p)
    n_p_obs = sum(
        1
        for n, r in mesh.well_role.items()
        if r in ("injector", "producer", "observer_p")
        and n in (samples[0].well_pressure or {})
    )
    use_esmda = bool(n_p_obs >= 2)
    history = run_time_series(
        mesh,
        samples,
        permeability_prior_m2=k_prior,
        porosity_prior=0.3,
        viscosity_pa_s=1.0e-3,
        n_k_iterations=2 if (n_p + n_s) < 4 else 3,
        assimilate_k=use_esmda,
        esmda_ne=12,
        esmda_assimilations=3,
        esmda_max_times=8,
    )
    last = history[-1]
    t_last, sw_cmg = sw_series[-1]
    sw_rel = rel_l2(last.sw, sw_cmg)

    # ΔSw footprint dice
    sw0 = sw_series[0][1]
    dsw_c = np.nan_to_num(np.abs(sw_cmg - sw0), nan=0.0)
    dsw_s = np.nan_to_num(np.abs(last.sw - history[0].sw), nan=0.0)
    thr_c = max(float(np.quantile(dsw_c, 0.75)), 1e-4)
    thr_s = max(float(np.quantile(dsw_s, 0.75)), 1e-4)
    dice = dice_masks(dsw_s >= thr_s, dsw_c >= thr_c)

    mask = truth_mask(meta)
    k = last.permeability
    k_ch = float(np.mean(k[mask])) if np.any(mask) else float("nan")
    k_out = float(np.mean(k[~mask])) if np.any(~mask) else float("nan")
    k_ratio = k_ch / k_out if k_out and np.isfinite(k_out) and k_out > 0 else float("nan")

    well_err = {}
    for nm, pobs in samples[-1].well_pressure.items():
        if nm not in mesh.well_cell_id:
            continue
        if mesh.well_role.get(nm) not in ("injector", "producer"):
            continue
        c = mesh.well_cell_id[nm]
        i, j, k_ = mesh.grid.ijk(c)
        well_err[nm] = abs(float(last.pressure[k_, j, i]) - float(pobs))

    interp_notes = [n for n in last.notes if "auto-interp" in n or "auto spatial" in n]

    # hold-out: cells never used as hard p/s probes/wells for pressure comparison
    hard = set(mesh.well_cell_id.values())
    free = [c for c in range(mesh.n_cells) if c not in hard]
    rng = np.random.default_rng(0)
    if free:
        hold = rng.choice(free, size=max(1, len(free) // 5), replace=False)
        # pressure hold-out vs CMG
        tp = p_series_pa[-1][1]
        err = []
        for c in hold:
            i, j, kk = mesh.grid.ijk(int(c))
            err.append((float(last.pressure[kk, j, i]) - float(tp[kk, j, i])) ** 2)
        p_hold_rmse = float(np.sqrt(np.mean(err))) if err else float("nan")
    else:
        p_hold_rmse = float("nan")

    return {
        "case": case_name,
        "ok": True,
        "n_total": n_total,
        "n_p": n_p,
        "n_s": n_s,
        "layout": layout_used,
        "assimilate_k": use_esmda,
        "n_times": len(samples),
        "sw_rel_l2": sw_rel,
        "delta_sw_dice": dice,
        "well_pressure_abs_err_pa": well_err,
        "k_channel_over_out": k_ratio,
        "k_mean": float(np.mean(k)),
        "p_holdout_rmse_pa": p_hold_rmse,
        "interp_notes_tail": interp_notes[-4:],
        "probe_names": [w.name for w in probe_pts],
    }


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# CMG 虚拟测点学习曲线",
        "",
        "从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。",
        "",
        "| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |",
        "|------|--------|---|---------|--------|-------------|------------|------------|----------------------|",
    ]
    for r in results:
        if not r.get("ok"):
            lines.append(
                f"| {r.get('case')} | — | {r.get('n_total')} | — | — | ERR: {r.get('error')} | | | |"
            )
            continue
        es = "Y" if r.get("assimilate_k") else "N"
        lines.append(
            f"| {r['case']} | {r['layout']} | {r['n_total']} | "
            f"{r['n_p']}/{r['n_s']} | {es} | {r['sw_rel_l2']:.4f} | {r['delta_sw_dice']:.3f} | "
            f"{r['k_channel_over_out']:.3f} | {r['p_holdout_rmse_pa']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 读法",
            "",
            "- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。",
            "- **ES-MDA=Y**：`invert_rock`（指示先验 + ES-MDA 更新 log k，再锁 k 正演）。",
            "- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。",
            "- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。",
            "- 井压硬约束误差应接近 0。",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _job(payload: tuple[str, str, int, str]) -> dict:
    """Picklable worker for process pool: (case_name, case_dir, n, layout)."""
    cname, cdir_s, n, layout = payload
    return run_one(cname, Path(cdir_s), n_total=n, layout=layout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CMG virtual probe N-sweep study")
    ap.add_argument("--cases", default="channel,fault", help="channel,fault")
    ap.add_argument("--n-list", default="0,4,8,12", help="comma total exclusive probes")
    ap.add_argument(
        "--layouts",
        default="uniform,adaptive",
        help="uniform,adaptive",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(os.cpu_count() or 2, 6)),
        help="parallel study jobs (process pool); max accuracy path, max wall-clock speed",
    )
    args = ap.parse_args(argv)

    case_map = {
        "channel": ("cmg_undulating_channel", CHANNEL),
        "fault": ("cmg_faulted_dogleg", FAULT),
        "channel_fine": ("cmg_undulating_channel_fine", CHANNEL_FINE),
        "fivespot": ("cmg_fivespot", FIVESPOT),
    }
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]

    tasks: list[tuple[str, str, int, str]] = []
    for ckey in cases:
        if ckey not in case_map:
            continue
        cname, cdir = case_map[ckey]
        for n in n_list:
            if n == 0:
                tasks.append((cname, str(cdir), 0, "uniform"))
                continue
            for layout in layouts:
                tasks.append((cname, str(cdir), n, layout))

    results: list[dict] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1 or len(tasks) <= 1:
        results = [_job(t) for t in tasks]
    else:
        # Windows-safe process pool; preserve submission order in output
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_job, t): i for i, t in enumerate(tasks)}
            ordered: list[dict | None] = [None] * len(tasks)
            for fut in as_completed(futs):
                ordered[futs[fut]] = fut.result()
            results = [r for r in ordered if r is not None]

    out_json = HERE / "probe_study_report.json"
    out_md = HERE / "PROBE_STUDY.md"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, out_md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"jobs={jobs} tasks={len(tasks)}")
    ok = sum(1 for r in results if r.get("ok"))
    print(f"ok {ok}/{len(results)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
