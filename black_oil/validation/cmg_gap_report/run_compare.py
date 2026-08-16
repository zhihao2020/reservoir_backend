"""Compare CMG heterogeneous IMEX fields vs software four-field reconstruction.

Reads existing CMG .out (channel / fault), extracts multi-time Sw (and wells),
runs ``run_time_series`` on the same sensors, writes quantitative gap metrics
and a Markdown report.

Usage (from repo root):
  python validation/cmg_gap_report/run_compare.py
  python validation/cmg_gap_report/run_compare.py --n-probes 8 --probe-layout adaptive
"""

from __future__ import annotations

import argparse
import json
import re
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
from cmg_io.grid_parse import parse_grid_series, parse_surface_rates_m3s  # noqa: E402

HERE = Path(__file__).resolve().parent
CHANNEL = VAL / "cmg_channel_3d"
FAULT = VAL / "cmg_fault_3d"


def ft_to_m(x: float) -> float:
    return float(x) * 0.3048


def psi_to_pa(p: float) -> float:
    return float(p) * 6894.757293168


def parse_sw_grids(out_path: Path, *, nx: int, ny: int, nz: int):
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


def parse_bhp(out_path: Path) -> dict[float, tuple[float, float]]:
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


def mesh_from_truth(meta: dict):
    g = meta["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    di, dj = ft_to_m(g["di_ft"]), ft_to_m(g["dj_ft"])
    dk = np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float)
    lx, ly, lz = nx * di, ny * dj, float(np.sum(dk))
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    dx = np.full(nx, di)
    dy = np.full(ny, dj)
    wells_meta = meta["wells"]

    def mid_k(w):
        if "k_perfs" in w and w["k_perfs"]:
            ks = sorted(int(x) for x in w["k_perfs"])
            return ks[len(ks) // 2]
        return int(w.get("k", 1))

    z_edges = np.concatenate([[0.0], np.cumsum(dk)])

    def center(i, j, k):
        return (
            (i - 0.5) * di,
            (j - 0.5) * dj,
            0.5 * (z_edges[k - 1] + z_edges[k]),
        )

    wi, wp = wells_meta["INJ"], wells_meta["PROD"]
    ix, iy, iz = center(wi["i"], wi["j"], mid_k(wi))
    px, py, pz = center(wp["i"], wp["j"], mid_k(wp))
    wells = [
        WellPoint("INJ", ix, iy, iz, role="injector"),
        WellPoint("PROD", px, py, pz, role="producer"),
    ]
    return build_mesh(bounds, dx, dy, dk, wells=wells), mid_k(wi), mid_k(wp)


def truth_mask(meta: dict) -> np.ndarray:
    g = meta["grid"]
    mask = np.zeros((g["nz"], g["ny"], g["nx"]), dtype=bool)
    for i, j, k in meta["channel_blocks_ijk"]:
        mask[k - 1, j - 1, i - 1] = True
    return mask


def parse_surface_rates_stb_day(out_path: Path) -> dict[float, dict[str, float]]:
    """Best-effort water rate parse from CMG well tables (STB/day).

    Returns time -> {INJ: +q, PROD: -q} in m^3/s when found.
    """
    text = out_path.read_text(encoding="latin-1", errors="ignore")
    # Cumulative injection water MSTB and time — fallback to constant ops from case
    # Prefer instantaneous rates near "Water" "STB/day" lines
    by_t: dict[float, dict[str, float]] = {}
    # Pattern from well report: Water STB/day columns for inj/prod
    for m in re.finditer(
        r"Time\s*=\s*([0-9.]+).*?Water\s+STB/day\s+\+\s+([0-9.E+-]+)\s+\+\s+([0-9.E+-]+)",
        text,
        re.S | re.I,
    ):
        t = float(m.group(1))
        # columns may be inj/prod depending on order — use magnitudes
        a, b = float(m.group(2)), float(m.group(3))
        # convert STB/day → m3/s : 1 STB = 0.158987 m3
        def stbday_to_m3s(x: float) -> float:
            return x * 0.158987 / 86400.0

        # typically one positive inj one production negative in report signs
        q1, q2 = stbday_to_m3s(a), stbday_to_m3s(b)
        by_t[t] = {"INJ": abs(q1), "PROD": -abs(q2) if abs(q2) > 0 else -abs(q1)}
    return by_t


def samples_from_cmg(meta: dict, out_path: Path, mesh, ik: int, pk: int):
    g = meta["grid"]
    sw_series = parse_sw_grids(out_path, nx=g["nx"], ny=g["ny"], nz=g["nz"])
    p_series = parse_grid_series(
        out_path, field="pressure", nx=g["nx"], ny=g["ny"], nz=g["nz"]
    )
    p_by_t = {float(t): np.asarray(a, dtype=float) * 6894.757293168 for t, a in p_series}
    bhp = parse_bhp(out_path)
    bhp_times = sorted(bhp.keys())
    rates = parse_surface_rates_m3s(out_path)
    if rates and max(float(v.get("INJ", 0.0)) for v in rates.values()) <= 1.0e-12:
        rates = {}
    rate_times = sorted(rates.keys())

    def nearest_bhp(t):
        if not bhp_times:
            return 4500.0, 2000.0
        nt = min(bhp_times, key=lambda x: abs(x - t))
        return bhp[nt]

    def nearest_rate(t):
        # default: seawater inject ~5000 STB/d, oil ~2500 STB/d from mxspr006
        if not rate_times:
            q_inj = 5000.0 * 0.158987 / 86400.0
            q_prod = -2500.0 * 0.158987 / 86400.0
            return {"INJ": q_inj, "PROD": q_prod}
        nt = min(rate_times, key=lambda x: abs(x - t))
        return rates[nt]

    def nearest_p(t):
        if not p_by_t:
            return None
        nt = min(p_by_t.keys(), key=lambda x: abs(x - t))
        return p_by_t[nt]

    wi, wp = meta["wells"]["INJ"], meta["wells"]["PROD"]
    samples = []
    for t, sw in sw_series:
        sw_inj = float(sw[ik - 1, wi["j"] - 1, wi["i"] - 1])
        sw_prod = float(sw[pk - 1, wp["j"] - 1, wp["i"] - 1])
        if not np.isfinite(sw_inj):
            sw_inj = 0.8
        if not np.isfinite(sw_prod):
            sw_prod = 0.2
        bi, bp = nearest_bhp(t)
        p_inj, p_prod = psi_to_pa(bi), psi_to_pa(bp)
        wr = nearest_rate(t)
        well_pressure = {"INJ": p_inj, "PROD": p_prod}
        well_sat = {
            "INJ": (sw_inj, max(0.0, 1.0 - sw_inj), 0.0),
            "PROD": (sw_prod, max(0.0, 1.0 - sw_prod), 0.0),
        }
        pres = nearest_p(t)
        if pres is not None:
            gi, gj, gk = int(wi["i"]) - 1, int(wi["j"]) - 1, int(ik) - 1
            pi, pj, pk_ = int(wp["i"]) - 1, int(wp["j"]) - 1, int(pk) - 1
            if np.isfinite(pres[gk, gj, gi]):
                well_pressure["INJ"] = float(pres[gk, gj, gi])
            if np.isfinite(pres[pk_, pj, pi]):
                well_pressure["PROD"] = float(pres[pk_, pj, pi])
        for name, role in mesh.well_role.items():
            if role not in ("observer_p", "observer_s"):
                continue
            cid = mesh.well_cell_id[name]
            ii, jj, kk = mesh.grid.ijk(cid)
            if role == "observer_p" and pres is not None:
                val = float(pres[kk, jj, ii])
                if np.isfinite(val):
                    well_pressure[name] = val
            elif role == "observer_s":
                s = float(sw[kk, jj, ii])
                if not np.isfinite(s):
                    s = 0.3
                well_sat[name] = (s, max(0.0, 1.0 - s), 0.0)
        samples.append(
            SensorSample(
                time=float(t),
                well_pressure=well_pressure,
                well_saturation=well_sat,
                boundary=BoundaryConditions(),
                well_rate={"INJ": float(wr["INJ"]), "PROD": float(wr["PROD"])},
            )
        )
    return samples, sw_series


def rel_l2(a, b) -> float:
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=float), nan=0.0)
    den = float(np.linalg.norm(b.ravel())) + 1.0e-30
    return float(np.linalg.norm((a - b).ravel()) / den)


def compare_case(
    name: str,
    case_dir: Path,
    *,
    n_probes: int = 0,
    probe_layout: str = "uniform",
) -> dict:
    truth_path = case_dir / ("truth_channel.json" if "channel" in name else "truth_fault.json")
    if not truth_path.is_file():
        # channel uses truth_channel, fault uses truth_fault
        cands = list(case_dir.glob("truth_*.json"))
        if not cands:
            return {"case": name, "ok": False, "error": "no truth json"}
        truth_path = cands[0]
    out_path = next(case_dir.glob("*.out"), None)
    if out_path is None or not out_path.is_file():
        return {"case": name, "ok": False, "error": "missing CMG .out"}

    meta = json.loads(truth_path.read_text(encoding="utf-8"))
    mesh, ik, pk = mesh_from_truth(meta)
    n_probes = max(0, int(n_probes))
    if n_probes > 0:
        n_p, n_s = split_n_probes(n_probes)
        g = meta["grid"]
        if probe_layout == "adaptive":
            p_series = parse_grid_series(
                out_path, field="pressure", nx=g["nx"], ny=g["ny"], nz=g["nz"]
            )
            sw_series0 = parse_sw_grids(out_path, nx=g["nx"], ny=g["ny"], nz=g["nz"])
            var_p = field_variance_over_time(
                [(t, np.asarray(a, dtype=float) * 6894.757293168) for t, a in p_series]
            )
            var_s = field_variance_over_time(sw_series0)
            specs = recommend_probes(
                mesh, n_p=n_p, n_s=n_s, mode="hybrid", prior_var_p=var_p, prior_var_s=var_s
            )
        else:
            specs = place_uniform_probes(mesh, n_p, n_s)
        # rebuild mesh with probes
        extra = [WellPoint(**s.as_well_point_kwargs()) for s in specs]
        # re-call mesh_from_truth structure with extra wells
        base_wells = [
            WellPoint(
                n,
                float(mesh.x[mesh.well_cell_id[n]]),
                float(mesh.y[mesh.well_cell_id[n]]),
                float(mesh.z[mesh.well_cell_id[n]]),
                role=mesh.well_role[n],
            )
            for n in ("INJ", "PROD")
            if n in mesh.well_cell_id
        ]
        g = meta["grid"]
        nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
        di, dj = ft_to_m(g["di_ft"]), ft_to_m(g["dj_ft"])
        dk = np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float)
        lx, ly, lz = nx * di, ny * dj, float(np.sum(dk))
        bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
        mesh = build_mesh(
            bounds, np.full(nx, di), np.full(ny, dj), dk, wells=base_wells + extra
        )
    samples, sw_series = samples_from_cmg(meta, out_path, mesh, ik, pk)
    if len(samples) < 2:
        return {"case": name, "ok": False, "error": "insufficient CMG time samples"}

    # heterogeneous prior: median background, not constant high
    k_bg = float(meta.get("background_perm_md", {}).get("kx", 50.0))
    # md → m2 : 1 md = 9.869e-16 m2
    k_prior = k_bg * 9.869233e-16
    history = run_time_series(
        mesh,
        samples,
        permeability_prior_m2=k_prior,
        porosity_prior=0.3,
        viscosity_pa_s=1.0e-3,
        n_k_iterations=2,
        assimilate_k=True,
        esmda_ne=12,
        esmda_assimilations=3,
    )

    # align last CMG Sw with last history
    t_cmg, sw_cmg = sw_series[-1]
    last = history[-1]
    sw_rel = rel_l2(last.sw, sw_cmg)
    # well match
    well_err = {}
    for nm, pobs in samples[-1].well_pressure.items():
        c = mesh.well_cell_id[nm]
        i, j, k = mesh.grid.ijk(c)
        well_err[nm] = abs(float(last.pressure[k, j, i]) - float(pobs))

    # channel enrichment: mean k in truth channel vs outside
    mask = truth_mask(meta)
    k = last.permeability
    k_ch = float(np.mean(k[mask])) if np.any(mask) else float("nan")
    k_out = float(np.mean(k[~mask])) if np.any(~mask) else float("nan")

    # water footprint dice CMG ΔSw vs software ΔSw
    sw0 = sw_series[0][1]
    dsw_cmg = np.nan_to_num(np.abs(sw_cmg - sw0), nan=0.0)
    dsw_sw = np.abs(last.sw - history[0].sw)
    thr_c = max(float(np.quantile(dsw_cmg, 0.75)), 1e-4)
    thr_s = max(float(np.quantile(dsw_sw, 0.75)), 1e-4)
    m_c = dsw_cmg >= thr_c
    m_s = dsw_sw >= thr_s
    inter = float(np.sum(m_c & m_s))
    dice = 2 * inter / (float(np.sum(m_c) + np.sum(m_s)) + 1e-30)

    return {
        "case": name,
        "ok": True,
        "cmg_out": str(out_path),
        "truth": str(truth_path),
        "n_times_cmg": len(sw_series),
        "n_times_software": len(history),
        "grid": meta["grid"],
        "heterogeneous": True,
        "background_k_md": k_bg,
        "channel_k_md": meta.get("channel_perm_md", {}),
        "metrics": {
            "sw_field_rel_l2_last": sw_rel,
            "well_pressure_abs_err_Pa": well_err,
            "k_mean_in_channel_m2": k_ch,
            "k_mean_outside_channel_m2": k_out,
            "k_channel_over_outside": (k_ch / k_out) if k_out > 0 else None,
            "delta_sw_footprint_dice_cmg_vs_software": dice,
            "cmg_sw_mean_last": float(np.nanmean(sw_cmg)),
            "software_sw_mean_last": float(np.mean(last.sw)),
            "software_k_mean_m2": float(np.mean(k)),
            "software_k_std_m2": float(np.std(k)),
        },
        "notes_last": last.notes[-8:],
    }


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# CMG vs 传感器四场软件 — 差距报告",
        "",
        f"生成：`validation/cmg_gap_report`",
        "",
        "## 结论摘要",
        "",
        "| 维度 | CMG | 本软件 | 差距判断 |",
        "|------|-----|--------|----------|",
        "| 角色 | 全物理黑油/海水驱 **正演** | 传感器 **反演/重建** | 目标不同，不追求逐格数值等价 |",
        "| 网格 | VARI/CART + DTOP/FAULT/TRANSI | 正交结构化 | 软件不保留角点/断层网格几何 |",
        "| 压力 | 多相耦合求解 | TPFA 单相 + 井 Dirichlet + 边界 P/Q | 井点可精确匹配；全场近似 |",
        "| 饱和度 | 多相输运 | 锁 k 后两相 fw 迎风输运 | 足迹应跟通道；非 IMEX 逐格等价 |",
        "| 物性 | 输入已知非均质 k | 指示先验 + ES-MDA 后验 log k | 欠定；通道区 k 应由观测抬升 |",
        "| 验证模型 | 起伏通道 / 断层狗腿（**非均质**） | 同算例井传感器驱动 | 禁止均质对照作为通过标准 |",
        "",
        "## 定量对比（本机 CMG .out）",
        "",
    ]
    for r in results:
        lines.append(f"### {r.get('case')}")
        if not r.get("ok"):
            lines.append(f"- **失败**: {r.get('error')}")
            lines.append("")
            continue
        m = r["metrics"]
        lines.extend(
            [
                f"- CMG 输出: `{r['cmg_out']}`",
                f"- 时刻数: CMG={r['n_times_cmg']}, 软件={r['n_times_software']}",
                f"- 网格: {r['grid']}",
                f"- **全场 Sw 相对 L2（末时刻）**: {m['sw_field_rel_l2_last']:.4f}",
                f"- **井点压力绝对误差 (Pa)**: {m['well_pressure_abs_err_Pa']}",
                f"- **ΔSw 足迹 Dice (CMG vs 软件)**: {m['delta_sw_footprint_dice_cmg_vs_software']:.3f}",
                f"- **通道内/外平均 k**: {m['k_mean_in_channel_m2']:.3e} / {m['k_mean_outside_channel_m2']:.3e} "
                f"(比={m['k_channel_over_outside']})",
                f"- CMG/软件 末时刻平均 Sw: {m['cmg_sw_mean_last']:.3f} / {m['software_sw_mean_last']:.3f}",
                f"- 软件 k 均值±std: {m['software_k_mean_m2']:.3e} ± {m['software_k_std_m2']:.3e}",
                "",
            ]
        )
    lines.extend(
        [
            "## 使用效果怎么看",
            "",
            "1. **井点压力**：应接近 0 误差 → 传感器硬约束工作正常。",
            "2. **ΔSw 足迹 Dice**：越高说明软件抓住了与 CMG 一致的水驱通道形态。",
            "3. **通道内外 k 比 > 1**：说明物性反演对非均质通道有区分能力（非全场常数）。",
            "4. **全场 Sw L2**：传感器反演不会等于 CMG 全物理饱和度；L2 偏大是正常差距。",
            "",
            "## 建议改进优先级",
            "",
            "1. 将 CMG 井控/产注量写入边界/井源项，缩小压力与足迹差。",
            "2. 断层狗腿仍弱：需要更强局部化或把盲测点 p/Sw 当成主指标。",
            "3. 主指标是盲测点 p/Sw 与 ΔSw Dice，不是全场 Sw L2。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CMG vs software gap report")
    ap.add_argument(
        "--n-probes",
        type=int,
        default=0,
        help="virtual exclusive probes sampled from CMG grids (default 0 = wells only)",
    )
    ap.add_argument(
        "--probe-layout",
        choices=("uniform", "adaptive"),
        default="uniform",
        help="probe placement when --n-probes > 0",
    )
    args = ap.parse_args(argv)

    results = []
    if (CHANNEL / "mxspr006_channel.out").is_file():
        results.append(
            compare_case(
                "cmg_undulating_channel",
                CHANNEL,
                n_probes=args.n_probes,
                probe_layout=args.probe_layout,
            )
        )
    if (FAULT / "mxspr006_fault.out").is_file():
        results.append(
            compare_case(
                "cmg_faulted_dogleg",
                FAULT,
                n_probes=args.n_probes,
                probe_layout=args.probe_layout,
            )
        )

    report = {
        "n_probes": args.n_probes,
        "probe_layout": args.probe_layout,
        "results": results,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "gap_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(results, HERE / "GAP_REPORT.md")
    print(json.dumps(report, indent=2))
    print(f"\nMarkdown: {HERE / 'GAP_REPORT.md'}")
    return 0 if results and all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
