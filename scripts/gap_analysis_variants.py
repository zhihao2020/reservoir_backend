"""Quantify reconstruction gaps against local GEM shale-oil truth.

1. Monthly 10-year export: ``results/shale_oil`` kriging vs ``shale_oil/cmg_fields.csv``.
2. Six GEM variants: sample SR3 at 0.01–30 d onto the 27 bottom probes, run kriging
   (and k/phi inversion for het_k / layered_k), compare plume and rock fields.

Does not change the reconstruction algorithm. Writes ``results/gap_analysis/``.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.lab_case import load_lab_case
from src.core.units import MD_TO_M2
from src.programs.pipeline import run_pipeline

OUT = ROOT / "results" / "gap_analysis"
EXAMPLE = ROOT / "examples" / "shale_oil" / "case.yaml"
TRUTH_CSV = ROOT / "shale_oil" / "cmg_fields.csv"
RECON = ROOT / "results" / "shale_oil"
VAR_DIR = ROOT / "results" / "shailoil_variants"

NX = NY = NZ = 15
DX = 0.02
DAY_S = 86400.0
TARGETS = (0.01, 0.1, 1.0, 3.0, 7.0, 15.0, 30.0)
VARIANTS = ("base", "het_k", "layered_k", "rate_x3", "rate_div3", "inj_bottom")
INVERT = {"het_k", "layered_k"}
KPA_TO_PA = 1.0e3
# SR3 PERMI is stored in darcy (see results/shailoil_variants/README.md).
DARCY_TO_MD = 1000.0


def _s(value) -> str:
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return str(value).split("\x00", 1)[0]


def to_soft(arr: np.ndarray) -> np.ndarray:
    """GEM I-fastest, K=1 at the top → software k=0 at the bottom, flat."""
    return np.asarray(arr, dtype=float).reshape(NZ, NY, NX)[::-1].reshape(-1)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _layer_mean(field: np.ndarray) -> list[float]:
    return [float(v) for v in field.reshape(NZ, NY, NX).mean(axis=(1, 2))]


def _bottom_top(field: np.ndarray) -> tuple[float, float]:
    layers = field.reshape(NZ, NY, NX)
    return float(layers[:3].mean()), float(layers[-3:].mean())


def plume_row(truth_sg: np.ndarray, recon_sg: np.ndarray, truth_p: np.ndarray, recon_p: np.ndarray,
              truth_so: np.ndarray, recon_so: np.ndarray, truth_z, recon_co2) -> dict:
    sb_t, st_t = _bottom_top(truth_sg)
    sb_r, st_r = _bottom_top(recon_sg)
    p_mean = float(np.mean(np.abs(truth_p)))
    row = {
        "p_rmse_pa": _rmse(recon_p, truth_p),
        "p_nrmse": _rmse(recon_p, truth_p) / p_mean if p_mean else None,
        "sg_rmse": _rmse(recon_sg, truth_sg),
        "so_rmse": _rmse(recon_so, truth_so),
        "sg_corr": _corr(recon_sg, truth_sg),
        "sg_bottom_truth": sb_t,
        "sg_bottom_recon": sb_r,
        "sg_top_truth": st_t,
        "sg_top_recon": st_r,
        "n_sg_gt_0.1_truth": int((truth_sg > 0.1).sum()),
        "n_sg_gt_0.1_recon": int((recon_sg > 0.1).sum()),
        "sg_mean_truth": float(truth_sg.mean()),
        "sg_mean_recon": float(recon_sg.mean()),
    }
    if truth_z is not None and recon_co2 is not None:
        row["zco2_mean_truth"] = float(np.mean(truth_z))
        row["co2_mean_recon"] = float(np.mean(recon_co2))
        row["co2_vs_zco2_corr"] = _corr(recon_co2, truth_z)
    return row


def _agg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def load_monthly_truth():
    times: list[float] = []
    cols = {n: [] for n in ("p_pa", "sw", "sg", "so", "phi", "k_m2")}
    cur = None
    with TRUTH_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            if cur is None or t != cur:
                cur = t
                times.append(cur)
            for n in cols:
                cols[n].append(float(row[n]))
    n_t = len(times)
    n_c = len(cols["p_pa"]) // n_t
    out = {n: np.array(cols[n], dtype=float).reshape(n_t, n_c) for n in cols}
    return np.array(times), out


def baseline() -> dict:
    times, truth = load_monthly_truth()
    recon = {name: np.load(RECON / f"{name}.npy") for name in ("p", "sw", "so", "sg", "phi", "k")}
    n = min(times.size, recon["p"].shape[0])
    times, recon = times[:n], {k: v[:n] for k, v in recon.items()}
    truth = {k: v[:n] for k, v in truth.items()}
    per = []
    for i, t in enumerate(times):
        row = plume_row(truth["sg"][i], recon["sg"][i], truth["p_pa"][i], recon["p"][i],
                        truth["so"][i], recon["so"][i], None, None)
        row["day"] = float(t / DAY_S)
        row["k_rmse_m2"] = _rmse(recon["k"][i], truth["k_m2"][i])
        row["phi_rmse"] = _rmse(recon["phi"][i], truth["phi"][i])
        per.append(row)
    sg_sse = np.sum((recon["sg"] - truth["sg"]) ** 2, axis=1)
    first = per[:10]
    rest = per[10:]

    def pack(rows: list[dict]) -> dict:
        return {
            "n_steps": len(rows),
            "day_first": rows[0]["day"] if rows else None,
            "day_last": rows[-1]["day"] if rows else None,
            "p_nrmse": _agg(rows, "p_nrmse"),
            "sg_rmse": _agg(rows, "sg_rmse"),
            "so_rmse": _agg(rows, "so_rmse"),
            "sg_bottom_truth": _agg(rows, "sg_bottom_truth"),
            "sg_bottom_recon": _agg(rows, "sg_bottom_recon"),
            "n_sg_gt_0.1_truth": _agg(rows, "n_sg_gt_0.1_truth"),
            "n_sg_gt_0.1_recon": _agg(rows, "n_sg_gt_0.1_recon"),
        }

    return {
        "source_truth": str(TRUTH_CSV.relative_to(ROOT)),
        "source_recon": str(RECON.relative_to(ROOT)),
        "note": "Existing offline kriging (forward.model none, k_homogeneous true) vs the 10-year monthly GEM export. First snapshot is day 30, so the early plume is not in this file.",
        "n_times": int(n),
        "overall": pack(per),
        "first_10_steps": pack(first),
        "after_step_10": pack(rest),
        "sg_sse_fraction_first_10": float(sg_sse[:10].sum() / sg_sse.sum()) if sg_sse.sum() else None,
        "day30": per[0],
        "per_time": [
            {
                "day": r["day"],
                "sg_rmse": r["sg_rmse"],
                "p_nrmse": r["p_nrmse"],
                "sg_bottom_truth": r["sg_bottom_truth"],
                "sg_bottom_recon": r["sg_bottom_recon"],
                "n_sg_gt_0.1_truth": r["n_sg_gt_0.1_truth"],
                "n_sg_gt_0.1_recon": r["n_sg_gt_0.1_recon"],
            }
            for r in per
        ],
    }


def _snaps(sr3: Path):
    import h5py

    with h5py.File(sr3, "r") as f:
        tt = f["General/MasterTimeTable"][:]
        tmap = {int(r["Index"]): float(r["Offset in days"]) for r in tt}
        sp = f["SpatialProperties"]
        found = []
        for key in sp:
            if not key.isdigit() or "SG" not in sp[key]:
                continue
            found.append((tmap[int(key)], int(key)))
        found.sort()
        chosen = []
        for target in TARGETS:
            day, idx = min(found, key=lambda r: abs(r[0] - target))
            if abs(day - target) > 1.0e-3:
                raise RuntimeError(f"{sr3.name}: no snapshot near {target} d (nearest {day})")
            g = sp[f"{idx:06d}"]
            chosen.append(
                {
                    "day": day,
                    "step": idx,
                    "p": to_soft(g["PRES"][:]) * KPA_TO_PA,
                    "sg": to_soft(g["SG"][:]),
                    "so": to_soft(g["SO"][:]),
                    "sw": to_soft(g["SW"][:]),
                    "z": to_soft(g["Z(5)"][:]),
                    "phi": to_soft(g["POROS"][:]),
                    "k": to_soft(g["PERMI"][:]) * DARCY_TO_MD * MD_TO_M2,
                }
            )
        wells = [_s(r["Name"]) for r in f["TimeSeries/WELLS/WellTable"][:]]
        wvars = [_s(v) for v in f["TimeSeries/WELLS/Variables"][:]]
        wts = np.asarray(f["TimeSeries/WELLS/Timesteps"][:], dtype=int)
        data = f["TimeSeries/WELLS/Data"]
        iv = {name: wvars.index(name) for name in ("BHP", "OILRATRC", "GASRATRC", "WATRATRC")}
        series = []
        for snap in chosen:
            row_i = int(np.where(wts == snap["step"])[0][0])
            block = np.asarray(data[row_i], dtype=float)
            series.append(
                {
                    "day": snap["day"],
                    "bhp_kpa": block[iv["BHP"]],
                    "oil_rc": block[iv["OILRATRC"]],
                    "gas_rc": block[iv["GASRATRC"]],
                    "wat_rc": block[iv["WATRATRC"]],
                }
            )
        layers: dict[str, list[tuple[int, int, int]]] = {}
        for rec in f["TimeSeries/LAYERS/LayerTable"][:]:
            parent = _s(rec["Parent"])
            parts = _s(rec["Name"]).split(",")
            if len(parts) != 3:
                continue
            ijk = tuple(int(float(p)) for p in parts)
            layers.setdefault(parent, []).append(ijk)
    return chosen, wells, series, layers


def _xyz(i1: int, j1: int, k1: int) -> tuple[float, float, float]:
    i0, j0 = i1 - 1, j1 - 1
    k0 = NZ - k1
    return ((i0 + 0.5) * DX, (j0 + 0.5) * DX, (k0 + 0.5) * DX)


def write_case(dest: Path, snaps, well_names, series, layers) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    probes = list(csv.DictReader((ROOT / "examples" / "shale_oil" / "probes.csv").open(encoding="utf-8")))
    shutil.copyfile(ROOT / "examples" / "shale_oil" / "probes.csv", dest / "probes.csv")
    obs_path = dest / "observations.csv"
    with obs_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "probe", "quantity", "value", "unit"])
        for snap in snaps:
            t = f"{snap['day'] * DAY_S:.8f}"
            for probe in probes:
                i = int(round(float(probe["x_m"]) / DX - 0.5))
                j = int(round(float(probe["y_m"]) / DX - 0.5))
                k = int(round(float(probe["z_m"]) / DX - 0.5))
                cell = k * NX * NY + j * NX + i
                pid = probe["id"]
                w.writerow([t, pid, "pressure", f"{snap['p'][cell]:.8g}", "Pa"])
                w.writerow([t, pid, "sg", f"{snap['sg'][cell]:.8g}", ""])
                w.writerow([t, pid, "so", f"{snap['so'][cell]:.8g}", ""])
                w.writerow([t, pid, "sw", f"{snap['sw'][cell]:.8g}", ""])
    geofac = {"INJ": 0.34}
    with (dest / "wells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "x_m", "y_m", "z_m", "x2_m", "y2_m", "z2_m", "geofac"])
        for name in well_names:
            perfs = layers.get(name) or []
            if not perfs:
                raise RuntimeError(f"no perforations for {name}")
            x, y, z = _xyz(*perfs[0])
            x2, y2, z2 = _xyz(*perfs[-1])
            kind = "injector" if name == "INJ" else "producer"
            w.writerow([name, kind, x, y, z, x2, y2, z2, geofac.get(name, 1.0)])
    with (dest / "series.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "well", "pw_pa", "q_m3s", "qw_m3s", "qo_m3s", "qg_m3s"])
        for row, names in zip(series, [well_names] * len(series)):
            t = f"{row['day'] * DAY_S:.8f}"
            for j, name in enumerate(names):
                sign = 1.0 if name == "INJ" else -1.0
                qo = sign * float(row["oil_rc"][j]) / DAY_S
                qg = sign * float(row["gas_rc"][j]) / DAY_S
                qw = sign * float(row["wat_rc"][j]) / DAY_S
                w.writerow([t, name, f"{float(row['bhp_kpa'][j]) * KPA_TO_PA:.8g}",
                            f"{qw + qo + qg:.8g}", f"{qw:.8g}", f"{qo:.8g}", f"{qg:.8g}"])
    text = EXAMPLE.read_text(encoding="utf-8")
    # This script is the kriging baseline. The example case may enable the
    # compositional forward; keep saturations interpolated here.
    text = text.replace("model: compositional", "model: none")
    (dest / "case.yaml").write_text(text, encoding="utf-8")
    (dest / "case_invert.yaml").write_text(
        text.replace("k_homogeneous: true", "k_homogeneous: false"), encoding="utf-8"
    )
    np.savez_compressed(
        dest / "truth.npz",
        days=np.array([s["day"] for s in snaps]),
        p=np.stack([s["p"] for s in snaps]),
        sg=np.stack([s["sg"] for s in snaps]),
        so=np.stack([s["so"] for s in snaps]),
        sw=np.stack([s["sw"] for s in snaps]),
        z=np.stack([s["z"] for s in snaps]),
        phi=np.stack([s["phi"] for s in snaps]),
        k=np.stack([s["k"] for s in snaps]),
    )


def _rock(truth_k, recon_k, truth_phi, recon_phi) -> dict:
    # Static rock: compare the first reconstructed slice (pipeline repeats it).
    kt, kr = truth_k[0], recon_k[0]
    pt, pr = truth_phi[0], recon_phi[0]
    kt_md = kt / MD_TO_M2
    kr_md = kr / MD_TO_M2
    return {
        "k_rmse_md": _rmse(kr_md, kt_md),
        "k_nrmse_vs_mean": _rmse(kr_md, kt_md) / float(np.mean(kt_md)),
        "k_log_corr": _corr(np.log(np.maximum(kr, 1e-30)), np.log(np.maximum(kt, 1e-30))),
        "phi_rmse": _rmse(pr, pt),
        "phi_truth_mean": float(pt.mean()),
        "phi_recon_mean": float(pr.mean()),
        "k_truth_layer_md": _layer_mean(kt_md),
        "k_recon_layer_md": _layer_mean(kr_md),
    }


def run_one(case_path: Path, snaps) -> dict:
    t0 = time.perf_counter()
    fields = run_pipeline(load_lab_case(case_path))
    elapsed = time.perf_counter() - t0
    rows = []
    for i, snap in enumerate(snaps):
        co2 = None if fields.co2 is None else fields.co2[i]
        row = plume_row(snap["sg"], fields.sg[i], snap["p"], fields.p[i], snap["so"], fields.so[i], snap["z"], co2)
        row["day"] = snap["day"]
        rows.append(row)
    diag = {k: fields.diagnostics.get(k) for k in (
        "converged", "status", "k_mean_error", "phi_mean_error", "phi_identifiable",
        "nfev", "cost", "relperm",
    )}
    return {
        "elapsed_s": round(elapsed, 3),
        "forward_model": "none",
        "overall": {
            "p_nrmse": _agg(rows, "p_nrmse"),
            "sg_rmse": _agg(rows, "sg_rmse"),
            "so_rmse": _agg(rows, "so_rmse"),
            "sg_corr": _agg(rows, "sg_corr"),
            "sg_bottom_truth": _agg(rows, "sg_bottom_truth"),
            "sg_bottom_recon": _agg(rows, "sg_bottom_recon"),
            "n_sg_gt_0.1_truth": _agg(rows, "n_sg_gt_0.1_truth"),
            "n_sg_gt_0.1_recon": _agg(rows, "n_sg_gt_0.1_recon"),
            "co2_vs_zco2_corr": _agg(rows, "co2_vs_zco2_corr"),
            "zco2_mean_truth": _agg(rows, "zco2_mean_truth"),
            "co2_mean_recon": _agg(rows, "co2_mean_recon"),
        },
        "rock": _rock(
            np.stack([s["k"] for s in snaps]),
            fields.k,
            np.stack([s["phi"] for s in snaps]),
            fields.phi,
        ),
        "diagnostics": diag,
        "per_time": rows,
    }


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _dump(obj) -> str:
    return json.dumps(_json_ready(obj), indent=2, allow_nan=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("baseline", flush=True)
    report = {"baseline_10y": baseline()}
    report["variants"] = {}
    for name in VARIANTS:
        print(f"ingest {name}", flush=True)
        sr3 = VAR_DIR / name / f"shailoil_{name}.sr3"
        snaps, wells, series, layers = _snaps(sr3)
        dest = OUT / name
        write_case(dest, snaps, wells, series, layers)
        print(f"kriging {name}", flush=True)
        block = {"kriging_homogeneous_k": run_one(dest / "case.yaml", snaps)}
        if name in INVERT:
            print(f"invert {name}", flush=True)
            block["k_phi_inversion"] = run_one(dest / "case_invert.yaml", snaps)
        report["variants"][name] = block
        (OUT / "metrics.json").write_text(_dump(report), encoding="utf-8")
        print(f"  sg_rmse {block['kriging_homogeneous_k']['overall']['sg_rmse']:.4f}", flush=True)
    print("wrote", OUT / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
