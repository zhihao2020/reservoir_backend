"""Parse PRES/SW from the five shale analog .out files and write a smoke report.

Confirms fields are finite, non-constant, and fracture cells deplete more than
far-field matrix. Does not run inversion. CMG is a ruler only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VAL = HERE.parent
ROOT = VAL.parents[1]
CMG_IO = ROOT / "black_oil" / "validation"
if str(CMG_IO) not in sys.path:
    sys.path.insert(0, str(CMG_IO))
from cmg_io.grid_parse import parse_grid_series  # noqa: E402

CASE_DIR = {
    "S1": "cmg_s1_hw5frac",
    "S2": "cmg_s2_hw9frac",
    "S3": "cmg_s3_twohw",
    "S4": "cmg_s4_parent_child",
    "S5": "cmg_s5_shutin",
}


def _mask_from_ijk(blocks: list[list[int]], nx: int, ny: int, nz: int) -> np.ndarray:
    mask = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in blocks:
        if 1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz:
            mask[k - 1, j - 1, i - 1] = True
    return mask


def _region_stats(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def smoke_one(case: str) -> dict:
    d = VAL / CASE_DIR[case]
    truth_path = d / f"truth_{case.lower()}.json"
    out_path = d / f"mxshale_{case.lower()}.out"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    grid = truth["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    frac = _mask_from_ijk(truth["high_k_blocks_ijk"], nx, ny, nz)
    srv = _mask_from_ijk(truth.get("srv_blocks_ijk", []), nx, ny, nz)
    matrix = ~(frac | srv)

    rec: dict = {
        "case": case,
        "dir": CASE_DIR[case],
        "out_exists": out_path.is_file(),
        "n_frac_blocks": int(truth["n_frac_blocks"]),
        "n_srv_blocks": int(truth["n_srv_blocks"]),
        "n_wells": len(truth["wells"]),
        "frac_i_planes": truth["frac_i_planes"],
    }
    if not out_path.is_file():
        rec["ok"] = False
        rec["reason"] = "missing .out"
        return rec

    text = out_path.read_text(encoding="latin-1", errors="replace")
    rec["normal_termination"] = "Normal Termination" in text
    rec["has_shutin_keyword"] = "*SHUTIN" in text or "SHUTIN" in text
    rec["has_open_keyword"] = "*OPEN" in text or " OPEN" in text

    sw = parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)
    pr = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    rec["n_sw_maps"] = len(sw)
    rec["n_p_maps"] = len(pr)
    if not sw or not pr:
        rec["ok"] = False
        rec["reason"] = "no parsed maps"
        return rec

    t0, p0 = pr[0]
    t1, p1 = pr[-1]
    _, sw0 = sw[0]
    _, sw1 = sw[-1]
    rec["t_first_days"] = float(t0)
    rec["t_last_days"] = float(t1)
    rec["p_finite_frac"] = float(np.mean(np.isfinite(p1)))
    rec["sw_finite_frac"] = float(np.mean(np.isfinite(sw1)))
    rec["p_min_psi"] = float(np.nanmin(p1))
    rec["p_max_psi"] = float(np.nanmax(p1))
    rec["p_std_psi"] = float(np.nanstd(p1))
    rec["sw_min"] = float(np.nanmin(sw1))
    rec["sw_max"] = float(np.nanmax(sw1))
    rec["sw_std"] = float(np.nanstd(sw1))

    p_frac = _region_stats(p1, frac)
    p_mat = _region_stats(p1, matrix)
    sw_frac = _region_stats(sw1, frac)
    sw_mat = _region_stats(sw1, matrix)
    rec["p_frac_mean_psi"] = p_frac["mean"]
    rec["p_matrix_mean_psi"] = p_mat["mean"]
    rec["dp_matrix_minus_frac_psi"] = float(p_mat["mean"] - p_frac["mean"])
    rec["sw_frac_mean"] = sw_frac["mean"]
    rec["sw_matrix_mean"] = sw_mat["mean"]
    rec["dsw_abs"] = float(abs(sw_frac["mean"] - sw_mat["mean"]))

    dp = p1 - p0
    rec["dp_frac_mean_psi"] = float(np.nanmean(dp[frac]))
    rec["dp_matrix_mean_psi"] = float(np.nanmean(dp[matrix]))
    rec["dsw_frac_mean"] = float(np.nanmean(sw1[frac] - sw0[frac]))
    rec["dsw_matrix_mean"] = float(np.nanmean(sw1[matrix] - sw0[matrix]))

    rec["ok"] = bool(
        rec["normal_termination"]
        and rec["p_finite_frac"] > 0.99
        and rec["sw_finite_frac"] > 0.99
        and rec["p_std_psi"] > 1.0
        and rec["dp_matrix_minus_frac_psi"] > 10.0
        and rec["n_sw_maps"] >= 2
        and rec["n_p_maps"] >= 2
    )
    return rec


def main() -> int:
    rows = [smoke_one(c) for c in CASE_DIR]
    report = {
        "analog_note": (
            "IMEX single-porosity black-oil analog of shale-oil depletion. "
            "Not GEM, not dual-porosity, not adsorption."
        ),
        "grid": "21 x 31 x 5 CART",
        "cases": rows,
        "all_ok": all(r.get("ok") for r in rows),
    }
    out = HERE / "smoke_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
