"""Sample concecpt sensors from an IMEX .out: one xyz, one kind."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VAL = HERE.parent
for p in (ROOT, VAL, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cmg_io.grid_parse import parse_grid_series, psi_to_pa
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.observation.operator import _trilinear

TRUTH = HERE / "truth_concept.json"
REPORT = HERE / "probe_timeseries.json"
FT_TO_M = 0.3048


def _grid(truth: dict) -> CartesianGrid:
    g = truth["grid"]
    dx = float(g.get("dx_m", float(g["di_ft"]) * FT_TO_M))
    n = int(g["nx"])
    return CartesianGrid(
        nx=n,
        ny=int(g["ny"]),
        nz=int(g["nz"]),
        dx=np.full(n, dx),
        dy=np.full(int(g["ny"]), dx),
        dz=np.full(int(g["nz"]), dx),
    )


def _cmg_to_our(arr: np.ndarray) -> np.ndarray:
    """CMG K=1 is top; product k=0 is bottom."""
    return np.asarray(arr, dtype=float)[::-1]


def _nearest_time(keys: list[float], t: float) -> float:
    return float(min(keys, key=lambda k: abs(k - t)))


def extract(out_path: Path, truth: dict) -> dict:
    grid = _grid(truth)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    pres = {t: _cmg_to_our(a) for t, a in parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)}
    sw = {t: _cmg_to_our(a) for t, a in parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)}
    if not pres or not sw:
        raise RuntimeError(f"missing grid series in {out_path}")
    times = [float(t) for t in truth["times_day"]]
    series = []
    seen: set[tuple[int, int, int]] = set()
    for s in truth["sensors"]:
        key = (round(s["x_m"] * 1e9), round(s["y_m"] * 1e9), round(s["z_m"] * 1e9))
        if key in seen:
            raise ValueError(f"duplicate xyz {s['name']}")
        seen.add(key)
        kind = s["kind"]
        field_by_t = pres if kind == "pressure" else sw
        samples = []
        for t_req in times:
            t = _nearest_time(list(field_by_t), t_req)
            flat = field_by_t[t].ravel()
            val = _trilinear(grid, flat, float(s["x_m"]), float(s["y_m"]), float(s["z_m"]))
            if kind == "pressure":
                val = psi_to_pa(val)
            samples.append({"time_day": t, "time_day_requested": t_req, "value": val})
        series.append(
            {
                "name": s["name"],
                "kind": kind,
                "source": s.get("source"),
                "x_m": s["x_m"],
                "y_m": s["y_m"],
                "z_m": s["z_m"],
                "unit": "Pa" if kind == "pressure" else "fraction",
                "samples": samples,
            }
        )
    return {
        "out": str(out_path),
        "n_series": len(series),
        "kinds": {
            "pressure": sum(1 for s in series if s["kind"] == "pressure"),
            "saturation": sum(1 for s in series if s["kind"] == "saturation"),
        },
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one-kind-per-xyz probe series from IMEX .out")
    parser.add_argument("--out", type=Path, help="IMEX .out path")
    parser.add_argument("--truth", type=Path, default=TRUTH)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    out_path = args.out
    if out_path is None:
        candidates = [
            HERE / "mxspr006_concept.out",
            Path(r"D:\Tool\CMG\_cmg_suite_runs\concept_lab\case_clone\mxspr006_concept.out"),
        ]
        out_path = next((p for p in candidates if p.is_file()), None)
        if out_path is None:
            raise FileNotFoundError("pass --out; no default .out found")
    report = extract(out_path, truth)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("out", "n_series", "kinds")}, indent=2))


if __name__ == "__main__":
    main()
