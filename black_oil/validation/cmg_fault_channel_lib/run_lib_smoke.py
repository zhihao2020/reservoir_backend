"""Same-rock F(K) vs IMEX on the liberation ruler."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAYERS = ROOT / "black_oil" / "validation" / "cmg_lab_layers"
VAL = ROOT / "black_oil" / "validation"
for p in (ROOT, VAL, LAYERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importlib.util

from cmg_io.grid_parse import parse_grid_series
from reservoir_backend.domain.types import Experiment
from reservoir_backend.inverse.parameterization import ContrastParameterization
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec

_spec = importlib.util.spec_from_file_location("cmg_lab_layers_invert", LAYERS / "run_invert_eval.py")
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {LAYERS / 'run_invert_eval.py'}")
_ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ll)

DAY_S = _ll.DAY_S
_cmg_to_our = _ll._cmg_to_our
_grid = _ll._grid
_nearest = _ll._nearest
_physics = _ll._physics
_ports = _ll._ports
_same_cmg_controls = _ll._same_cmg_controls

OUT = HERE / "fault_channel_lib.out"
TRUTH = HERE / "truth_fault_channel_lib.json"
PSI = 6894.757293168


def _rmse(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def main() -> int:
    if not OUT.is_file():
        raise SystemExit(f"missing {OUT}; run IMEX on fault_channel_lib.dat")
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth)
    rid = np.load(HERE / "region_id.npy")
    face = np.load(HERE / "face_mult_x.npy")
    k = np.where(rid == 1, float(truth["channel"]["k_hi_m2"]), float(truth["channel"]["k_lo_m2"]))
    t_end = 1.0 * DAY_S
    times = np.array([t_end])
    param = ContrastParameterization(rid, phi=float(truth["controls"]["phi"]))
    inj, prod = _ports(grid, truth=truth)
    fim = any(a in {"--fim", "fim"} for a in sys.argv[1:])
    phys = _physics(
        p_init=float(truth["controls"]["pres_psi"]) * PSI,
        sw_init=float(truth["controls"]["swi"]),
        sg_init=float(truth["controls"].get("sgi", 0.0)),
        three_phase=True,
    )
    phys.fully_implicit = bool(fim)
    twin = DigitalTwin(
        grid,
        Experiment(size_m=grid.size_m(), sensors=[], controls=_same_cmg_controls(truth, times), observations=[]),
        [inj, prod],
        phys,
        param,
        face_mult_x=face,
        inverse=InverseSpec(n_ensemble=2, n_assimilations=1),
    )
    t0 = time.perf_counter()
    traj = twin.simulate(twin.rock_from_k(k), t_end=t_end, report_times=times)
    elapsed = time.perf_counter() - t0
    st = traj.states[-1]
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p_s = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_s = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    sg_s = parse_grid_series(OUT, field="sg", nx=nx, ny=ny, nz=nz)
    if not p_s or not sw_s or not sg_s:
        raise SystemExit(f"missing CMG p/Sw/Sg maps in {OUT}")
    tkey = _nearest(np.array([t for t, _ in p_s]), 1.0)
    p_cmg = _cmg_to_our(dict(p_s)[tkey]).ravel()
    sw_cmg = _cmg_to_our(dict(sw_s)[tkey]).ravel()
    sg_cmg = _cmg_to_our(dict(sg_s)[tkey]).ravel()
    sg_f = np.zeros(grid.n_cells) if st.sg is None else np.asarray(st.sg, dtype=float)
    payload = {
        "elapsed_s": elapsed,
        "prod_bhp_psi": float(truth["controls"]["prod_bhp_psi"]),
        "pb_psi": float(truth["controls"]["pb_psi"]),
        "sg_init": float(truth["controls"]["sgi"]),
        "f_p_min_psi": float(np.min(st.pressure) / PSI),
        "f_p_mean_psi": float(np.mean(st.pressure) / PSI),
        "f_sg_mean": float(np.mean(sg_f)),
        "f_sg_max": float(np.max(sg_f)),
        "cmg_p_min_psi": float(np.min(p_cmg)),
        "cmg_p_mean_psi": float(np.mean(p_cmg)),
        "cmg_sg_mean": float(np.mean(sg_cmg)),
        "cmg_sg_max": float(np.max(sg_cmg)),
        "p_rmse_psi": _rmse(st.pressure / PSI, p_cmg),
        "sw_rmse": _rmse(st.sw, sw_cmg),
        "sg_rmse": _rmse(sg_f, sg_cmg),
        "so_rmse": _rmse(st.so(), 1.0 - sw_cmg - sg_cmg),
    }
    report = HERE / ("lib_smoke_fim.json" if fim else "lib_smoke.json")
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tag = "fim" if fim else "lib"
    print(
        f"{tag}  {elapsed:.2f}s  1.00d p={payload['p_rmse_psi']:.2f} psi  "
        f"sw={payload['sw_rmse']:.3f}  sg={payload['sg_rmse']:.3f}  "
        f"F Sg={payload['f_sg_mean']:.4f}  IMEX Sg={payload['cmg_sg_mean']:.4f}  "
        f"F pmin={payload['f_p_min_psi']:.0f}  IMEX pmin={payload['cmg_p_min_psi']:.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
