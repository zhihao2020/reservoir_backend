"""Scheme-to-scheme FIM vs sequential ladder (dead oil / no-lib / liberation).

Does not compare to CMG. Isolates which physics layer first diverges.
"""

from __future__ import annotations

import copy
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

from reservoir_backend.domain.types import Experiment
from reservoir_backend.inverse.parameterization import ContrastParameterization
from reservoir_backend.physics.pvt import PSI, BlackOilPVT
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec

_spec = importlib.util.spec_from_file_location("cmg_lab_layers_invert", LAYERS / "run_invert_eval.py")
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {LAYERS / 'run_invert_eval.py'}")
_ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ll)

DAY_S = _ll.DAY_S
_grid = _ll._grid
_physics = _ll._physics
_ports = _ll._ports
_same_cmg_controls = _ll._same_cmg_controls

TRUTH = HERE / "truth_fault_channel_lib.json"
REPORT = HERE / "fim_ladder.json"


def _ijk(grid, flat: int) -> tuple[int, int, int]:
    i = int(flat) % int(grid.nx)
    j = (int(flat) // int(grid.nx)) % int(grid.ny)
    k = int(flat) // (int(grid.nx) * int(grid.ny))
    return i, j, k


def _run_one(*, truth: dict, grid, rid, face, k, phys, t_end: float, fim: bool):
    times = np.array([t_end])
    param = ContrastParameterization(rid, phi=float(truth["controls"]["phi"]))
    inj, prod = _ports(grid, truth=truth)
    phys = copy.deepcopy(phys)
    phys.fully_implicit = bool(fim)
    twin = DigitalTwin(
        grid,
        Experiment(size_m=grid.size_m(), sensors=[], controls=_same_cmg_controls(truth, times), observations=[]),
        [inj, prod],
        phys,
        param,
        face_mult_x=face,
        inverse=InverseSpec(max_iter=2),
    )
    t0 = time.perf_counter()
    traj = twin.simulate(twin.rock_from_k(k), t_end=t_end, report_times=times)
    elapsed = time.perf_counter() - t0
    st = traj.states[-1]
    sg = np.zeros(grid.n_cells) if st.sg is None else np.asarray(st.sg, dtype=float)
    newt = 0
    for ex in getattr(traj, "extras", []) or []:
        if isinstance(ex, dict) and "newton_its" in ex:
            newt += int(ex["newton_its"])
    return {
        "elapsed_s": float(elapsed),
        "p": np.asarray(st.pressure, dtype=float),
        "sw": np.asarray(st.sw, dtype=float),
        "sg": sg,
        "pmin_psi": float(np.min(st.pressure) / PSI),
        "pmean_psi": float(np.mean(st.pressure) / PSI),
        "sg_mean": float(np.mean(sg)),
        "newton_its": int(newt),
    }


def _compare(grid, seq, fim) -> dict:
    dp = (fim["p"] - seq["p"]) / PSI
    dsw = fim["sw"] - seq["sw"]
    dsg = fim["sg"] - seq["sg"]
    imax = int(np.argmax(np.abs(dp)))
    i, j, k = _ijk(grid, imax)
    return {
        "dp_rmse_psi": float(np.sqrt(np.mean(dp * dp))),
        "dp_max_psi": float(np.max(np.abs(dp))),
        "dp_max_ijk": [i, j, k],
        "dsw_max": float(np.max(np.abs(dsw))),
        "dsg_max": float(np.max(np.abs(dsg))),
        "seq_pmin_psi": seq["pmin_psi"],
        "fim_pmin_psi": fim["pmin_psi"],
        "seq_sg_mean": seq["sg_mean"],
        "fim_sg_mean": fim["sg_mean"],
        "seq_s": seq["elapsed_s"],
        "fim_s": fim["elapsed_s"],
        "fim_newton_its": fim["newton_its"],
    }


def _rung0_dead_oil(truth, p_init: float):
    """Dead oil but keep three_phase tables (FIM branch needs three_phase)."""
    phys = _physics(
        p_init=p_init,
        sw_init=float(truth["controls"]["swi"]),
        sg_init=float(truth["controls"].get("sgi", 0.0)),
        three_phase=True,
    )
    phys.pvt = BlackOilPVT.slightly_compressible(1.0e-9, pref=p_init)
    assert not phys.pvt.has_live_oil()
    return phys


def _rung_live(truth, p_init: float, *, prod_bhp_psi: float):
    t = copy.deepcopy(truth)
    t["controls"] = dict(t["controls"])
    t["controls"]["prod_bhp_psi"] = float(prod_bhp_psi)
    phys = _physics(
        p_init=p_init,
        sw_init=float(t["controls"]["swi"]),
        sg_init=float(t["controls"].get("sgi", 0.0)),
        three_phase=True,
    )
    return t, phys


def main() -> int:
    truth0 = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth0)
    rid = np.load(HERE / "region_id.npy")
    face = np.load(HERE / "face_mult_x.npy")
    k = np.where(rid == 1, float(truth0["channel"]["k_hi_m2"]), float(truth0["channel"]["k_lo_m2"]))
    t_end = 1.0 * DAY_S
    p_init = float(truth0["controls"]["pres_psi"]) * PSI
    pb = float(truth0["controls"]["pb_psi"])

    rungs = []
    # Rung 0: dead oil
    phys0 = _rung0_dead_oil(truth0, p_init)
    seq0 = _run_one(truth=truth0, grid=grid, rid=rid, face=face, k=k, phys=phys0, t_end=t_end, fim=False)
    fim0 = _run_one(truth=truth0, grid=grid, rid=rid, face=face, k=k, phys=phys0, t_end=t_end, fim=True)
    c0 = _compare(grid, seq0, fim0)
    c0["name"] = "rung0_dead_oil"
    c0["note"] = "slightly_compressible PVT; three_phase kept"
    rungs.append(c0)
    print(
        f"rung0_dead_oil  dp_rmse={c0['dp_rmse_psi']:.3f} psi  dp_max={c0['dp_max_psi']:.3f} @ijk={c0['dp_max_ijk']}  "
        f"dsw_max={c0['dsw_max']:.4f}  dsg_max={c0['dsg_max']:.4f}  "
        f"pmin seq/fim={c0['seq_pmin_psi']:.0f}/{c0['fim_pmin_psi']:.0f}",
        flush=True,
    )

    # Rung 1: live oil, prod BHP above pb (no liberation)
    t1, phys1 = _rung_live(truth0, p_init, prod_bhp_psi=max(3000.0, pb + 500.0))
    seq1 = _run_one(truth=t1, grid=grid, rid=rid, face=face, k=k, phys=phys1, t_end=t_end, fim=False)
    fim1 = _run_one(truth=t1, grid=grid, rid=rid, face=face, k=k, phys=phys1, t_end=t_end, fim=True)
    c1 = _compare(grid, seq1, fim1)
    c1["name"] = "rung1_live_oil_above_pb"
    c1["note"] = f"prod_bhp={t1['controls']['prod_bhp_psi']} psi > pb={pb}"
    rungs.append(c1)
    print(
        f"rung1_no_lib    dp_rmse={c1['dp_rmse_psi']:.3f} psi  dp_max={c1['dp_max_psi']:.3f} @ijk={c1['dp_max_ijk']}  "
        f"dsw_max={c1['dsw_max']:.4f}  dsg_max={c1['dsg_max']:.4f}  "
        f"pmin seq/fim={c1['seq_pmin_psi']:.0f}/{c1['fim_pmin_psi']:.0f}  "
        f"sg seq/fim={c1['seq_sg_mean']:.4f}/{c1['fim_sg_mean']:.4f}",
        flush=True,
    )

    # Rung 2: liberation (deck BHP)
    t2, phys2 = _rung_live(truth0, p_init, prod_bhp_psi=float(truth0["controls"]["prod_bhp_psi"]))
    seq2 = _run_one(truth=t2, grid=grid, rid=rid, face=face, k=k, phys=phys2, t_end=t_end, fim=False)
    fim2 = _run_one(truth=t2, grid=grid, rid=rid, face=face, k=k, phys=phys2, t_end=t_end, fim=True)
    c2 = _compare(grid, seq2, fim2)
    c2["name"] = "rung2_liberation"
    c2["note"] = f"prod_bhp={t2['controls']['prod_bhp_psi']} psi < pb={pb}"
    rungs.append(c2)
    print(
        f"rung2_liberate  dp_rmse={c2['dp_rmse_psi']:.3f} psi  dp_max={c2['dp_max_psi']:.3f} @ijk={c2['dp_max_ijk']}  "
        f"dsw_max={c2['dsw_max']:.4f}  dsg_max={c2['dsg_max']:.4f}  "
        f"pmin seq/fim={c2['seq_pmin_psi']:.0f}/{c2['fim_pmin_psi']:.0f}  "
        f"sg seq/fim={c2['seq_sg_mean']:.4f}/{c2['fim_sg_mean']:.4f}",
        flush=True,
    )

    # First diverging rung (threshold: pressure RMSE > 1 psi or max > 3 psi)
    first = None
    for c in rungs:
        if c["dp_rmse_psi"] > 1.0 or c["dp_max_psi"] > 3.0 or c["dsg_max"] > 5.0e-3:
            first = c["name"]
            break
    payload = {
        "t_end_day": 1.0,
        "first_divergence": first,
        "threshold": {"dp_rmse_psi": 1.0, "dp_max_psi": 3.0, "dsg_max": 5.0e-3},
        "rungs": rungs,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"first_divergence={first}  wrote {REPORT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
