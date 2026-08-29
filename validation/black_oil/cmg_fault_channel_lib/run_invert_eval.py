"""One invert on the live-oil liberation ruler. Truth vs posterior."""

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

from cmg_io.grid_parse import parse_grid_series, psi_to_pa
from reservoir_backend.domain.types import Experiment, ObservationSeries, Sensor, State
from reservoir_backend.inverse.parameterization import ContrastParameterization
from reservoir_backend.observation.operator import ObservationOperator
from reservoir_backend.exceptions import TimeStepUnderflow
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec

_spec = importlib.util.spec_from_file_location("cmg_lab_layers_invert", LAYERS / "run_invert_eval.py")
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {LAYERS / 'run_invert_eval.py'}")
_ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ll)
DAY_S = _ll.DAY_S
MD_TO_M2 = _ll.MD_TO_M2
_cmg_to_our = _ll._cmg_to_our
_grid = _ll._grid
_nearest = _ll._nearest
_physics = _ll._physics
_ports = _ll._ports
_sample_gauges = _ll._sample_gauges
_same_cmg_controls = _ll._same_cmg_controls
_score_k = _ll._score_k

OUT = HERE / "fault_channel_lib.out"
TRUTH = HERE / "truth_fault_channel_lib.json"
REPORT = HERE / "invert_eval_report.json"
REPORT_FIM = HERE / "invert_eval_report_fim.json"


def _face_mult() -> np.ndarray:
    return np.load(HERE / "face_mult_x.npy")


def _rid() -> np.ndarray:
    return np.load(HERE / "region_id.npy")


def _k_true(truth: dict) -> np.ndarray:
    rid = _rid()
    return np.where(rid == 1, float(truth["channel"]["k_hi_m2"]), float(truth["channel"]["k_lo_m2"]))


def channel_sensors(grid, *, p_sigma: float, s_sigma: float) -> tuple[list[Sensor], set[str]]:
    """Gauges in channel, matrix, and both sides of the fault."""
    cx, cy, cz = [], [], []
    for i, j, k in ((1, 3, 2), (1, 1, 2), (4, 3, 3), (7, 6, 3), (10, 6, 2), (10, 2, 2)):
        c = grid.index(i, j, k)
        x, y, z = grid.cell_centers()[c]
        cx.append(x)
        cy.append(y)
        cz.append(z)
    names = ("Pch_w", "Pmx_w", "Pmid_w", "Pmid_e", "Pch_e", "Pmx_e")
    sensors = [
        Sensor(n, "pressure", float(x), float(y), float(z), sigma=p_sigma)
        for n, x, y, z in zip(names, cx, cy, cz)
    ]
    z_lo, z_hi = 0.22 * grid.size_m()[2], 0.78 * grid.size_m()[2]
    sensors.append(Sensor("Sch_w", "saturation", cx[0], cy[0], z_hi, sigma=s_sigma))
    sensors.append(Sensor("Smx_w", "saturation", cx[1], cy[1], z_lo, sigma=s_sigma))
    sensors.append(Sensor("Sch_m", "saturation", cx[2], cy[2], z_hi, sigma=s_sigma))
    sensors.append(Sensor("Sch_e", "saturation", cx[4], cy[4], z_hi, sigma=s_sigma))
    sensors.append(Sensor("Smx_e", "saturation", cx[5], cy[5], z_lo, sigma=s_sigma))
    return sensors, {"Pch_e", "Sch_e"}


def gas_sensors(grid, *, p_sigma: float, s_sigma: float, g_sigma: float):
    sensors, hold = channel_sensors(grid, p_sigma=p_sigma, s_sigma=s_sigma)
    z_hi = 0.78 * grid.size_m()[2]
    pch = next(s for s in sensors if s.name == "Pch_w")
    pche = next(s for s in sensors if s.name == "Pch_e")
    pmx = next(s for s in sensors if s.name == "Pmx_w")
    sensors.append(Sensor("Gch_w", "gas_saturation", pch.x, pch.y, z_hi, sigma=g_sigma))
    sensors.append(Sensor("Gmx_w", "gas_saturation", pmx.x, pmx.y, z_hi, sigma=g_sigma))
    sensors.append(Sensor("Gch_e", "gas_saturation", pche.x, pche.y, z_hi, sigma=g_sigma))
    hold = set(hold) | {"Gch_e"}
    return sensors, hold


def _twin(truth, grid, param, sensors, times, *, history_end_s,
          fully_implicit: bool = False, n_ensemble: int = 10,
          n_assimilations: int = 3, time_limit_s: float = 1500.0):
    inj, prod = _ports(grid, truth=truth)
    experiment = Experiment(
        size_m=grid.size_m(),
        sensors=sensors,
        controls=_same_cmg_controls(truth, times),
        observations=[],
        history_end_s=float(history_end_s),
    )
    phys = _physics(
        p_init=psi_to_pa(float(truth["controls"]["pres_psi"])),
        sw_init=float(truth["controls"]["swi"]),
        sg_init=float(truth["controls"].get("sgi", 0.0)),
        three_phase=True,
    )
    # --fully-implicit forces FIM; PhysicsSpec default is now True.
    if fully_implicit:
        phys.fully_implicit = True
    return DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        phys,
        param,
        face_mult_x=_face_mult(),
        inverse=InverseSpec(
            prior_mean=np.log(50.0 * MD_TO_M2),
            prior_std=0.30,
            max_iter=int(n_assimilations),
            time_limit_s=float(time_limit_s),
        ),
    )

def _traj_step_stats(traj) -> dict:
    reps = list(getattr(traj, "reports", None) or [])
    dts = [float(r.dt) for r in reps]
    its = [getattr(r, "newton_its", None) for r in reps]
    its_n = [int(x) for x in its if x is not None]
    return {
        "nsteps": len(dts),
        "dt_min": (min(dts) if dts else None),
        "dt_mean": (float(np.mean(dts)) if dts else None),
        "dt_max": (max(dts) if dts else None),
        "newton_min": (min(its_n) if its_n else None),
        "newton_mean": (float(np.mean(its_n)) if its_n else None),
        "newton_max": (max(its_n) if its_n else None),
        "dt_history": [round(d, 6) for d in dts],
        "newton_history": its,
    }


def invert(truth, grid, *, fully_implicit: bool = False, n_ensemble: int = 10, n_assimilations: int = 3, time_limit_s: float = 1500.0) -> dict:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    sg_series = parse_grid_series(OUT, field="sg", nx=nx, ny=ny, nz=nz)
    if not p_series or not sw_series or not sg_series:
        raise SystemExit(f"missing CMG p/Sw/Sg maps in {OUT}")
    want = [0.25, 0.50, 1.00, 2.00]
    p_t = np.array([t for t, _ in p_series])
    days = [t for t in want if np.min(np.abs(p_t - t)) < 0.08]
    times = np.asarray(days, dtype=float) * DAY_S
    hist_end = float(1.00 * DAY_S)
    p_map = {t: (_cmg_to_our(a) * 6894.757293168).ravel() for t, a in p_series}
    sw_map = {t: _cmg_to_our(a).ravel() for t, a in sw_series}
    sg_map = {t: _cmg_to_our(a).ravel() for t, a in sg_series}

    def pick(store, t):
        ks = np.array(list(store.keys()), dtype=float)
        return store[_nearest(ks, t)]

    k_lo = float(truth["channel"]["k_lo_m2"])
    k_hi = float(truth["channel"]["k_hi_m2"])
    k_true = _k_true(truth)
    rid = _rid()
    param = ContrastParameterization(
        rid,
        phi=float(truth["controls"]["phi"]),
        log_contrast_mean=float(np.log(40.0)),
        log_contrast_std=0.35,
    )
    # Sw extras ~0.04 are F≠IMEX, not K. Assimilate P+Sg so R does not swallow contrast.
    sensors, hold = gas_sensors(grid, p_sigma=psi_to_pa(6.0), s_sigma=0.03, g_sigma=0.015)
    sensors = [s for s in sensors if s.kind != "saturation"]
    hold = {n for n in hold if n.startswith("P") or n.startswith("G")}
    op = ObservationOperator(grid, sensors)
    rng = np.random.default_rng(8)
    noisy, clean = [], []
    cmg_clean: dict[str, list[float]] = {}
    for s in sensors:
        vals = []
        for td in days:
            st = State(pressure=pick(p_map, td), sw=pick(sw_map, td), sg=pick(sg_map, td))
            vals.append(op.sample(s, st))
        cmg_clean[s.name] = [float(v) for v in vals]
        va = np.asarray(vals, dtype=float)
        clean.append(ObservationSeries(s.name, s.kind, times, va, np.full(len(vals), s.sigma), s.name in hold))
        noisy.append(
            ObservationSeries(
                s.name,
                s.kind,
                times,
                va + rng.normal(0.0, 0.35 * s.sigma, size=len(vals)),
                np.full(len(vals), s.sigma),
                s.name in hold,
            )
        )
    twin = _twin(
        truth, grid, param, sensors, times, history_end_s=hist_end,
        fully_implicit=fully_implicit, n_ensemble=n_ensemble,
        n_assimilations=n_assimilations, time_limit_s=time_limit_s,
    )
    twin.experiment.observations = noisy
    default_fim = bool(PhysicsSpec().fully_implicit)
    print(
        f"   fully_implicit={bool(twin.physics.fully_implicit)}  "
        f"default_fully_implicit={default_fim}  "
        f"max_iter={twin.inverse.max_iter}  "
        f"max_steps={twin.physics.max_steps}  hist_end_d={hist_end / DAY_S:.2f}",
        flush=True,
    )
    # Gate passed: default may be True. --fully-implicit still forces FIM on.
    fim_stats = {
        "fully_implicit": bool(twin.physics.fully_implicit),
        "default_fully_implicit": default_fim,
        "max_steps": int(twin.physics.max_steps),
        "max_iter": int(twin.inverse.max_iter),
        "history_end_s": float(hist_end),
        "TimeStepUnderflow": False,
        "underflow_where": None,
        "underflow_msg": None,
        "nsteps": None,
        "dt_min": None,
        "dt_mean": None,
        "dt_max": None,
        "newton_min": None,
        "newton_mean": None,
        "newton_max": None,
        "dt_history": [],
        "newton_history": [],
    }
    _orig_sim = twin.simulate

    def _sim_capture(*args, **kwargs):
        try:
            traj = _orig_sim(*args, **kwargs)
        except TimeStepUnderflow as exc:
            fim_stats["TimeStepUnderflow"] = True
            if fim_stats["underflow_where"] is None:
                fim_stats["underflow_where"] = "simulate"
                fim_stats["underflow_msg"] = str(exc)
            raise
        if fim_stats["nsteps"] is None:
            fim_stats.update(_traj_step_stats(traj))
        return traj

    twin.simulate = _sim_capture  # type: ignore[method-assign]

    def _fail(where: str, exc: Exception, elapsed: float) -> dict:
        print(f"   FAIL TimeStepUnderflow during {where}: {exc}", flush=True)
        fim_stats["TimeStepUnderflow"] = True
        fim_stats["underflow_where"] = where
        fim_stats["underflow_msg"] = str(exc)
        print(
            f"   default_fully_implicit={PhysicsSpec().fully_implicit}  "
            f"nsteps={fim_stats.get('nsteps')}  elapsed={elapsed:.1f}s",
            flush=True,
        )
        return {
            "pass": False,
            "TimeStepUnderflow": True,
            "fim_stats": fim_stats,
            "elapsed_s": elapsed,
            "times_day": days,
            "n_theta": param.n_params,
            "k_lo_md_post": float("nan"),
            "k_hi_md_post": float("nan"),
            "k_contrast_post": float("nan"),
            "logk_rmse_post": float("nan"),
            "holdout_nrmse": float("nan"),
        }

    try:
        extras = twin.inflate_observations(
            twin.rock_from_k(k_true), clean=clean, history_end_s=hist_end, extra_cap_mult=1.0
        )
    except TimeStepUnderflow as exc:
        return _fail("inflate_observations", exc, 0.0)
    print(f"   extras={ {k: round(v, 3) if v < 10 else round(v) for k, v in extras.items()} }", flush=True)
    print("invert CMG gauges (liberation, known channel) ...", flush=True)
    t0 = time.perf_counter()
    try:
        post = twin.calibrate()
    except TimeStepUnderflow as exc:
        return _fail("calibrate", exc, time.perf_counter() - t0)
    k_name = "k_post_fim.npy" if fully_implicit else "k_post.npy"
    np.save(HERE / k_name, post.k)
    try:
        fc = twin.forecast(post)
        forecast_nrmse = twin.score_forecast(fc)
    except Exception as exc:
        print(f"   forecast skipped: {exc}", flush=True)
        fc = post.history
        forecast_nrmse = float("nan")
    out = _score_k(k_true, rid, post, k_lo, k_hi)
    out["forecast_nrmse"] = forecast_nrmse
    out["elapsed_s"] = time.perf_counter() - t0
    out["times_day"] = days
    out["gauges_truth"] = {"times_day": days, "series": cmg_clean}
    out["gauges_post"] = _sample_gauges(twin, sensors, fc)
    out["holdout"] = sorted(hold)
    out["model_error_extras"] = extras
    out["n_theta"] = param.n_params
    out["fully_implicit"] = bool(twin.physics.fully_implicit)
    out["default_fully_implicit"] = bool(PhysicsSpec().fully_implicit)
    out["TimeStepUnderflow"] = bool(fim_stats["TimeStepUnderflow"])
    out["fim_stats"] = fim_stats
    out["pass"] = bool(
        out["k_contrast_post"] > 8.0 and out["logk_rmse_post"] < out["logk_rmse_prior"]
    )
    print(
        f"   K {out['k_lo_md_post']:.0f}/{out['k_hi_md_post']:.0f}  "
        f"contrast {out['k_contrast_post']:.2f}  logk {out['logk_rmse_post']:.3f}  "
        f"hold {out['holdout_nrmse']:.3f}  pass={out['pass']}",
        flush=True,
    )
    print(
        f"   FIM_STEPS nsteps={fim_stats.get('nsteps')}  "
        f"dt_min/mean/max={fim_stats.get('dt_min')}/{fim_stats.get('dt_mean')}/{fim_stats.get('dt_max')}  "
        f"newton_min/mean/max={fim_stats.get('newton_min')}/{fim_stats.get('newton_mean')}/{fim_stats.get('newton_max')}  "
        f"underflow={fim_stats['TimeStepUnderflow']}  default_fully_implicit={out['default_fully_implicit']}",
        flush=True,
    )
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Liberation-ruler product invert")
    ap.add_argument(
        "--fully-implicit",
        action="store_true",
        help="run this invert with FIM (PhysicsSpec default is True)",
    )
    ap.add_argument("--n-ensemble", type=int, default=10)
    ap.add_argument("--n-assimilations", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=1500.0)
    args = ap.parse_args()
    if not OUT.is_file():
        raise SystemExit(f"missing {OUT}; run IMEX on fault_channel_lib.dat")
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth)
    print(
        f"product invert  grid={grid.nx}x{grid.ny}x{grid.nz}  "
        f"fully_implicit={args.fully_implicit}  default={PhysicsSpec().fully_implicit}  "
        f"ne={args.n_ensemble}  na={args.n_assimilations}  time_limit={args.time_limit}",
        flush=True,
    )
    inv = invert(
        truth,
        grid,
        fully_implicit=bool(args.fully_implicit),
        n_ensemble=int(args.n_ensemble),
        n_assimilations=int(args.n_assimilations),
        time_limit_s=float(args.time_limit),
    )
    report_path = REPORT_FIM if args.fully_implicit else REPORT
    report_path.write_text(
        json.dumps(
            {
                "philosophy": "CMG virtual experiment with live-oil liberation; one invert",
                "pvt": "cmg_seawater",
                "fully_implicit": bool(args.fully_implicit),
                "default_fully_implicit": bool(PhysicsSpec().fully_implicit),
                "invert": inv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "k_lo_md": inv.get("k_lo_md_post"),
                "k_hi_md": inv.get("k_hi_md_post"),
                "contrast": inv.get("k_contrast_post"),
                "logk": inv.get("logk_rmse_post"),
                "hold": inv.get("holdout_nrmse"),
                "pass": inv.get("pass"),
                "fully_implicit": bool(args.fully_implicit),
                "default_fully_implicit": bool(PhysicsSpec().fully_implicit),
                "TimeStepUnderflow": bool(inv.get("TimeStepUnderflow", False)),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    print(f"confirm default_fully_implicit={PhysicsSpec().fully_implicit}", flush=True)
    if args.fully_implicit:
        return 2 if inv.get("TimeStepUnderflow") else 0
    return 0 if inv["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
