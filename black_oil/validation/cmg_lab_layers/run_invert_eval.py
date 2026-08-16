"""Inversion ruler: same CMG *operating conditions*, our lab F, not a CMG clone.

Digital twin (not an open-source reservoir clone)
-------------------------------------------------
Open-source engines (OPM, MRST, …) are forward models F. This product is

    x = F_lab(m, u),   d_sim = H(x),   d_obs → m_post

Matching a CMG *case* means matching the experiment:
  u(t)  — same well controls (here a BHP pair so Δp stays in the data)
  H     — gauges at real (x, y, z); depths do not have to sit on one plane
  d_obs — what those gauges would read

It does **not** mean F_lab ≡ IMEX. Fitting K so that a cloned F recovers
K_CMG is history-matching / 调参, not a lab or production twin.

Protocols
---------
A. Self-consistent: d = H(F_lab(m_true)). Pass = recover layer K.
   Also compare sparse mid-plane gauges vs multi-depth P+Sw (diverse).
B. Cross-simulator: d from CMG, invert with F_lab. Pass = observation /
   hold-out / forecast fit. Posterior K is *equivalent lab K*, not K_CMG.

B does not assimilate well rates: Peaceman vs FlowPort would otherwise
be absorbed into K (the small-scale version of 调参).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = Path(__file__).resolve().parents[1]
for p in (ROOT, VAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cmg_io.grid_parse import parse_grid_series, parse_surface_rates_m3s, psi_to_pa
from reservoir_backend.domain.types import (
    ControlSeries,
    Experiment,
    ObservationSeries,
    Sensor,
    State,
    column_sensors,
)
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import RegionParameterization
from reservoir_backend.observation.operator import ObservationOperator
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock, log_permeability
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec

HERE = Path(__file__).resolve().parent
OUT = HERE / "lab_layers.out"
TRUTH = HERE / "truth_lab_layers.json"
REPORT = HERE / "invert_eval_report.json"
FT_TO_M = 0.3048
MD_TO_M2 = 9.869233e-16
DAY_S = 86400.0


def _cmg_to_our(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)[::-1]


def _nearest(keys: np.ndarray, t: float) -> float:
    return float(keys[int(np.argmin(np.abs(keys - t)))])


def _grid(truth: dict) -> CartesianGrid:
    g = truth["grid"]
    return CartesianGrid(
        nx=int(g["nx"]),
        ny=int(g["ny"]),
        nz=int(g["nz"]),
        dx=np.full(int(g["nx"]), float(g["di_ft"]) * FT_TO_M),
        dy=np.full(int(g["ny"]), float(g["dj_ft"]) * FT_TO_M),
        dz=np.full(int(g["nz"]), float(g["dk_ft"]) * FT_TO_M),
    )


def _region(grid: CartesianGrid, n_top: int) -> np.ndarray:
    rid = np.zeros(grid.n_cells, dtype=np.int64)
    for c in range(grid.n_cells):
        _i, _j, k = grid.ijk(c)
        if k >= grid.nz - n_top:
            rid[c] = 1
    return rid


def _cmg_k_to_ours(nz: int, k_cmg) -> list[int]:
    """CMG KDIR DOWN, K=1 is top. Our k=0 is the bottom layer."""
    return [int(nz) - int(k) for k in k_cmg]


def _ports(grid: CartesianGrid, *, inj_control: str = "pressure", truth: dict | None = None) -> tuple[FlowPort, FlowPort]:
    """Same open intervals as the IMEX deck.

    INJ is not open on the top two high-K layers; PROD is the full column.
    That asymmetry is in the IMEX *PERF list. Copying it is the experiment,
    not a Peaceman clone. Cell Dirichlet on this *partial* INJ column does
    not pin both layers at both ends, so p still sees K.
    """
    wells = (truth or {}).get("wells") or {}
    inj_spec = wells.get("INJ") or {}
    prod_spec = wells.get("PROD") or {}
    inj_i = int(inj_spec.get("i", 1)) - 1
    inj_j = int(inj_spec.get("j", (grid.ny + 1) // 2)) - 1
    prod_i = int(prod_spec.get("i", grid.nx)) - 1
    prod_j = int(prod_spec.get("j", (grid.ny + 1) // 2)) - 1
    inj_ks = _cmg_k_to_ours(grid.nz, inj_spec.get("k_cmg") or (3, 4, 5, 6))
    prod_ks = _cmg_k_to_ours(grid.nz, prod_spec.get("k_cmg") or (1, 2, 3, 4, 5, 6))
    inj_cells = np.array([grid.index(inj_i, inj_j, k) for k in inj_ks], dtype=np.int64)
    prod_cells = np.array([grid.index(prod_i, prod_j, k) for k in prod_ks], dtype=np.int64)
    inj = FlowPort(
        name="INJ",
        role="injector",
        control=inj_control,
        cell_ids=inj_cells,
        sw_inj=0.80,
    )
    prod = FlowPort(
        name="PROD",
        role="producer",
        control="pressure",
        cell_ids=prod_cells,
    )
    return inj, prod


def _same_cmg_controls(truth: dict, times: np.ndarray) -> list[ControlSeries]:
    """u(t) copied from the IMEX deck: BHP pair + injection composition."""
    p_inj = psi_to_pa(float(truth["controls"]["inj_bhp_psi"]))
    p_prod = psi_to_pa(float(truth["controls"]["prod_bhp_psi"]))
    return [
        ControlSeries("INJ", "pressure", times, np.full(times.size, p_inj)),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.85)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, p_prod)),
    ]


def _depths(grid: CartesianGrid) -> dict[str, float]:
    lz = grid.size_m()[2]
    return {"bot": 0.18 * lz, "mid": 0.50 * lz, "top": 0.82 * lz}


def diverse_sensors(grid: CartesianGrid, *, p_sigma: float, s_sigma: float, with_rate: bool) -> tuple[list[Sensor], set[str]]:
    """Two columns at different x, each with P and Sw at top and bottom."""
    lx, ly, _lz = grid.size_m()
    z = _depths(grid)
    sensors: list[Sensor] = []
    sensors += column_sensors("Pin", "pressure", 0.25 * lx, 0.50 * ly, [z["bot"], z["top"]], sigma=p_sigma, labels=("bot", "top"))
    sensors += column_sensors("Pmid", "pressure", 0.50 * lx, 0.50 * ly, [z["mid"]], sigma=p_sigma, labels=("mid",))
    sensors += column_sensors("Pout", "pressure", 0.75 * lx, 0.50 * ly, [z["bot"], z["top"]], sigma=p_sigma, labels=("bot", "top"))
    sensors += column_sensors("Sin", "saturation", 0.35 * lx, 0.50 * ly, [z["bot"], z["top"]], sigma=s_sigma, labels=("bot", "top"))
    sensors += column_sensors("Sout", "saturation", 0.65 * lx, 0.50 * ly, [z["top"]], sigma=s_sigma, labels=("top",))
    if with_rate:
        sensors.append(
            Sensor(
                "QW",
                "phase_rate",
                (grid.nx - 0.5) * float(grid.dx[0]),
                3.5 * float(grid.dy[0]),
                0.45 * grid.size_m()[2],
                port_name="PROD",
                sigma=5.0e-7,
            )
        )
    hold = {"Pout_top", "Sout_top"}
    return sensors, hold


def sparse_sensors(grid: CartesianGrid, *, p_sigma: float, s_sigma: float) -> tuple[list[Sensor], set[str]]:
    """Single mid-plane P + S. Weak on layered K."""
    lx, ly, _lz = grid.size_m()
    z = _depths(grid)["mid"]
    sensors = [
        Sensor("P_in", "pressure", 0.25 * lx, 0.50 * ly, z, sigma=p_sigma),
        Sensor("P_out", "pressure", 0.75 * lx, 0.50 * ly, z, sigma=p_sigma),
        Sensor("S_mid", "saturation", 0.50 * lx, 0.50 * ly, z, sigma=s_sigma),
    ]
    return sensors, {"P_out"}


def _score_k(k_true, region, post, k_lo, k_hi) -> dict:
    k_post = post.esmda.k_mean
    prior = np.full(k_true.size, 100.0 * MD_TO_M2)
    return {
        "theta_true": [float(np.log(k_lo)), float(np.log(k_hi))],
        "theta_post": post.esmda.theta_mean.tolist(),
        "theta_std": post.esmda.theta_std.tolist(),
        "identifiability": post.identifiability.tolist(),
        "k_lo_md_post": float(np.mean(k_post[region == 0]) / MD_TO_M2),
        "k_hi_md_post": float(np.mean(k_post[region == 1]) / MD_TO_M2),
        "k_contrast_true": float(k_hi / k_lo),
        "k_contrast_post": float(np.mean(k_post[region == 1]) / max(float(np.mean(k_post[region == 0])), 1e-30)),
        "logk_rmse_prior": float(np.sqrt(np.mean((log_permeability(prior) - log_permeability(k_true)) ** 2))),
        "logk_rmse_post": float(np.sqrt(np.mean((log_permeability(k_post) - log_permeability(k_true)) ** 2))),
        "assimilate_nrmse": post.assimilate_rmse,
        "holdout_nrmse": post.holdout_rmse,
        "esmda_mismatch": post.esmda.diagnostics.data_mismatch,
        "notes": list(post.notes),
    }


def _physics(*, p_init: float, sw_init: float) -> PhysicsSpec:
    return PhysicsSpec(
        relperm=CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        capillary=NoCapillary(),
        sw_init=sw_init,
        p_init=p_init,
        dt_init=30.0,
        dt_min=0.5,
        dt_max=120.0,
        max_cfl=0.50,
        max_ds=0.18,
    )


def _sample_gauges(twin: DigitalTwin, sensors: list[Sensor], traj) -> dict:
    series = {}
    for s in sensors:
        vals = []
        for i, t in enumerate(traj.times_s):
            rates = traj.port_rates[i] if i < len(traj.port_rates) else {}
            vals.append(float(twin.operator.sample(s, traj.state_at(float(t)), port_rates=rates)))
        series[s.name] = vals
    return {"times_s": np.asarray(traj.times_s, dtype=float).tolist(), "series": series}


def _make_obs_from_traj(twin: DigitalTwin, sensors: list[Sensor], hold: set[str], times: np.ndarray, traj, seed: int) -> list[ObservationSeries]:
    rng = np.random.default_rng(seed)
    obs = []
    for s in sensors:
        vals = []
        for t in times:
            idx = int(np.argmin(np.abs(traj.times_s - t)))
            vals.append(twin.operator.sample(s, traj.state_at(t), port_rates=traj.port_rates[idx]))
        va = np.asarray(vals, dtype=float) + rng.normal(0.0, s.sigma, size=len(vals))
        obs.append(ObservationSeries(s.name, s.kind, times, va, np.full(len(vals), s.sigma), s.name in hold))
    return obs


def self_consistent(truth: dict, grid: CartesianGrid, *, design: str) -> dict:
    k_lo = float(truth["layers"]["k_lo_m2"])
    k_hi = float(truth["layers"]["k_hi_m2"])
    region = _region(grid, int(truth["layers"]["n_top_high"]))
    k_true = np.where(region == 1, k_hi, k_lo)
    param = RegionParameterization(region, phi=float(truth["controls"]["phi"]))
    p_sigma, s_sigma = 2.5e4, 0.04
    if design == "diverse":
        sensors, hold = diverse_sensors(grid, p_sigma=p_sigma, s_sigma=s_sigma, with_rate=False)
    elif design == "sparse":
        sensors, hold = sparse_sensors(grid, p_sigma=p_sigma, s_sigma=s_sigma)
    else:
        raise ValueError(design)
    times = np.array([0.125, 0.25, 0.375, 0.50]) * DAY_S
    inj, prod = _ports(grid, truth=truth)
    experiment = Experiment(
        size_m=grid.size_m(),
        sensors=sensors,
        controls=_same_cmg_controls(truth, times),
        observations=[],
        history_end_s=float(0.375 * DAY_S),
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        _physics(p_init=psi_to_pa(float(truth["controls"]["pres_psi"])), sw_init=float(truth["controls"]["swi"])),
        param,
        inverse=InverseSpec(
            n_ensemble=12,
            n_assimilations=4,
            seed=3 if design == "diverse" else 11,
            prior_mean=np.log(100.0 * MD_TO_M2),
            prior_std=0.8,
            n_workers=4,
        ),
    )
    print(f"A/{design}) self-consistent F(m_true), n_sensors={len(sensors)} ...", flush=True)
    traj = twin.simulate(Rock(k_true, np.full(grid.n_cells, param.phi)), t_end=float(times[-1]), report_times=times)
    twin.experiment.observations = _make_obs_from_traj(twin, sensors, hold, times, traj, seed=4)
    t0 = time.perf_counter()
    post = twin.calibrate()
    fc = twin.forecast(post)
    out = _score_k(k_true, region, post, k_lo, k_hi)
    out["forecast_nrmse"] = twin.score_forecast(fc)
    out["elapsed_s"] = time.perf_counter() - t0
    out["gauges_truth"] = _sample_gauges(twin, sensors, traj)
    out["gauges_post"] = _sample_gauges(twin, sensors, fc)
    out["n_sensors"] = len(sensors)
    out["sensor_names"] = [s.name for s in sensors]
    out["holdout"] = sorted(hold)
    out["design"] = design
    out["pass"] = bool(
        out["k_contrast_post"] > 5.0
        and out["logk_rmse_post"] < out["logk_rmse_prior"]
        and out["esmda_mismatch"][-1] < out["esmda_mismatch"][0]
    )
    print(
        f"   contrast {out['k_contrast_post']:.2f}  logk_rmse {out['logk_rmse_post']:.3f}  "
        f"hold {out['holdout_nrmse']:.3f}  ident {out['identifiability']}  pass={out['pass']}",
        flush=True,
    )
    np.save(HERE / f"k_post_self_{design}.npy", post.esmda.k_mean)
    if design == "diverse":
        np.save(HERE / "k_true.npy", k_true)
        np.save(HERE / "k_post_self.npy", post.esmda.k_mean)
    return out


def cmg_design_report(grid: CartesianGrid) -> dict:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    rates = parse_surface_rates_m3s(OUT)
    snaps = []
    for t, a in p_series:
        if abs(t - round(t * 4.0) / 4.0) > 1.0e-9 and t not in {0.25, 0.5, 1.0, 2.0, 4.0, 8.0}:
            continue
        if t not in {0.25, 0.5, 1.0, 2.0, 4.0, 8.0}:
            continue
        sw = next((b for ts, b in sw_series if abs(ts - t) < 1.0e-9), None)
        rec = {
            "day": t,
            "p_mean_psi": float(a.mean()),
            "p_std_psi": float(a.std()),
            "p_min_psi": float(a.min()),
            "p_max_psi": float(a.max()),
        }
        if sw is not None:
            rec.update(
                {
                    "sw_mean": float(sw.mean()),
                    "sw_std": float(sw.std()),
                    "sw_min": float(sw.min()),
                    "sw_max": float(sw.max()),
                }
            )
        if rates:
            rk = np.array(list(rates.keys()), dtype=float)
            q = rates[_nearest(rk, t)]
            rec["inj_stbday"] = float(q["INJ"] * DAY_S / 0.158987)
            rec["prod_water_stbday"] = float(q["PROD"] * DAY_S / 0.158987)
        snaps.append(rec)
    informative = all(s["p_std_psi"] > 20.0 for s in snaps if s["day"] <= 1.0)
    return {"snapshots": snaps, "delta_p_informative": informative}


def cross_cmg(truth: dict, grid: CartesianGrid) -> dict:
    """Invert CMG gauge readings with F_lab. Same u(t) and ICs as the deck."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    if not p_series or not sw_series:
        raise SystemExit("CMG .out missing p/Sw maps")
    want = [0.25, 0.50, 1.00, 2.00]
    p_t = np.array([t for t, _ in p_series])
    s_t = np.array([t for t, _ in sw_series])
    days = [t for t in want if np.min(np.abs(p_t - t)) < 0.08 and np.min(np.abs(s_t - t)) < 0.08]
    if len(days) < 3:
        raise SystemExit(f"not enough CMG times: {days}")
    times = np.asarray(days, dtype=float) * DAY_S
    hist_end = float(1.00 * DAY_S)
    p_map = {t: (_cmg_to_our(a) * 6894.757293168).ravel() for t, a in p_series}
    sw_map = {t: _cmg_to_our(a).ravel() for t, a in sw_series}

    def pick(store, t):
        ks = np.array(list(store.keys()), dtype=float)
        return store[_nearest(ks, t)]

    k_lo = float(truth["layers"]["k_lo_m2"])
    k_hi = float(truth["layers"]["k_hi_m2"])
    region = _region(grid, int(truth["layers"]["n_top_high"]))
    k_true = np.where(region == 1, k_hi, k_lo)
    param = RegionParameterization(region, phi=float(truth["controls"]["phi"]))
    sensors, hold = diverse_sensors(grid, p_sigma=psi_to_pa(30.0), s_sigma=0.05, with_rate=False)
    op = ObservationOperator(grid, sensors)
    rng = np.random.default_rng(8)
    obs = []
    cmg_clean: dict[str, list[float]] = {}
    for s in sensors:
        vals = []
        for td in days:
            st = State(pressure=pick(p_map, td), sw=pick(sw_map, td))
            vals.append(op.sample(s, st))
        cmg_clean[s.name] = [float(v) for v in vals]
        va = np.asarray(vals, dtype=float) + rng.normal(0.0, s.sigma, size=len(vals))
        obs.append(ObservationSeries(s.name, s.kind, times, va, np.full(len(vals), s.sigma), s.name in hold))
    inj, prod = _ports(grid, inj_control="pressure", truth=truth)
    experiment = Experiment(
        size_m=grid.size_m(),
        sensors=sensors,
        controls=_same_cmg_controls(truth, times),
        observations=obs,
        history_end_s=hist_end,
    )
    # Model-error R: residual of F(K_CMG) vs CMG gauges is not rock. Do not dump it into K.
    probe = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        _physics(p_init=psi_to_pa(float(truth["controls"]["pres_psi"])), sw_init=float(truth["controls"]["swi"])),
        param,
    )
    traj_m = probe.simulate(
        Rock(k_true, np.full(grid.n_cells, param.phi)),
        t_end=float(times[min(1, len(times) - 1)]),
        report_times=times[times <= hist_end + 1.0],
    )
    inflated = []
    extras = {}
    for series in obs:
        sensor = next(s for s in sensors if s.name == series.sensor_name)
        pred = []
        mask = series.times_s <= hist_end + 1.0
        for t in series.times_s:
            idx = int(np.argmin(np.abs(traj_m.times_s - t)))
            pred.append(probe.operator.sample(sensor, traj_m.state_at(t), port_rates=traj_m.port_rates[idx]))
        pred_a = np.asarray(pred, dtype=float)
        extra = float(np.sqrt(np.mean((pred_a[mask] - series.values[mask]) ** 2))) if np.any(mask) else 0.0
        extras[series.sensor_name] = extra
        sig = np.sqrt(series.sigma**2 + extra * extra)
        inflated.append(
            ObservationSeries(series.sensor_name, series.kind, series.times_s, series.values, sig, series.holdout)
        )
    experiment.observations = inflated
    print(f"   model-error inflate extras={ {k: round(v, 3) if v < 10 else round(v) for k, v in extras.items()} }", flush=True)
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        _physics(p_init=psi_to_pa(float(truth["controls"]["pres_psi"])), sw_init=float(truth["controls"]["swi"])),
        param,
        inverse=InverseSpec(
            n_ensemble=12,
            n_assimilations=3,
            seed=6,
            prior_mean=np.log(100.0 * MD_TO_M2),
            prior_std=0.8,
            n_workers=4,
        ),
    )
    print("B) invert CMG gauges with F_lab (K_CMG is not the pass bar) ...", flush=True)
    t0s = time.perf_counter()
    post = twin.calibrate()
    fc = twin.forecast(post)
    out = _score_k(k_true, region, post, k_lo, k_hi)
    out["forecast_nrmse"] = twin.score_forecast(fc)
    out["elapsed_s"] = time.perf_counter() - t0s
    out["times_day"] = days
    out["p_std_psi_last"] = float(np.std(pick(p_map, days[-1]) / 6894.757293168))
    out["gauges_cmg"] = {"times_day": days, "series": cmg_clean}
    out["gauges_post"] = _sample_gauges(twin, sensors, fc)
    out["sensor_names"] = [s.name for s in sensors]
    out["holdout"] = sorted(hold)
    out["pass_observations"] = bool(
        out["esmda_mismatch"][-1] < out["esmda_mismatch"][0] and np.isfinite(out["holdout_nrmse"])
    )
    out["k_direction_ok"] = bool(out["k_contrast_post"] > 1.2)
    print(
        f"   misfit {out['esmda_mismatch'][0]:.3f}→{out['esmda_mismatch'][-1]:.3f}  "
        f"hold {out['holdout_nrmse']:.3f}  forecast {out['forecast_nrmse']:.3f}  "
        f"equiv contrast {out['k_contrast_post']:.2f}  pass_obs={out['pass_observations']}",
        flush=True,
    )
    np.save(HERE / "k_post_cmg_obs.npy", post.esmda.k_mean)
    return out


def main() -> int:
    if not OUT.is_file():
        raise SystemExit(f"missing {OUT}")
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth)
    text = OUT.read_text(encoding="latin-1", errors="ignore")
    health = {"normal_termination": "Normal Termination" in text}
    design = cmg_design_report(grid)
    print("IMEX", health, "informative_dp", design["delta_p_informative"], flush=True)
    for snap in design["snapshots"]:
        print(
            f"   t={snap['day']:.2f}d  p_std={snap['p_std_psi']:.1f} psi  "
            f"sw_std={snap.get('sw_std', float('nan')):.3f}",
            flush=True,
        )
    a_div = self_consistent(truth, grid, design="diverse")
    b = cross_cmg(truth, grid)
    report = {
        "philosophy": {
            "same_as_cmg": "same well controls and gauge coordinates, not F ≡ IMEX",
            "pass_A": "recover layer K from our own F on the CMG grid",
            "pass_B": "fit/hold-out CMG *observations*; posterior K is equivalent lab K",
        },
        "imex": health,
        "cmg_experiment_design": design,
        "A_self_consistent_diverse": a_div,
        "B_cross_cmg_observations": b,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "A_diverse_pass": a_div["pass"],
                "A_contrast": a_div["k_contrast_post"],
                "A_logk_rmse": a_div["logk_rmse_post"],
                "B_pass_obs": b["pass_observations"],
                "B_hold": b["holdout_nrmse"],
                "B_contrast": b["k_contrast_post"],
                "B_forecast": b.get("forecast_nrmse"),
            },
            indent=2,
        )
    )
    return 0 if a_div["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
