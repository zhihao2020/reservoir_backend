"""Map a CMG case (truth.json + .out) onto lab F / H / u(t)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor, State, column_sensors
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import (
    CoarseFieldParameterization,
    ContrastParameterization,
    RegionParameterization,
)
from reservoir_backend.physics.capillary import TableCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import TableTwoPhase
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec
from reservoir_backend.validation.cmg_harness.catalog import DAY_S, FT_TO_M, MD_TO_M2, PSI, CaseSpec


def _ensure_cmg_io() -> None:
    val = Path(__file__).resolve().parents[3] / "black_oil" / "validation"
    if str(val) not in sys.path:
        sys.path.insert(0, str(val))


def load_truth(spec: CaseSpec) -> dict:
    return json.loads(spec.truth_path.read_text(encoding="utf-8"))


def k1_is_top(truth: dict) -> bool:
    return str((truth.get("grid") or {}).get("k1", "top")).lower() == "top"


def our_k_from_cmg(nz: int, k_cmg: int, *, k1_top: bool) -> int:
    k = int(k_cmg)
    if k1_top:
        return int(nz) - k
    return k - 1


def cmg_field_to_ours(arr: NDArray[np.float64], *, k1_top: bool) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=float)
    return a[::-1] if k1_top else a


def grid_from_truth(truth: dict) -> CartesianGrid:
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    dx = np.full(nx, float(g["di_ft"]) * FT_TO_M)
    dy = np.full(ny, float(g["dj_ft"]) * FT_TO_M)
    dk = g["dk_ft"]
    if isinstance(dk, (int, float)):
        dz = np.full(nz, float(dk) * FT_TO_M)
    else:
        dz = np.asarray(dk, dtype=float) * FT_TO_M
        if k1_is_top(truth):
            dz = dz[::-1]
    return CartesianGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)


def _well_ks(spec_w: dict, nz: int, *, k1_top: bool) -> list[int]:
    raw = spec_w.get("k_cmg") or spec_w.get("k_perfs") or list(range(1, nz + 1))
    return [our_k_from_cmg(nz, int(k), k1_top=k1_top) for k in raw]


def _wells_map(truth: dict) -> dict:
    wells = truth.get("wells") or {}
    if isinstance(wells, list):
        return {str(w.get("name", f"W{i}")): w for i, w in enumerate(wells)}
    return dict(wells)


def ports_from_truth(grid: CartesianGrid, truth: dict) -> list[FlowPort]:
    wells = _wells_map(truth)
    k1_top = k1_is_top(truth)
    ports: list[FlowPort] = []
    for name, spec_w in wells.items():
        role = str(spec_w.get("role") or ("injector" if "inj" in name.lower() else "producer"))
        cells = _well_cells(grid, spec_w, k1_top=k1_top)
        rate = spec_w.get("rate_m3s")
        if rate is not None and abs(float(rate)) > 0.0:
            control = "rate"
        else:
            control = "pressure"
        ports.append(
            FlowPort(
                name=str(name),
                role=role,
                control=control,
                cell_ids=cells,
                sw_inj=1.0 if role == "injector" else 1.0,
            )
        )
    return ports


def _well_cells(grid: CartesianGrid, spec_w: dict, *, k1_top: bool) -> NDArray[np.int64]:
    if spec_w.get("i0") is not None and spec_w.get("i1") is not None:
        i0, i1 = int(spec_w["i0"]) - 1, int(spec_w["i1"]) - 1
        j = int(spec_w.get("j", 1)) - 1
        kk = our_k_from_cmg(grid.nz, int(spec_w.get("k", 1)), k1_top=k1_top)
        lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
        return np.array([grid.index(i, j, kk) for i in range(lo, hi + 1)], dtype=np.int64)
    i = int(spec_w.get("i", 1)) - 1
    j = int(spec_w.get("j", 1)) - 1
    ks = _well_ks(spec_w, grid.nz, k1_top=k1_top)
    return np.array([grid.index(i, j, k) for k in ks], dtype=np.int64)


def producer_cells(ports: list[FlowPort]) -> NDArray[np.int64]:
    cells: list[int] = []
    for p in ports:
        if p.role == "producer":
            cells.extend(int(c) for c in p.cell_ids)
    return np.asarray(cells, dtype=np.int64)


def _rate_curve(out_path: Path, times_s: NDArray[np.float64], key: str) -> NDArray[np.float64] | None:
    _ensure_cmg_io()
    from cmg_io.grid_parse import parse_liquid_rates_m3s

    series = parse_liquid_rates_m3s(out_path)
    if not series:
        return None
    t_d = np.asarray(sorted(series), dtype=float)
    y = np.asarray([series[float(t)][key] for t in t_d], dtype=float)
    if key == "INJ" and float(np.max(y)) <= 0.0:
        return None
    return np.interp(np.asarray(times_s, dtype=float), t_d * DAY_S, y, left=y[0], right=y[-1])


def controls_from_truth(
    truth: dict,
    ports: list[FlowPort],
    times: NDArray[np.float64],
    *,
    out_path: Path | None = None,
) -> list[ControlSeries]:
    ctrl = truth.get("controls") or {}
    out: list[ControlSeries] = []
    if "inj_bhp_psi" in ctrl and "prod_bhp_psi" in ctrl:
        p_inj = float(ctrl["inj_bhp_psi"]) * PSI
        p_prod = float(ctrl["prod_bhp_psi"]) * PSI
        for port in ports:
            if port.control != "pressure":
                continue
            val = p_inj if port.role == "injector" else p_prod
            out.append(ControlSeries(port.name, "pressure", times, np.full(times.size, val)))
            if port.role == "injector":
                out.append(ControlSeries(port.name, "composition", times, np.full(times.size, 1.0)))
        return out

    inj_curve = (
        _rate_curve(out_path, times, "INJ")
        if out_path is not None and Path(out_path).is_file()
        else None
    )
    prod_curve = (
        _rate_curve(out_path, times, "PROD")
        if out_path is not None and Path(out_path).is_file()
        else None
    )
    inj_ports = [p for p in ports if p.role == "injector"]
    n_inj = max(len(inj_ports), 1)
    for port in ports:
        spec_w = _wells_map(truth).get(port.name) or {}
        values = None
        if spec_w.get("rate_m3s") is not None:
            values = np.full(times.size, float(spec_w["rate_m3s"]))
        elif inj_curve is not None and port.role == "injector":
            values = inj_curve / n_inj
        elif prod_curve is not None and port.role == "producer":
            values = prod_curve
        elif port.control == "rate":
            values = np.zeros(times.size)
        if values is None:
            bhp = 3200.0 * PSI if port.role == "injector" else 2800.0 * PSI
            out.append(ControlSeries(port.name, "pressure", times, np.full(times.size, bhp)))
            if port.role == "injector":
                out.append(ControlSeries(port.name, "composition", times, np.full(times.size, 1.0)))
            continue
        object.__setattr__(port, "control", "rate")
        out.append(ControlSeries(port.name, "rate", times, values))
        if port.role == "injector":
            out.append(ControlSeries(port.name, "composition", times, np.full(times.size, 1.0)))
    return out


def _channel_region(grid: CartesianGrid, truth: dict) -> NDArray[np.int64] | None:
    blocks = truth.get("channel_blocks_ijk") or []
    if not blocks:
        return None
    k1_top = k1_is_top(truth)
    rid = np.zeros(grid.n_cells, dtype=np.int64)
    for trip in blocks:
        i, j, kk = int(trip[0]) - 1, int(trip[1]) - 1, our_k_from_cmg(grid.nz, int(trip[2]), k1_top=k1_top)
        if 0 <= i < grid.nx and 0 <= j < grid.ny and 0 <= kk < grid.nz:
            rid[grid.index(i, j, kk)] = 1
    if int(rid.max()) < 1:
        return None
    return rid


def parameterization_from_truth(spec: CaseSpec, grid: CartesianGrid, truth: dict):
    phi = float((truth.get("controls") or {}).get("phi", 0.30))
    if spec.parameterization == "region":
        if truth.get("layers"):
            n_top = int((truth.get("layers") or {}).get("n_top_high", grid.nz // 2))
            rid = np.zeros(grid.n_cells, dtype=np.int64)
            for c in range(grid.n_cells):
                _i, _j, k = grid.ijk(c)
                if k >= grid.nz - n_top:
                    rid[c] = 1
            return ContrastParameterization(rid, phi=phi)
        rid = _channel_region(grid, truth)
        if rid is not None:
            return ContrastParameterization(rid, phi=phi)
        raise ValueError(f"{spec.id} asked for region parameterization but has no layers/channel_blocks")
    nx, ny, nz = spec.coarse or (3, 3, 2)
    return CoarseFieldParameterization(grid, nx=nx, ny=ny, nz=nz, phi=phi)


def k_true_m2(grid: CartesianGrid, truth: dict) -> NDArray[np.float64] | None:
    layers = truth.get("layers") or {}
    if "k_lo_m2" in layers and "k_hi_m2" in layers:
        n_top = int(layers.get("n_top_high", grid.nz // 2))
        k = np.full(grid.n_cells, float(layers["k_lo_m2"]))
        for c in range(grid.n_cells):
            _i, _j, kk = grid.ijk(c)
            if kk >= grid.nz - n_top:
                k[c] = float(layers["k_hi_m2"])
        return k
    bg = (truth.get("background_perm_md") or {}).get("kx")
    if bg is None:
        return None
    k = np.full(grid.n_cells, float(bg) * MD_TO_M2)
    ch = (truth.get("channel_perm_md") or {}).get("kx")
    blocks = truth.get("channel_blocks_ijk") or []
    if ch is None or not blocks:
        return k
    k1_top = k1_is_top(truth)
    kh = float(ch) * MD_TO_M2
    for trip in blocks:
        i, j, kk = int(trip[0]) - 1, int(trip[1]) - 1, our_k_from_cmg(grid.nz, int(trip[2]), k1_top=k1_top)
        if 0 <= i < grid.nx and 0 <= j < grid.ny and 0 <= kk < grid.nz:
            k[grid.index(i, j, kk)] = kh
    return k


def kz_ratio_from_truth(grid: CartesianGrid, truth: dict) -> NDArray[np.float64]:
    """Cell-wise kz/kx. Known structure, not inverted."""
    layers = truth.get("layers") or {}
    if "k_lo_m2" in layers:
        # lab_layers PERMK/PERMI = 50/500
        return np.full(grid.n_cells, 0.10)
    bg = truth.get("background_perm_md") or {}
    kx_bg = float(bg.get("kx") or 40.0)
    kz_bg = float(bg.get("kz") or 0.1 * kx_bg)
    ratio = np.full(grid.n_cells, kz_bg / max(kx_bg, 1.0e-30))
    ch = truth.get("channel_perm_md") or {}
    blocks = truth.get("channel_blocks_ijk") or []
    if ch.get("kx") and ch.get("kz") and blocks:
        r_ch = float(ch["kz"]) / max(float(ch["kx"]), 1.0e-30)
        k1_top = k1_is_top(truth)
        for trip in blocks:
            i, j, kk = int(trip[0]) - 1, int(trip[1]) - 1, our_k_from_cmg(grid.nz, int(trip[2]), k1_top=k1_top)
            if 0 <= i < grid.nx and 0 <= j < grid.ny and 0 <= kk < grid.nz:
                ratio[grid.index(i, j, kk)] = r_ch
    return ratio


def face_mult_from_truth(grid: CartesianGrid, truth: dict) -> NDArray[np.float64] | None:
    """I-face TRANSI multipliers (known fault). None if the deck is open."""
    fault = truth.get("fault") or {}
    if not fault:
        return None
    seal = float(fault.get("seal_transi_mult", 0.0))
    window = float(fault.get("window_transi_mult", 1.0))
    seal_j = [int(j) - 1 for j in (fault.get("seal_j") or [])]
    window_j = [int(j) - 1 for j in (fault.get("window_j") or [])]
    if not seal_j and not window_j:
        return None
    # CMG TRANSI at I=5 is the +I face of cell I=5 → between 0-based 4 and 5.
    i_face = 4
    if grid.nx < 6:
        return None
    mx = np.ones((grid.nz, grid.ny, grid.nx - 1), dtype=float)
    for j in seal_j:
        if 0 <= j < grid.ny:
            mx[:, j, i_face] = seal
    for j in window_j:
        if 0 <= j < grid.ny:
            mx[:, j, i_face] = window
    return mx


def default_sensors(grid: CartesianGrid) -> tuple[list[Sensor], set[str]]:
    lx, ly, lz = grid.size_m()
    bot, mid, top = 0.18 * lz, 0.50 * lz, 0.82 * lz
    p_sigma, s_sigma = 30.0 * PSI, 0.05
    sensors: list[Sensor] = []
    sensors += column_sensors("Pin", "pressure", 0.25 * lx, 0.50 * ly, [bot, top], sigma=p_sigma, labels=("bot", "top"))
    sensors += column_sensors("Pmid", "pressure", 0.50 * lx, 0.50 * ly, [mid], sigma=p_sigma, labels=("mid",))
    sensors += column_sensors("Pout", "pressure", 0.75 * lx, 0.50 * ly, [bot, top], sigma=p_sigma, labels=("bot", "top"))
    sensors += column_sensors("Sin", "saturation", 0.35 * lx, 0.50 * ly, [bot, top], sigma=s_sigma, labels=("bot", "top"))
    sensors += column_sensors("Sout", "saturation", 0.65 * lx, 0.50 * ly, [top], sigma=s_sigma, labels=("top",))
    return sensors, {"Pout_top", "Sout_top"}


def sensors_for_spec(spec: CaseSpec, grid: CartesianGrid) -> tuple[list[Sensor], set[str]]:
    """Gauges follow the well pattern. Not a search over K."""
    lx, ly, lz = grid.size_m()
    mid = 0.50 * lz
    p_sigma, s_sigma = 30.0 * PSI, 0.05
    if spec.kind == "fivespot":
        sensors: list[Sensor] = []
        for tag, x, y in (
            ("sw", 0.12 * lx, 0.12 * ly),
            ("se", 0.88 * lx, 0.12 * ly),
            ("nw", 0.12 * lx, 0.88 * ly),
            ("ne", 0.88 * lx, 0.88 * ly),
        ):
            sensors.append(Sensor(f"P_{tag}", "pressure", x, y, mid, sigma=p_sigma))
        sensors.append(Sensor("P_ctr", "pressure", 0.50 * lx, 0.50 * ly, mid, sigma=p_sigma))
        sensors.append(Sensor("S_ctr", "saturation", 0.50 * lx, 0.50 * ly, mid, sigma=s_sigma))
        sensors.append(Sensor("S_ne", "saturation", 0.70 * lx, 0.70 * ly, mid, sigma=s_sigma))
        sensors.append(_mean_pressure_sensor(grid, p_sigma=80.0 * PSI))
        return sensors, {"P_ne", "S_ne"}
    if spec.kind in {"fault", "channel"}:
        sensors = []
        sensors += column_sensors("Pin", "pressure", 0.18 * lx, 0.45 * ly, [0.30 * lz, 0.70 * lz], sigma=p_sigma, labels=("lo", "hi"))
        sensors += column_sensors("Pmid", "pressure", 0.50 * lx, 0.55 * ly, [mid], sigma=p_sigma, labels=("mid",))
        sensors += column_sensors("Pout", "pressure", 0.85 * lx, 0.82 * ly, [0.30 * lz, 0.70 * lz], sigma=p_sigma, labels=("lo", "hi"))
        sensors.append(Sensor("Sin", "saturation", 0.30 * lx, 0.45 * ly, mid, sigma=s_sigma))
        sensors.append(Sensor("Sout", "saturation", 0.72 * lx, 0.78 * ly, mid, sigma=s_sigma))
        # Channel vs matrix saturations break the (high k_lo, mild contrast) tradeoff.
        sensors.append(Sensor("S_ch", "saturation", 0.22 * lx, 0.50 * ly, mid, sigma=s_sigma))
        sensors.append(Sensor("S_mx", "saturation", 0.55 * lx, 0.18 * ly, mid, sigma=s_sigma))
        sensors.append(_mean_pressure_sensor(grid, p_sigma=80.0 * PSI))
        return sensors, {"Pout_hi", "Sout", "S_mx"}
    return default_sensors(grid)


def _mean_pressure_sensor(grid: CartesianGrid, *, p_sigma: float) -> Sensor:
    """Domain-mean pressure. One observation for the ct nuisance."""
    lx, ly, lz = grid.size_m()
    return Sensor(
        "Pbar",
        "pressure",
        0.50 * lx,
        0.50 * ly,
        0.50 * lz,
        volume_m3=8.0 * lx * ly * lz,
        sigma=p_sigma,
    )


def physics_from_truth(truth: dict, spec: CaseSpec | None = None) -> PhysicsSpec:
    ctrl = truth.get("controls") or {}
    swi = float(ctrl.get("swi", 0.20))
    if "pres_psi" in ctrl:
        p_init = float(ctrl["pres_psi"]) * PSI
    else:
        p_init = 3000.0 * PSI
    dt_init = 30.0 if spec is None else float(spec.dt_init_s)
    dt_max = 120.0 if spec is None else float(spec.dt_max_s)
    pb = float(ctrl.get("pb_psi", 2500.0)) * PSI
    pvt = BlackOilPVT.cmg_seawater(p_init=p_init, pb=pb)
    field = spec is not None and spec.dt_max_s >= 3600.0
    mu_w = pvt.mu_w if field else 1.0e-3
    mu_o = pvt.mu_o if field else 1.0e-3
    bg = truth.get("background_perm_md") or {}
    if bg.get("kx") and bg.get("kz"):
        kz_over = float(bg["kz"]) / max(float(bg["kx"]), 1.0e-30)
    else:
        kz_over = 0.10
    return PhysicsSpec(
        relperm=TableTwoPhase.cmg_seawater(mu_w=mu_w, mu_o=mu_o),
        capillary=TableCapillary.cmg_swt(),
        pvt=pvt,
        gravity=9.80665 if field else 0.0,
        kz_over_kx=kz_over,
        sw_init=swi,
        p_init=p_init,
        dt_init=dt_init,
        dt_min=0.5 if dt_max <= 120.0 else 60.0,
        dt_max=dt_max,
        max_cfl=0.50,
        max_ds=0.22,
        implicit_transport=True,
    )


def load_cmg_maps(spec: CaseSpec, truth: dict, days: tuple[float, ...]) -> dict[float, dict[str, NDArray[np.float64]]]:
    _ensure_cmg_io()
    from cmg_io.grid_parse import parse_grid_series

    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    p_series = parse_grid_series(spec.out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    s_series = parse_grid_series(spec.out_path, field="sw", nx=nx, ny=ny, nz=nz)
    k1_top = k1_is_top(truth)

    def nearest(series, day: float):
        ts = np.array([t for t, _ in series], dtype=float)
        t = float(ts[int(np.argmin(np.abs(ts - day)))])
        arr = next(a for tt, a in series if tt == t)
        return t, arr

    out: dict[float, dict[str, NDArray[np.float64]]] = {}
    for d in days:
        _tp, p = nearest(p_series, d)
        _ts, s = nearest(s_series, d)
        out[float(d)] = {
            "p": cmg_field_to_ours(p, k1_top=k1_top),
            "sw": cmg_field_to_ours(s, k1_top=k1_top),
        }
    return out


def observations_from_cmg(
    grid: CartesianGrid,
    sensors: list[Sensor],
    hold: set[str],
    maps: dict[float, dict[str, NDArray[np.float64]]],
    days: tuple[float, ...],
) -> list[ObservationSeries]:
    from reservoir_backend.observation.operator import ObservationOperator

    op = ObservationOperator(grid, sensors)
    times = np.asarray(days, dtype=float) * DAY_S
    rng = np.random.default_rng(8)
    obs: list[ObservationSeries] = []
    for s in sensors:
        vals = []
        for d in days:
            st = State(
                pressure=np.asarray(maps[float(d)]["p"], dtype=float).ravel() * PSI,
                sw=np.asarray(maps[float(d)]["sw"], dtype=float).ravel(),
            )
            vals.append(op.sample(s, st))
        va = np.asarray(vals, dtype=float) + rng.normal(0.0, s.sigma, size=len(vals))
        obs.append(
            ObservationSeries(s.name, s.kind, times, va, np.full(len(vals), s.sigma), s.name in hold)
        )
    return obs


def build_twin(
    spec: CaseSpec,
    *,
    knobs: dict | None = None,
    with_observations: bool = True,
) -> tuple[DigitalTwin, dict]:
    if spec.status == "unsupported":
        raise RuntimeError(f"{spec.id} is unsupported: {spec.note}")
    if not spec.out_path.is_file():
        raise FileNotFoundError(f"missing IMEX out: {spec.out_path}")
    truth = load_truth(spec)
    grid = grid_from_truth(truth)
    ports = ports_from_truth(grid, truth)
    sensors, hold = sensors_for_spec(spec, grid)
    days = spec.history_days
    if spec.forecast_day is not None:
        days = tuple(list(days) + [spec.forecast_day])
    maps = load_cmg_maps(spec, truth, days)
    times = np.asarray(days, dtype=float) * DAY_S
    controls = controls_from_truth(truth, ports, times, out_path=spec.out_path)
    obs = observations_from_cmg(grid, sensors, hold, maps, days) if with_observations else []
    hist_end = float(spec.history_days[-1] * DAY_S)
    knobs = dict(knobs or {})
    inverse = InverseSpec(
        n_ensemble=int(knobs.get("n_ensemble", 12)),
        n_assimilations=int(knobs.get("n_assimilations", 3)),
        seed=int(knobs.get("seed", 6)),
        prior_mean=float(np.log(spec.prior_k_md * MD_TO_M2)),
        prior_std=float(knobs.get("prior_std", 0.35 if spec.kind != "layers" else 0.8)),
        inflation=float(knobs.get("inflation", 1.02)),
        algorithm=str(knobs.get("algorithm", "esmda")),
        n_workers=int(knobs.get("n_workers", 4)),
    )
    twin = DigitalTwin(
        grid,
        Experiment(
            size_m=grid.size_m(),
            sensors=sensors,
            controls=controls,
            observations=obs,
            history_end_s=hist_end,
        ),
        ports,
        physics_from_truth(truth, spec),
        parameterization_from_truth(spec, grid, truth),
        face_mult_x=face_mult_from_truth(grid, truth),
        kz_ratio=kz_ratio_from_truth(grid, truth),
        inverse=inverse,
    )
    extra = {
        "truth": truth,
        "maps": maps,
        "hold": hold,
        "k_true": k_true_m2(grid, truth),
        "producer_cells": producer_cells(ports),
        "sw_init": float((truth.get("controls") or {}).get("swi", 0.20)),
    }
    return twin, extra


def inflate_model_error(
    twin: DigitalTwin,
    k_true: NDArray[np.float64] | None,
    *,
    demean_pressure: bool = False,
    skip_pressure: bool = False,
    extra_scale: float = 0.70,
) -> None:
    """Add a fraction of F(K_CMG) vs CMG-gauge residual into R.

    Full extra would hide the contrast signal once F is near the floor.
    ``extra_scale<1`` keeps a model-error floor without locking θ to the prior.
    """
    if k_true is None or not twin.experiment.observations:
        return
    hist = float(twin.experiment.history_end_s or twin.experiment.controls[0].times_s[-1])
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    traj = twin.simulate(twin.rock_from_k(k_true), t_end=hist)
    inflated = []
    for series in twin.experiment.observations:
        sensor = next(s for s in twin.experiment.sensors if s.name == series.sensor_name)
        pred = []
        mask = series.times_s <= hist + 1.0
        for t in series.times_s:
            idx = int(np.argmin(np.abs(traj.times_s - t)))
            pred.append(twin.operator.sample(sensor, traj.state_at(t), port_rates=traj.port_rates[idx]))
        pred_a = np.asarray(pred, dtype=float)
        if not np.any(mask):
            extra = 0.0
        elif sensor.name == "Pbar":
            extra = 0.0
        elif sensor.kind == "pressure" and skip_pressure:
            extra = 0.0
        elif sensor.kind == "pressure" and demean_pressure:
            # Field mean p is PVT voidage, not rock. Lab keeps raw residual.
            pr = pred_a[mask] - float(np.mean(pred_a[mask]))
            ob = series.values[mask] - float(np.mean(series.values[mask]))
            extra = float(np.sqrt(np.mean((pr - ob) ** 2)))
        else:
            extra = float(np.sqrt(np.mean((pred_a[mask] - series.values[mask]) ** 2)))
        extra = float(extra_scale) * extra
        sig = np.sqrt(series.sigma**2 + extra * extra)
        inflated.append(
            ObservationSeries(series.sensor_name, series.kind, series.times_s, series.values, sig, series.holdout)
        )
    twin.experiment.observations = inflated
