"""Load an engineering YAML case and convert to SI at the boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import CoarseFieldParameterization, RegionParameterization
from reservoir_backend.physics.capillary import capillary_from_name
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec
from reservoir_backend.io.units import to_m2, to_m3_s, to_metres, to_pa, to_seconds


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("case file must be a mapping")
    return data


def _read_control_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = (str(row["port"]), str(row["kind"]))
        grouped.setdefault(key, []).append((float(row["time_s"]), float(row["value"])))
    out = []
    for (port, kind), pairs in grouped.items():
        pairs.sort()
        out.append(
            {
                "port": port,
                "kind": kind,
                "times": [p[0] for p in pairs],
                "values": [p[1] for p in pairs],
            }
        )
    return out


def _read_observation_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: dict[tuple[str, str], list[tuple[float, float, float, bool]]] = {}
    for row in rows:
        key = (str(row["sensor"]), str(row.get("kind", "pressure")))
        hold = str(row.get("holdout", "0")).strip() in {"1", "true", "True", "yes"}
        grouped.setdefault(key, []).append(
            (float(row["time_s"]), float(row["value"]), float(row.get("sigma", 1.0)), hold)
        )
    out = []
    for (sensor, kind), pairs in grouped.items():
        pairs.sort()
        out.append(
            {
                "sensor": sensor,
                "kind": kind,
                "times": [p[0] for p in pairs],
                "values": [p[1] for p in pairs],
                "sigma": [p[2] for p in pairs],
                "holdout": any(p[3] for p in pairs),
            }
        )
    return out


def _maybe_convert(value: float, unit: str | None, kind: str) -> float:
    if not unit:
        return float(value)
    if kind == "length":
        return to_metres(value, unit)
    if kind == "time":
        return to_seconds(value, unit)
    if kind == "pressure":
        return to_pa(value, unit)
    if kind == "rate":
        return to_m3_s(value, unit)
    if kind == "perm":
        return to_m2(value, unit)
    return float(value)


def grid_from_cfg(cfg: dict[str, Any]) -> CartesianGrid:
    geom = cfg.get("geometry") or {}
    size = tuple(float(x) for x in geom.get("size_m", [0.3, 0.3, 0.3]))
    origin = tuple(float(x) for x in geom.get("origin_m", [0.0, 0.0, 0.0]))
    grid_cfg = cfg.get("grid") or {}
    spacing = grid_cfg.get("spacing_m", 0.01)
    if isinstance(spacing, (int, float)):
        return CartesianGrid.uniform(size, float(spacing), origin=origin)
    return CartesianGrid.uniform(size, tuple(float(x) for x in spacing), origin=origin)


def build_twin(cfg: dict[str, Any], *, cfg_dir: str | Path = ".") -> DigitalTwin:
    cfg_dir = Path(cfg_dir)
    grid = grid_from_cfg(cfg)
    phys_cfg = cfg.get("physics") or {}
    model = str(phys_cfg.get("model", "two_phase_immiscible")).lower()
    cap_name = phys_cfg.get("capillary", "brooks_corey")
    if cap_name is True:
        cap_name = "brooks_corey"
    if cap_name is False:
        cap_name = "none"
    capillary = capillary_from_name(str(cap_name))
    p_init = float(phys_cfg.get("p_init", 1.0e6))
    pvt_name = str(phys_cfg.get("pvt", "")).strip().lower()
    if pvt_name in {"cmg", "cmg_seawater", "black_oil"} or model in {"black_oil", "d"}:
        pvt = BlackOilPVT.cmg_seawater(p_init=p_init)
    else:
        comp = phys_cfg.get("compressibility", "incompressible")
        if str(comp) in {"incompressible", "none", "0", ""}:
            pvt = BlackOilPVT.incompressible()
        else:
            ct = float(comp) if not isinstance(comp, str) else 1.5e-9
            pvt = BlackOilPVT.slightly_compressible(ct, pref=p_init)
    relperm = CoreyTwoPhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o)
    three = CoreyThreePhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o, mu_g=pvt.mu_g) if model in {"three_phase", "three_phase_immiscible", "c"} else None
    physics = PhysicsSpec(
        relperm=relperm,
        three_phase=three,
        capillary=capillary,
        pvt=pvt,
        single_phase=model in {"single_phase", "single", "a"},
        sw_init=float(phys_cfg.get("sw_init", relperm.swi)),
        sg_init=float(phys_cfg.get("sg_init", 0.0)),
        p_init=p_init,
        dt_init=float(phys_cfg.get("dt_init", 5.0)),
        dt_max=float(phys_cfg.get("dt_max", 30.0)),
    )

    ports: list[FlowPort] = []
    for p in cfg.get("ports") or []:
        xyz = (float(p["x"]), float(p["y"]), float(p["z"]))
        ports.append(
            FlowPort.at_point(
                grid,
                str(p["name"]),
                str(p.get("role", "injector")),
                str(p.get("control", "rate")),
                xyz,
                radius_m=float(p.get("radius_m", 0.0)),
                sw_inj=float(p.get("sw_inj", 1.0)),
                use_productivity=bool(p.get("use_productivity", False)),
            )
        )

    sensors: list[Sensor] = []
    for s in cfg.get("sensors") or []:
        sensors.append(
            Sensor(
                name=str(s["name"]),
                kind=str(s["kind"]),
                x=float(s["x"]),
                y=float(s["y"]),
                z=float(s["z"]),
                volume_m3=float(s.get("volume_m3", 0.0)),
                port_name=s.get("port"),
                sigma=float(s.get("sigma", 1.0)),
            )
        )

    exp_cfg = cfg.get("experiment") or {}
    controls: list[ControlSeries] = []
    controls_src = exp_cfg.get("controls") or cfg.get("controls") or []
    if isinstance(controls_src, str):
        controls_src = _read_control_csv(Path(cfg_dir) / controls_src)
    for c in controls_src:
        kind = str(c["kind"])
        unit = c.get("unit")
        conv = "rate" if kind == "rate" else "pressure" if kind == "pressure" else None
        values = np.asarray(c["values"], dtype=float)
        if conv:
            values = np.array([_maybe_convert(v, unit, conv) for v in values], dtype=float)
        t_unit = c.get("time_unit")
        times = np.asarray(c["times"], dtype=float)
        if t_unit:
            times = np.array([to_seconds(t, t_unit) for t in times], dtype=float)
        controls.append(ControlSeries(str(c["port"]), kind, times, values))

    holdout = set(str(x) for x in (exp_cfg.get("holdout_sensors") or []))
    observations: list[ObservationSeries] = []
    obs_src = exp_cfg.get("observations") or cfg.get("observations") or []
    if isinstance(obs_src, str):
        obs_src = _read_observation_csv(Path(cfg_dir) / obs_src)
    for o in obs_src:
        kind = str(o.get("kind", "pressure"))
        unit = o.get("unit")
        conv = "pressure" if kind == "pressure" else "rate" if kind == "phase_rate" else None
        values = np.asarray(o["values"], dtype=float)
        if conv:
            values = np.array([_maybe_convert(v, unit, conv) for v in values], dtype=float)
        t_unit = o.get("time_unit")
        times = np.asarray(o["times"], dtype=float)
        if t_unit:
            times = np.array([to_seconds(t, t_unit) for t in times], dtype=float)
        sigma = np.asarray(o.get("sigma", 1.0), dtype=float)
        name = str(o["sensor"])
        observations.append(
            ObservationSeries(
                sensor_name=name,
                kind=kind,
                times_s=times,
                values=values,
                sigma=sigma,
                holdout=name in holdout or bool(o.get("holdout", False)),
            )
        )

    history_end = exp_cfg.get("history_end_s")
    if history_end is None and exp_cfg.get("history_end") is not None:
        history_end = to_seconds(float(exp_cfg["history_end"]), str(exp_cfg.get("history_end_unit", "s")))

    experiment = Experiment(
        size_m=grid.size_m(),
        origin_m=grid.origin,
        sensors=sensors,
        controls=controls,
        observations=observations,
        history_end_s=None if history_end is None else float(history_end),
    )

    inv = cfg.get("inverse") or {}
    kind = str(inv.get("parameterization", "coarse_field")).lower()
    phi = float((cfg.get("rock") or {}).get("porosity", 0.20))
    if kind == "region":
        nreg = int(inv.get("n_regions", 2))
        z = grid.cell_centers()[:, 2]
        cuts = np.quantile(z, np.linspace(0, 1, nreg + 1)[1:-1]) if nreg > 1 else []
        rid = np.zeros(grid.n_cells, dtype=np.int64)
        for i, c in enumerate(cuts, start=1):
            rid[z >= c] = i
        param: RegionParameterization | CoarseFieldParameterization = RegionParameterization(rid, phi=phi)
    else:
        coarse = inv.get("coarse_n", [6, 6, 6])
        param = CoarseFieldParameterization(grid, int(coarse[0]), int(coarse[1]), int(coarse[2]), phi=phi)

    from reservoir_backend.inverse.presets import knobs_for

    preset = inv.get("preset")
    knobs = knobs_for(str(preset)) if preset else {}
    inverse = InverseSpec(
        n_ensemble=int(inv.get("ensemble_size", knobs.get("n_ensemble", 24))),
        n_assimilations=int(inv.get("assimilation_steps", knobs.get("n_assimilations", 4))),
        seed=int(inv.get("seed", 7)),
        prior_mean=float(inv.get("prior_mean", np.log(1.0e-12))),
        prior_std=float(inv.get("prior_std", knobs.get("prior_std", 0.8))),
        inflation=float(inv.get("inflation", knobs.get("inflation", 1.02))),
        algorithm=str(inv.get("algorithm", knobs.get("algorithm", "esmda"))),
        time_limit_s=None if inv.get("time_limit_s") is None else float(inv["time_limit_s"]),
        n_workers=None if inv.get("n_workers") is None else int(inv["n_workers"]),
        reconstruct_members=int(inv.get("reconstruct_members", 8)),
    )
    return DigitalTwin(grid, experiment, ports, physics, param, inverse=inverse)


def load_case(path: str | Path) -> DigitalTwin:
    path = Path(path)
    return build_twin(_load_yaml(path), cfg_dir=path.parent)
