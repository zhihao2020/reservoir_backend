"""Load an engineering YAML case and convert to SI at the boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor
from reservoir_backend.io.grid_cfg import grid_from_cfg
from reservoir_backend.io.parameterization_cfg import parameterization_from_cfg
from reservoir_backend.io.pvt_cfg import pvt_from_cfg, pvt_preset_name
from reservoir_backend.physics.capillary import capillary_from_name
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.io.well_load import ports_from_cfg
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


def _csv_time(row: dict[str, str], default_unit: str | None) -> float:
    if row.get("time_s") not in (None, ""):
        return float(row["time_s"])
    if row.get("time") not in (None, ""):
        unit = str(row.get("time_unit") or default_unit or "s")
        return to_seconds(float(row["time"]), unit)
    raise ValueError("observation CSV row needs time_s or time")


def _read_observation_csv(
    path: Path,
    *,
    time_unit: str | None = None,
    pressure_unit: str | None = None,
) -> list[dict[str, Any]]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"observation CSV is empty: {path}")
    grouped: dict[tuple[str, str], list[tuple[float, float, float, bool]]] = {}
    for row in rows:
        sensor = str(row.get("sensor") or row.get("sensor_id") or row.get("well") or "").strip()
        if not sensor:
            continue
        kind = str(row.get("kind", "pressure") or "pressure")
        key = (sensor, kind)
        hold = str(row.get("holdout", "0")).strip() in {"1", "true", "True", "yes"}
        t = _csv_time(row, time_unit)
        value = float(row["value"])
        sigma = float(row.get("sigma", 1.0) or 1.0)
        unit = str(row.get("unit") or "").strip() or (pressure_unit if kind == "pressure" else None)
        if kind == "pressure" and unit:
            value = _maybe_convert(value, unit, "pressure")
            sigma = _maybe_convert(sigma, unit, "pressure")
        elif kind == "phase_rate" and unit:
            value = _maybe_convert(value, unit, "rate")
            sigma = _maybe_convert(sigma, unit, "rate")
        grouped.setdefault(key, []).append((t, value, sigma, hold))
    if not grouped:
        raise ValueError(f"observation CSV has no sensor rows: {path}")
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



def _read_sensors_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"sensors CSV is empty: {path}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        name = str(row.get("sensor_id") or row.get("name") or row.get("sensor") or f"S{i}").strip()
        if not name:
            continue
        kind = str(row.get("kind") or "saturation")
        sigma = row.get("sigma")
        if sigma in (None, ""):
            if row.get("sensor_id"):
                raise ValueError(f"sensor {name} is missing sigma")
            sigma = 2.0e3 if kind.lower() in {"p", "pressure"} else 0.04
        x = row.get("x_m", row.get("x"))
        y = row.get("y_m", row.get("y"))
        z = row.get("z_m", row.get("z"))
        if x in (None, "") or y in (None, "") or z in (None, ""):
            raise ValueError(f"sensor {name} needs x_m,y_m,z_m")
        medium = str(row.get("continuum") or row.get("medium") or "")
        if not medium:
            medium = "bulk" if kind.lower() not in {"p", "pressure", "bhp"} else "fracture"
        out.append(
            {
                "name": name,
                "kind": kind,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "volume_m3": float(row.get("volume_m3") or 0.0),
                "port": row.get("port") or None,
                "sigma": float(sigma),
                "medium": medium,
            }
        )
        probe = row.get("probe_diameter_m")
        if probe not in (None, ""):
            out[-1]["probe_diameter_m"] = float(probe)
    if not out:
        raise ValueError(f"sensors CSV has no named rows: {path}")
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


def build_twin(cfg: dict[str, Any], *, cfg_dir: str | Path = ".") -> DigitalTwin:
    cfg_dir = Path(cfg_dir)
    grid = grid_from_cfg(cfg, cfg_dir=cfg_dir)
    phys_cfg = cfg.get("physics") or {}
    model = str(phys_cfg.get("model", "two_phase_immiscible")).lower()
    dpdp = model in {"dpdp", "compositional_dpdp", "dual", "dual_compositional"}
    compositional = model in {"compositional", "comp", "eos"} or dpdp
    cap_name = phys_cfg.get("capillary", "brooks_corey")
    if cap_name is True:
        cap_name = "brooks_corey"
    if cap_name is False:
        cap_name = "none"
    capillary = capillary_from_name(str(cap_name))
    p_init = float(phys_cfg.get("p_init", 1.0e6))
    fluid = None
    if compositional:
        fluid_raw = phys_cfg.get("fluid", "example")
        from reservoir_backend.comp.fluid import CompSpec, fluid_from_name
        from reservoir_backend.io.eos_load import load_eos_card

        z_init = phys_cfg.get("z_init")
        z_inj = phys_cfg.get("z_inj")
        kwargs: dict = {}
        if phys_cfg.get("temperature_k") is not None:
            kwargs["temperature_k"] = float(phys_cfg["temperature_k"])
        if z_init is not None:
            kwargs["z_init"] = np.asarray(z_init, dtype=float)
        if z_inj is not None:
            kwargs["z_inj"] = np.asarray(z_inj, dtype=float)
        if phys_cfg.get("mu_liquid") is not None:
            kwargs["mu_liquid"] = float(phys_cfg["mu_liquid"])
        if phys_cfg.get("mu_vapor") is not None:
            kwargs["mu_vapor"] = float(phys_cfg["mu_vapor"])
        if str(phys_cfg.get("has_water", "")).lower() in {"1", "true", "yes", "on"} or phys_cfg.get("has_water") is True:
            kwargs["has_water"] = True
            kwargs["sw_init"] = float(phys_cfg.get("sw_init", 0.20))
        card_path = None
        preset = "example"
        if isinstance(fluid_raw, dict):
            raw_path = fluid_raw.get("file") or fluid_raw.get("gem_deck")
            if raw_path:
                card_path = Path(cfg_dir) / str(raw_path)
                if not card_path.is_file():
                    card_path = Path(raw_path)
                if not card_path.is_file():
                    raise ValueError(
                        f"fluid card {raw_path} not found; refuse invented Jiyang Tc/Pc"
                    )
            preset = str(fluid_raw.get("preset", "example"))
        else:
            preset = str(fluid_raw)
        if card_path is not None:
            eos = load_eos_card(card_path)
            if "z_init" not in kwargs:
                kwargs["z_init"] = np.full(eos.nc, 1.0 / eos.nc)
            if "z_inj" not in kwargs:
                z_inj0 = np.zeros(eos.nc)
                z_inj0[0] = 1.0
                kwargs["z_inj"] = z_inj0
            if card_path.suffix.lower() in {".yaml", ".yml"}:
                extra = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
                if extra.get("mu_liquid") is not None and "mu_liquid" not in kwargs:
                    kwargs["mu_liquid"] = float(extra["mu_liquid"])
                if extra.get("mu_vapor") is not None and "mu_vapor" not in kwargs:
                    kwargs["mu_vapor"] = float(extra["mu_vapor"])
                for src, dest in (
                    ("sorg", "sorg"),
                    ("sgr", "sgr"),
                    ("kro0", "kro0"),
                    ("krg0", "krg0"),
                    ("n_oil", "no"),
                    ("n_gas", "ng"),
                    ("no", "no"),
                    ("ng", "ng"),
                ):
                    if extra.get(src) is not None and dest not in kwargs:
                        kwargs[dest] = float(extra[src])
            fluid = CompSpec(eos=eos, **kwargs)
        else:
            fluid = fluid_from_name(preset, **kwargs)
        pvt = pvt_from_cfg({"pvt": "incompressible"}, p_init=p_init, model="two_phase_immiscible", cfg_dir=cfg_dir)
        relperm = CoreyTwoPhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o)
        three = None
        single = False
        implicit = True
        fully_implicit = False
    else:
        pvt = pvt_from_cfg(phys_cfg, p_init=p_init, model=model, cfg_dir=cfg_dir)
        relperm = CoreyTwoPhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o)
        three = (
            CoreyThreePhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o, mu_g=pvt.mu_g)
            if model in {"three_phase", "three_phase_immiscible", "c"}
            else None
        )
        single = model in {"single_phase", "single", "a"}
        default_transport = "explicit" if single else "implicit"
        transport = str(phys_cfg.get("transport", default_transport)).lower()
        implicit = transport in {"implicit", "impl", "true", "1", "on"}
        fim_raw = phys_cfg.get("fully_implicit", True)
        fully_implicit = str(fim_raw).lower() in {"1", "true", "yes", "on", "fim"} if not isinstance(fim_raw, bool) else bool(fim_raw)
        if fully_implicit:
            implicit = True
        if fully_implicit and three is None:
            from reservoir_backend.twin.offline import three_phase_for_fim
            three = three_phase_for_fim(relperm)
    physics = PhysicsSpec(
        relperm=relperm,
        three_phase=three,
        capillary=capillary,
        pvt=pvt,
        single_phase=single,
        sw_init=float(phys_cfg.get("sw_init", relperm.swi)),
        sg_init=float(phys_cfg.get("sg_init", 0.0)),
        p_init=p_init,
        dt_init=float(phys_cfg.get("dt_init", 5.0)),
        dt_min=float(phys_cfg.get("dt_min", 1.0e-6)),
        dt_max=float(phys_cfg.get("dt_max", 30.0)),
        max_cfl=float(phys_cfg.get("max_cfl", 0.5)),
        max_ds=float(phys_cfg.get("max_ds", 0.15)),
        max_steps=int(phys_cfg.get("max_steps", 12000)),
        implicit_transport=bool(implicit),
        fully_implicit=bool(fully_implicit),
        model="compositional_dpdp" if dpdp else ("compositional" if compositional else model),
        fluid=fluid,
        temperature_k=float(phys_cfg.get("temperature_k", 350.0)),
        z_init=None if phys_cfg.get("z_init") is None else np.asarray(phys_cfg.get("z_init"), dtype=float),
        shape_factor=float(
            (phys_cfg.get("transfer") or {}).get(
                "shape_factor_m2_inv", phys_cfg.get("shape_factor", 40.0)
            )
        ),
        phi_fracture=float(phys_cfg.get("phi_fracture", 0.02)),
        k_matrix_m2=(
            None
            if (phys_cfg.get("k_matrix_m2") is None and (cfg.get("rock") or {}).get("k_matrix_m2") is None)
            else float(phys_cfg.get("k_matrix_m2", (cfg.get("rock") or {}).get("k_matrix_m2")))
        ),
    )
    if not compositional:
        assert abs(float(physics.relperm.mu_o) - float(pvt.mu_o)) < 1.0e-15
        assert abs(float(physics.relperm.mu_w) - float(pvt.mu_w)) < 1.0e-15
        if pvt_preset_name(phys_cfg, model=model) == "cmg_seawater" and not pvt.has_live_oil():
            raise ValueError("physics.pvt=cmg_seawater must load live-oil tables")

    ports = ports_from_cfg(cfg, grid, cfg_dir=cfg_dir)

    defaults = cfg.get("sensors_defaults") or {}
    default_probe = float(defaults.get("probe_diameter_m", 0.0) or 0.0)
    sensors_src = cfg.get("sensors") or []
    if isinstance(sensors_src, str):
        sensors_src = _read_sensors_csv(Path(cfg_dir) / sensors_src)
    extra = cfg.get("sensors_extra") or []
    sensors: list[Sensor] = []
    for s in list(sensors_src) + list(extra):
        probe = s.get("probe_diameter_m", default_probe)
        sensors.append(
            Sensor(
                name=str(s["name"]),
                kind=str(s["kind"]),
                x=float(s["x"]),
                y=float(s["y"]),
                z=float(s["z"]),
                volume_m3=float(s.get("volume_m3", 0.0)),
                probe_diameter_m=float(probe or 0.0),
                port_name=s.get("port"),
                sigma=float(s.get("sigma", 1.0)),
                medium=str(s.get("medium", "bulk" if str(s["kind"]).lower() not in {"p", "pressure", "bhp"} else "fracture")),
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
        obs_src = _read_observation_csv(
            Path(cfg_dir) / obs_src,
            time_unit=exp_cfg.get("observation_time_unit"),
            pressure_unit=exp_cfg.get("observation_pressure_unit"),
        )
    for o in obs_src:
        kind = str(o.get("kind", "pressure"))
        unit = o.get("unit")
        conv = (
            "pressure"
            if kind in {"pressure", "bhp"}
            else "rate"
            if kind in {"phase_rate", "q_oil", "q_gas", "q_inj"}
            else None
        )
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

    known = {s.name for s in sensors}
    unknown = sorted({o.sensor_name for o in observations} - known)
    if unknown:
        raise ValueError(f"observations name sensors not in case: {unknown}")

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
    param = parameterization_from_cfg(grid, cfg, Path(cfg_dir))
    inverse = inverse_spec_from_cfg(inv)
    return DigitalTwin(grid, experiment, ports, physics, param, inverse=inverse)


_FORBIDDEN_INVERSE = frozenset(
    {
        "n_workers",
        "preset",
        "reconstruct_members",
    }
)

_CF_KINDS = frozenset(
    {
        "log_conductivity",
        "cf",
        "scalar_cf",
        "fracture_conductivity",
        "log_cf_tmf",
        "cf_tmf",
        "joint_cf_tmf",
    }
)


def inverse_spec_from_cfg(inv: dict[str, Any]) -> InverseSpec:
    bad = sorted(str(k) for k in inv if str(k) in _FORBIDDEN_INVERSE)
    if bad:
        raise ValueError("inverse keys not accepted: " + ", ".join(bad))
    pe = inv.get("post_ensemble") or {}
    pm = inv.get("prior_mean", np.log(1.0e-12))
    ps = inv.get("prior_std", 0.8)
    if isinstance(pm, list):
        pm = np.asarray(pm, dtype=float)
    if isinstance(ps, list):
        ps = np.asarray(ps, dtype=float)
    kind = str(inv.get("parameterization", "region")).lower()
    algo_raw = inv.get("algorithm")
    if algo_raw is None:
        algorithm = "esmda" if kind in _CF_KINDS else "lm"
    else:
        algorithm = str(algo_raw).strip().lower()
    if kind in {"log_cf_tmf", "cf_tmf", "joint_cf_tmf"}:
        if inv.get("prior_mean") is None:
            pm = np.array([0.0, 0.0], dtype=float)
        if inv.get("prior_std") is None:
            ps = np.array([0.8, 0.5], dtype=float)
    n_ens = inv.get("ensemble_size", inv.get("n_ensemble", 12))
    n_a = inv.get("assimilation_steps", inv.get("n_assimilations", 4))
    alpha = inv.get("alpha", inv.get("inflation"))
    return InverseSpec(
        prior_mean=pm,
        prior_std=ps,
        max_iter=int(inv.get("max_iter", 8)),
        fd_rel=float(inv.get("fd_rel", 0.05)),
        time_limit_s=None if inv.get("time_limit_s") is None else float(inv["time_limit_s"]),
        post_ensemble_enabled=bool(pe.get("enabled", False)),
        post_ensemble_ne=int(pe.get("ne", 8)),
        post_ensemble_seed=int(pe.get("seed", inv.get("seed", 0))),
        algorithm=algorithm,
        ensemble_size=int(n_ens),
        assimilation_steps=int(n_a),
        seed=int(inv.get("seed", 0)),
        alpha=None if alpha is None else list(alpha),
        clip_innovation=bool(inv.get("clip_innovation", False) or inv.get("robust_observations", False)),
        n_workers=None if inv.get("n_workers") is None else int(inv.get("n_workers")),
        outlier_nsigma=float(inv.get("outlier_nsigma", 8.0)),
    )


def load_case(path: str | Path) -> DigitalTwin:
    path = Path(path)
    cfg = _load_yaml(path)
    inv = cfg.get("inverse") or {}
    if inv.get("truth_json"):
        raise ValueError(
            "inverse.truth_json is not accepted; the shale IMEX loader was removed"
        )
    return build_twin(cfg, cfg_dir=path.parent)
