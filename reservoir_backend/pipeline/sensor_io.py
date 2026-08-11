"""Load multi-time well / probe / boundary sensors from CSV for inversion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from reservoir_backend.pipeline.state import BoundaryConditions, SensorSample, WellPoint

# Long-format rows (one quantity set per name per time):
#   time,well,role,pressure_pa,sw,so,sg,rate_m3_s
#
# Rules:
#   - injector/producer: may fill p and/or S and optional rate
#   - observer_p: fill pressure_pa only (leave sw empty)
#   - observer_s: fill sw[,so,sg] only (leave pressure_pa empty)
#
# Boundary CSV:
#   time,side,pressure_pa[,flux_m3_s]


def load_well_series_csv(
    path: str | Path,
) -> tuple[list[SensorSample], dict[str, str], dict[str, tuple[float, float, float] | None]]:
    """Load long-format multi-time sensors.

    Returns
    -------
    samples :
        One ``SensorSample`` per distinct ``time`` (sorted).
    roles :
        ``name -> role`` from CSV ``role`` column (or inferred).
    locations :
        Optional ``name -> (x,y,z)`` if columns ``x,y,z`` present; else empty dict.
    """
    path = Path(path)
    rows = _read_csv_dicts(path)
    if not rows:
        raise ValueError(f"no rows in well series CSV: {path}")

    by_time: dict[float, dict[str, dict[str, Any]]] = {}
    roles: dict[str, str] = {}
    locations: dict[str, tuple[float, float, float]] = {}

    for row in rows:
        t = float(row["time"])
        name = str(row["well"]).strip()
        if not name:
            raise ValueError("well name must be non-empty")

        role = _normalize_role(row.get("role", ""), name=name, row=row)
        if name in roles and roles[name] != role:
            # allow first definition to win; warn via role consistency later
            pass
        else:
            roles[name] = role

        if row.get("x") not in (None, "") and row.get("y") not in (None, "") and row.get("z") not in (None, ""):
            locations[name] = (float(row["x"]), float(row["y"]), float(row["z"]))

        entry: dict[str, Any] = {}
        has_p = row.get("pressure_pa") not in (None, "")
        has_s = row.get("sw") not in (None, "")

        if role == "observer_p":
            if has_s:
                raise ValueError(f"t={t} {name}: observer_p cannot have saturation")
            if not has_p:
                raise ValueError(f"t={t} {name}: observer_p requires pressure_pa")
            if row.get("rate_m3_s") not in (None, ""):
                raise ValueError(f"t={t} {name}: observer_p cannot have rate")
        elif role == "observer_s":
            if has_p:
                raise ValueError(f"t={t} {name}: observer_s cannot have pressure_pa")
            if not has_s:
                raise ValueError(f"t={t} {name}: observer_s requires sw")
            if row.get("rate_m3_s") not in (None, ""):
                raise ValueError(f"t={t} {name}: observer_s cannot have rate")
        elif role == "observer":
            if has_p and has_s:
                raise ValueError(
                    f"t={t} {name}: a probe cannot measure both p and S; use observer_p or observer_s"
                )
            if not has_p and not has_s:
                raise ValueError(f"t={t} {name}: observer needs p or S")

        if has_p:
            entry["pressure"] = float(row["pressure_pa"])
        if has_s:
            sw = float(row["sw"])
            so = float(row["so"]) if row.get("so") not in (None, "") else max(0.0, 1.0 - sw)
            sg = float(row["sg"]) if row.get("sg") not in (None, "") else max(0.0, 1.0 - sw - so)
            entry["sw"], entry["so"], entry["sg"] = sw, so, sg
        if row.get("rate_m3_s") not in (None, "") and role in ("injector", "producer"):
            entry["rate"] = float(row["rate_m3_s"])

        by_time.setdefault(t, {})[name] = entry

    samples: list[SensorSample] = []
    for t in sorted(by_time):
        wells = by_time[t]
        well_p = {n: v["pressure"] for n, v in wells.items() if "pressure" in v}
        well_s = {
            n: (v["sw"], v["so"], v["sg"]) for n, v in wells.items() if "sw" in v
        }
        rates = {n: v["rate"] for n, v in wells.items() if "rate" in v}
        samples.append(
            SensorSample(
                time=t,
                well_pressure=well_p,
                well_saturation=well_s,
                boundary=BoundaryConditions(),
                well_rate=rates,
            )
        )
    return samples, roles, locations


def locations_to_well_points(
    roles: dict[str, str],
    locations: dict[str, tuple[float, float, float]],
    *,
    fallback_wells: list[WellPoint] | None = None,
) -> list[WellPoint]:
    """Build ``WellPoint`` list from CSV roles/locations, merging YAML wells."""
    pts: dict[str, WellPoint] = {}
    for w in fallback_wells or []:
        pts[w.name] = w
    for name, role in roles.items():
        if name in locations:
            x, y, z = locations[name]
            pts[name] = WellPoint(name=name, x=x, y=y, z=z, role=role)
        elif name in pts:
            # keep coordinates from YAML, update role if more specific
            old = pts[name]
            pts[name] = WellPoint(name=name, x=old.x, y=old.y, z=old.z, role=role)
    return list(pts.values())


def load_boundary_series_csv(
    path: str | Path,
) -> tuple[dict[float, dict[str, float]], dict[float, dict[str, float]]]:
    """Load boundary pressures and optional fluxes keyed by time then side."""
    path = Path(path)
    rows = _read_csv_dicts(path)
    p_out: dict[float, dict[str, float]] = {}
    q_out: dict[float, dict[str, float]] = {}
    for row in rows:
        t = float(row["time"])
        side = str(row["side"]).strip().lower()
        if row.get("pressure_pa") not in (None, ""):
            p_out.setdefault(t, {})[side] = float(row["pressure_pa"])
        if row.get("flux_m3_s") not in (None, ""):
            q_out.setdefault(t, {})[side] = float(row["flux_m3_s"])
    return p_out, q_out


def merge_boundary_series(
    samples: list[SensorSample],
    boundaries_by_time: dict[float, dict[str, float]],
    *,
    flux_by_time: dict[float, dict[str, float]] | None = None,
    tol: float = 1.0e-9,
) -> list[SensorSample]:
    """Attach boundary pressures/fluxes to samples with matching times."""
    if not boundaries_by_time and not flux_by_time:
        return samples
    p_times = sorted(boundaries_by_time) if boundaries_by_time else []
    q_times = sorted(flux_by_time) if flux_by_time else []
    merged: list[SensorSample] = []
    for s in samples:
        side_map = (
            _nearest_time_map(s.time, boundaries_by_time, p_times, tol=tol)
            if boundaries_by_time
            else {}
        )
        flux_map = (
            _nearest_time_map(s.time, flux_by_time, q_times, tol=tol)
            if flux_by_time
            else {}
        )
        if side_map is None:
            side_map = {}
        if flux_map is None:
            flux_map = {}
        merged.append(
            SensorSample(
                time=s.time,
                well_pressure=dict(s.well_pressure),
                well_saturation=dict(s.well_saturation),
                boundary=BoundaryConditions(
                    pressure=dict(side_map) if side_map else dict(s.boundary.pressure),
                    flux=dict(flux_map) if flux_map else dict(s.boundary.flux),
                ),
                well_rate=dict(s.well_rate or {}),
            )
        )
    return merged


def load_sensor_series(
    well_csv: str | Path,
    boundary_csv: str | Path | None = None,
) -> tuple[list[SensorSample], dict[str, str], dict[str, tuple[float, float, float]]]:
    """Wells/probes multi-time CSV + optional boundary CSV."""
    samples, roles, locations = load_well_series_csv(well_csv)
    if boundary_csv is not None and Path(boundary_csv).is_file():
        bmap, qmap = load_boundary_series_csv(boundary_csv)
        samples = merge_boundary_series(samples, bmap, flux_by_time=qmap)
    return samples, roles, locations


def write_well_series_csv(path: str | Path, samples: list[SensorSample], *, roles: dict[str, str] | None = None) -> Path:
    """Write long-format multi-time series (exclusive p/S for probes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roles = roles or {}
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["time", "well", "role", "pressure_pa", "sw", "so", "sg", "rate_m3_s"],
        )
        w.writeheader()
        for s in sorted(samples, key=lambda x: x.time):
            names = set(s.well_pressure) | set(s.well_saturation) | set(s.well_rate or {})
            for name in sorted(names):
                role = roles.get(name, "injector" if name in (s.well_rate or {}) else "observer")
                row = {
                    "time": s.time,
                    "well": name,
                    "role": role,
                    "pressure_pa": "",
                    "sw": "",
                    "so": "",
                    "sg": "",
                    "rate_m3_s": "",
                }
                if name in s.well_pressure:
                    row["pressure_pa"] = s.well_pressure[name]
                if name in s.well_saturation:
                    sw, so, sg = s.well_saturation[name]
                    row["sw"], row["so"], row["sg"] = sw, so, sg
                if name in (s.well_rate or {}):
                    row["rate_m3_s"] = s.well_rate[name]
                w.writerow(row)
    return path


def write_boundary_series_csv(path: str | Path, samples: list[SensorSample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "side", "pressure_pa", "flux_m3_s"])
        w.writeheader()
        for s in sorted(samples, key=lambda x: x.time):
            sides = set(s.boundary.pressure) | set(s.boundary.flux)
            for side in sorted(sides):
                row = {"time": s.time, "side": side, "pressure_pa": "", "flux_m3_s": ""}
                if side in s.boundary.pressure:
                    row["pressure_pa"] = s.boundary.pressure[side]
                if side in s.boundary.flux:
                    row["flux_m3_s"] = s.boundary.flux[side]
                w.writerow(row)
    return path


def samples_from_config_block(cfg: dict[str, Any]) -> list[SensorSample] | None:
    """If config has series CSV paths, load samples; else None."""
    series = cfg.get("series") or {}
    well_csv = cfg.get("sensors_csv") or series.get("wells_csv")
    if not well_csv:
        return None
    boundary_csv = cfg.get("boundary_csv") or series.get("boundary_csv")
    samples, _, _ = load_sensor_series(well_csv, boundary_csv)
    return samples


def load_series_bundle(cfg: dict[str, Any]) -> tuple[list[SensorSample], dict[str, str], dict[str, tuple[float, float, float]]]:
    """Load multi-time samples + roles + locations from config series block."""
    series = cfg.get("series") or {}
    well_csv = cfg.get("sensors_csv") or series.get("wells_csv")
    if not well_csv:
        raise ValueError("config series.wells_csv (or sensors_csv) is required")
    boundary_csv = cfg.get("boundary_csv") or series.get("boundary_csv")
    return load_sensor_series(well_csv, boundary_csv)


def _normalize_role(raw: str, *, name: str, row: dict[str, str]) -> str:
    role = str(raw or "").lower().strip()
    if role in ("inj", "injection", "injector"):
        return "injector"
    if role in ("prod", "production", "producer"):
        return "producer"
    if role in ("obs_p", "observer_p", "pressure_probe", "p_probe", "pressure_only"):
        return "observer_p"
    if role in ("obs_s", "observer_s", "saturation_probe", "s_probe", "sw_probe", "saturation_only"):
        return "observer_s"
    if role in ("obs", "observer", "probe", "monitor"):
        # infer from which columns filled
        has_p = row.get("pressure_pa") not in (None, "")
        has_s = row.get("sw") not in (None, "")
        if has_p and not has_s:
            return "observer_p"
        if has_s and not has_p:
            return "observer_s"
        return "observer"
    # infer if empty
    if not role:
        has_p = row.get("pressure_pa") not in (None, "")
        has_s = row.get("sw") not in (None, "")
        has_r = row.get("rate_m3_s") not in (None, "")
        if has_r or (has_p and has_s):
            # flowing well if rate or both p and s
            if has_r:
                q = float(row["rate_m3_s"])
                return "injector" if q >= 0 else "producer"
            return "injector" if "inj" in name.lower() else "producer"
        if has_p:
            return "observer_p"
        if has_s:
            return "observer_s"
    raise ValueError(f"cannot determine role for well {name}: role={raw!r}")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        field_map = {name: name.strip().lower() for name in reader.fieldnames}
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {
                field_map[k]: (v.strip() if isinstance(v, str) else v)
                for k, v in raw.items()
                if k is not None
            }
            if not any(row.values()):
                continue
            rows.append(row)
    return rows


def _nearest_time_map(
    t: float,
    by_time: dict[float, dict[str, float]],
    times: list[float],
    *,
    tol: float,
) -> dict[str, float] | None:
    if not times:
        return None
    if t in by_time:
        return by_time[t]
    best = min(times, key=lambda x: abs(x - t))
    if abs(best - t) <= max(tol, 1.0e-6) * max(1.0, abs(t)):
        return by_time[best]
    return by_time.get(best)
