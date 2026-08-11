"""Load multi-time well / boundary sensors from CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from reservoir_backend.pipeline.state import BoundaryConditions, SensorSample

# Expected long-format well rows:
#   time,well,pressure_pa,sw,so,sg[,rate_m3_s]
# rate_m3_s: signed volumetric rate (+inject, -produce), optional
# Optional boundary CSV:
#   time,side,pressure_pa[,flux_m3_s]
# flux_m3_s: net volumetric rate into domain (optional)


def load_well_series_csv(path: str | Path) -> list[SensorSample]:
    """Load long-format well sensors and group into ``SensorSample`` list.

    Required columns: ``time``, ``well``, ``pressure_pa``.
    Saturation columns ``sw``, ``so``, ``sg`` optional (default oil fill).
    """
    path = Path(path)
    rows = _read_csv_dicts(path)
    if not rows:
        raise ValueError(f"no rows in well series CSV: {path}")

    by_time: dict[float, dict[str, dict[str, float]]] = {}
    for row in rows:
        t = float(row["time"])
        name = str(row["well"]).strip()
        if not name:
            raise ValueError("well name must be non-empty")
        if "pressure_pa" not in row or row["pressure_pa"] == "":
            raise ValueError(f"missing pressure_pa for well {name} at t={t}")
        p = float(row["pressure_pa"])
        sw = float(row["sw"]) if row.get("sw") not in (None, "") else 0.3
        so = float(row["so"]) if row.get("so") not in (None, "") else max(0.0, 1.0 - sw)
        sg = float(row["sg"]) if row.get("sg") not in (None, "") else max(0.0, 1.0 - sw - so)
        entry: dict[str, float] = {
            "pressure": p,
            "sw": sw,
            "so": so,
            "sg": sg,
        }
        if row.get("rate_m3_s") not in (None, ""):
            entry["rate"] = float(row["rate_m3_s"])
        by_time.setdefault(t, {})[name] = entry

    samples: list[SensorSample] = []
    for t in sorted(by_time):
        wells = by_time[t]
        rates = {n: v["rate"] for n, v in wells.items() if "rate" in v}
        samples.append(
            SensorSample(
                time=t,
                well_pressure={n: v["pressure"] for n, v in wells.items()},
                well_saturation={
                    n: (v["sw"], v["so"], v["sg"]) for n, v in wells.items()
                },
                boundary=BoundaryConditions(),
                well_rate=rates,
            )
        )
    return samples


def load_boundary_series_csv(
    path: str | Path,
) -> tuple[dict[float, dict[str, float]], dict[float, dict[str, float]]]:
    """Load boundary pressures and optional fluxes keyed by time then side.

    Returns ``(pressure_by_time, flux_by_time)``.
    """
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
            )
        )
    return merged


def load_sensor_series(
    well_csv: str | Path,
    boundary_csv: str | Path | None = None,
) -> list[SensorSample]:
    """Convenience: wells CSV + optional boundary CSV → sample list."""
    samples = load_well_series_csv(well_csv)
    if boundary_csv is not None and Path(boundary_csv).is_file():
        bmap, qmap = load_boundary_series_csv(boundary_csv)
        samples = merge_boundary_series(samples, bmap, flux_by_time=qmap)
    return samples


def write_well_series_csv(path: str | Path, samples: list[SensorSample]) -> Path:
    """Write long-format well series (for fixtures / export)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["time", "well", "pressure_pa", "sw", "so", "sg", "rate_m3_s"],
        )
        w.writeheader()
        for s in sorted(samples, key=lambda x: x.time):
            for name, p in s.well_pressure.items():
                sw, so, sg = s.well_saturation.get(name, (0.3, 0.7, 0.0))
                rate = ""
                if name in (s.well_rate or {}):
                    rate = s.well_rate[name]
                w.writerow(
                    {
                        "time": s.time,
                        "well": name,
                        "pressure_pa": p,
                        "sw": sw,
                        "so": so,
                        "sg": sg,
                        "rate_m3_s": rate,
                    }
                )
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
    """If config has ``sensors_csv`` / ``series``, load samples; else None."""
    series = cfg.get("series") or {}
    well_csv = cfg.get("sensors_csv") or series.get("wells_csv")
    if not well_csv:
        return None
    boundary_csv = cfg.get("boundary_csv") or series.get("boundary_csv")
    return load_sensor_series(well_csv, boundary_csv)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        # normalize keys
        field_map = {name: name.strip().lower() for name in reader.fieldnames}
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {field_map[k]: (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k is not None}
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
    if t in by_time:
        return by_time[t]
    # exact float match failed — nearest within tol
    best = min(times, key=lambda x: abs(x - t))
    if abs(best - t) <= tol * max(1.0, abs(t)):
        return by_time[best]
    # also allow absolute day-scale tol
    if abs(best - t) <= max(tol, 1.0e-6):
        return by_time[best]
    return by_time.get(best)  # soft attach nearest if only one boundary table
