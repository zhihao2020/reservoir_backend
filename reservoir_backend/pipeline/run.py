"""Orchestrate mesh build and four-field reconstruction for one time stamp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.pressure_field import reconstruct_pressure
from reservoir_backend.pipeline.property_field import invert_rock_properties
from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
from reservoir_backend.pipeline.state import (
    AxisAlignedBounds,
    BoundaryConditions,
    FieldBundle,
    MeshBundle,
    SensorSample,
    WellPoint,
)


def run_time_slice(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_prior_m2: float = 1.0e-13,
    porosity_prior: float = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    previous: FieldBundle | None = None,
    dt: float | None = None,
) -> FieldBundle:
    """Run pressure → saturation → property inversion for one sensor sample."""
    pressure, p_notes = reconstruct_pressure(
        mesh,
        sample,
        permeability_m2=permeability_prior_m2,
        viscosity_pa_s=viscosity_pa_s,
    )
    sw, so, sg, s_notes = reconstruct_saturation(mesh, sample)
    k, phi, r_notes = invert_rock_properties(
        mesh,
        pressure,
        sw,
        so,
        sg,
        viscosity_pa_s=viscosity_pa_s,
        permeability_prior_m2=permeability_prior_m2,
        porosity_prior=porosity_prior,
        pressure_prev=None if previous is None else previous.pressure,
        sw_prev=None if previous is None else previous.sw,
        dt=dt,
    )
    return FieldBundle(
        time=sample.time,
        pressure=pressure,
        sw=sw,
        so=so,
        sg=sg,
        permeability=k,
        porosity=phi,
        notes=p_notes + s_notes + r_notes,
    )


def save_fields(mesh: MeshBundle, fields: FieldBundle, output_dir: str | Path) -> Path:
    """Write mesh table and four fields under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = out / "mesh.csv"
    with mesh_path.open("w", encoding="utf-8") as fh:
        fh.write("cell_id,i,j,k,x,y,z\n")
        for n in range(mesh.n_cells):
            fh.write(
                f"{int(mesh.cell_id[n])},{int(mesh.i[n])},{int(mesh.j[n])},{int(mesh.k[n])},"
                f"{mesh.x[n]:.8g},{mesh.y[n]:.8g},{mesh.z[n]:.8g}\n"
            )
    np.save(out / "pressure.npy", fields.pressure)
    np.savez(
        out / "saturation.npz",
        sw=fields.sw,
        so=fields.so,
        sg=fields.sg,
    )
    np.savez(
        out / "properties.npz",
        permeability_m2=fields.permeability,
        porosity=fields.porosity,
    )
    summary = {
        "time": fields.time,
        "n_cells": mesh.n_cells,
        "pressure_min": float(np.min(fields.pressure)),
        "pressure_max": float(np.max(fields.pressure)),
        "sw_mean": float(np.mean(fields.sw)),
        "permeability_mean_m2": float(np.mean(fields.permeability)),
        "porosity_mean": float(np.mean(fields.porosity)),
        "notes": fields.notes,
        "well_cell_id": mesh.well_cell_id,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def load_sensor_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("sensor config must be a mapping")
    return data


def mesh_from_config(cfg: dict[str, Any]) -> MeshBundle:
    b = cfg["bounds"]
    bounds = AxisAlignedBounds(
        xmin=float(b["xmin"]),
        xmax=float(b["xmax"]),
        ymin=float(b["ymin"]),
        ymax=float(b["ymax"]),
        zmin=float(b["zmin"]),
        zmax=float(b["zmax"]),
    )
    wells = [
        WellPoint(name=str(w["name"]), x=float(w["x"]), y=float(w["y"]), z=float(w["z"]))
        for w in cfg.get("wells", [])
    ]
    g = cfg["grid"]
    return build_mesh(bounds, g["dx"], g["dy"], g["dz"], wells=wells)


def sample_from_config(cfg: dict[str, Any], time: float | None = None) -> SensorSample:
    sensors = cfg["sensors"]
    t = float(sensors.get("time", 0.0) if time is None else time)
    well_p = {str(k): float(v) for k, v in sensors.get("well_pressure", {}).items()}
    well_s: dict[str, tuple[float, float, float]] = {}
    for name, val in sensors.get("well_saturation", {}).items():
        if isinstance(val, (list, tuple)) and len(val) == 3:
            well_s[str(name)] = (float(val[0]), float(val[1]), float(val[2]))
        else:
            sw = float(val)
            well_s[str(name)] = (sw, max(0.0, 1.0 - sw), 0.0)
    boundary = BoundaryConditions(
        pressure={str(k): float(v) for k, v in sensors.get("boundary_pressure", {}).items()},
        flux={str(k): float(v) for k, v in sensors.get("boundary_flux", {}).items()},
    )
    return SensorSample(time=t, well_pressure=well_p, well_saturation=well_s, boundary=boundary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sensor four-field reconstruction")
    parser.add_argument("--config", required=True, help="YAML sensor case path")
    parser.add_argument("--output", default="results/sensor_run", help="output directory")
    parser.add_argument("--time", type=float, default=None, help="override sample time")
    args = parser.parse_args(argv)

    cfg = load_sensor_config(args.config)
    mesh = mesh_from_config(cfg)
    sample = sample_from_config(cfg, time=args.time)
    priors = cfg.get("priors", {})
    fields = run_time_slice(
        mesh,
        sample,
        permeability_prior_m2=float(priors.get("permeability_m2", 1.0e-13)),
        porosity_prior=float(priors.get("porosity", 0.2)),
        viscosity_pa_s=float(priors.get("viscosity_pa_s", 1.0e-3)),
    )
    out = save_fields(mesh, fields, args.output)
    print(json.dumps({"success": True, "output": str(out), "time": fields.time}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
