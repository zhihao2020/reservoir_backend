"""Orchestrate mesh build and four-field reconstruction for one time stamp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

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
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    previous: FieldBundle | None = None,
    dt: float | None = None,
    n_k_iterations: int = 2,
) -> FieldBundle:
    """软件要求 steps 2–4 at one time ``t``.

    2. well P + boundary P/flux → pressure field  
    3. well S + boundary flux cues → sw, so, sg  
    4. p, S, flux → heterogeneous k, φ (Darcy + continuity)

    When ``n_k_iterations > 1``, k is fed back into the TPFA pressure solve
    (array prior — suitable for non-uniform rock).
    """
    if previous is not None:
        k_work: float | NDArray[np.float64] = previous.permeability
        phi_work: float | NDArray[np.float64] = previous.porosity
    else:
        k_work = permeability_prior_m2
        phi_work = porosity_prior

    iters = max(1, int(n_k_iterations))
    p_notes: list[str] = []
    r_notes: list[str] = []
    s_notes: list[str] = []
    pressure = np.zeros(mesh.grid.shape, dtype=float)
    k = np.zeros(mesh.grid.shape, dtype=float)
    phi = np.zeros(mesh.grid.shape, dtype=float)
    sw = so = sg = np.zeros(mesh.grid.shape, dtype=float)
    flux_dict: dict[str, NDArray[np.float64]] = {}

    for it in range(iters):
        # Step 2
        pressure, p_notes = reconstruct_pressure(
            mesh,
            sample,
            permeability_m2=k_work,
            viscosity_pa_s=viscosity_pa_s,
        )
        # Step 3 (use latest pressure for flow-aligned IDW)
        sw, so, sg, s_notes = reconstruct_saturation(mesh, sample, pressure=pressure)
        # Step 4
        k, phi, r_notes, flux_dict = invert_rock_properties(
            mesh,
            pressure,
            sw,
            so,
            sg,
            viscosity_pa_s=viscosity_pa_s,
            permeability_prior_m2=k_work,
            porosity_prior=phi_work,
            pressure_prev=None if previous is None else previous.pressure,
            sw_prev=None if previous is None else previous.sw,
            dt=dt,
        )
        k_work = k
        if it == 0:
            phi_work = phi

    notes = (
        ["four-field steps: pressure → saturation → rock (k,phi)"]
        + p_notes
        + s_notes
        + r_notes
        + [f"k-pressure fixed-point iterations={iters}"]
    )
    return FieldBundle(
        time=sample.time,
        pressure=pressure,
        sw=sw,
        so=so,
        sg=sg,
        permeability=k,
        porosity=phi,
        notes=notes,
        flux_x=flux_dict.get("flux_x"),
        flux_y=flux_dict.get("flux_y"),
        flux_z=flux_dict.get("flux_z"),
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
    if fields.flux_x is not None:
        np.savez(
            out / "fluxes.npz",
            flux_x=fields.flux_x,
            flux_y=fields.flux_y,
            flux_z=fields.flux_z,
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
    parser.add_argument(
        "--mode",
        choices=("slice", "series", "discovery", "esmda"),
        default="slice",
        help="slice=single time; series=CSV/multi-time; discovery=shape; esmda=ensemble k",
    )
    parser.add_argument("--wells-csv", default=None, help="long-format well sensor CSV")
    parser.add_argument("--boundary-csv", default=None, help="boundary pressure CSV")
    parser.add_argument("--ne", type=int, default=None, help="ES-MDA ensemble size")
    parser.add_argument("--na", type=int, default=None, help="ES-MDA assimilation steps per time")
    args = parser.parse_args(argv)

    cfg = load_sensor_config(args.config)
    mesh = mesh_from_config(cfg)
    priors = cfg.get("priors", {})
    k0 = float(priors.get("permeability_m2", 1.0e-13))
    phi0 = float(priors.get("porosity", 0.2))
    mu = float(priors.get("viscosity_pa_s", 1.0e-3))
    out_root = Path(args.output)

    # Resolve samples for multi-time modes
    samples = _resolve_samples(cfg, args)

    if args.mode == "slice":
        if samples:
            sample = samples[0] if args.time is None else min(samples, key=lambda s: abs(s.time - args.time))
        else:
            sample = sample_from_config(cfg, time=args.time)
        fields = run_time_slice(
            mesh,
            sample,
            permeability_prior_m2=k0,
            porosity_prior=phi0,
            viscosity_pa_s=mu,
        )
        out = save_fields(mesh, fields, out_root)
        print(json.dumps({"success": True, "mode": "slice", "output": str(out), "time": fields.time}, indent=2))
        return 0

    if not samples:
        raise SystemExit("mode requires multi-time sensors (series in config or --wells-csv)")

    if args.mode == "series":
        from reservoir_backend.pipeline.time_series import run_time_series

        history = run_time_series(
            mesh,
            samples,
            permeability_prior_m2=k0,
            porosity_prior=phi0,
            viscosity_pa_s=mu,
        )
        for i, fb in enumerate(history):
            save_fields(mesh, fb, out_root / f"t_{i:04d}")
        summary = {
            "success": True,
            "mode": "series",
            "n_times": len(history),
            "times": [h.time for h in history],
            "output": str(out_root),
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "series_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if args.mode == "discovery":
        from reservoir_backend.pipeline.time_series import run_shape_discovery, save_discovery

        result = run_shape_discovery(
            mesh,
            samples,
            permeability_prior_m2=k0,
            porosity_prior=phi0,
            viscosity_pa_s=mu,
        )
        save_discovery(result, str(out_root))
        print(
            json.dumps(
                {
                    "success": True,
                    "mode": "discovery",
                    "output": str(out_root),
                    "indicator_stats": result.indicator_stats,
                    "notes": result.notes,
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "esmda":
        from reservoir_backend.pipeline.esmda import run_esmda_permeability

        es = cfg.get("esmda", {})
        ne = int(args.ne if args.ne is not None else es.get("ne", 20))
        na = int(args.na if args.na is not None else es.get("n_assimilations", 4))
        loc_r = es.get("localization_radius_m", None)
        result = run_esmda_permeability(
            mesh,
            samples,
            ne=ne,
            n_assimilations=na,
            k_mean=float(es.get("k_mean", k0)),
            logk_std=float(es.get("logk_std", 1.0)),
            corr_len_cells=float(es.get("corr_len_cells", 3.0)),
            obs_std_frac=float(es.get("obs_std_frac", 0.02)),
            porosity_prior=phi0,
            viscosity_pa_s=mu,
            seed=int(es.get("seed", 42)),
            localization_radius_m=None if loc_r is None else float(loc_r),
            ensemble_inflation=float(es.get("ensemble_inflation", 1.02)),
        )
        out_root.mkdir(parents=True, exist_ok=True)
        np.save(out_root / "k_mean.npy", result.k_mean)
        np.save(out_root / "k_std.npy", result.k_std)
        np.save(out_root / "k_ensemble.npy", result.k_ensemble)
        for i, fb in enumerate(result.history_mean):
            save_fields(mesh, fb, out_root / "mean_history" / f"t_{i:04d}")
        report = {
            "success": True,
            "mode": "esmda",
            "output": str(out_root),
            "ne": ne,
            "n_assimilations": na,
            "k_mean_avg": float(np.mean(result.k_mean)),
            "k_std_avg": float(np.mean(result.k_std)),
            "observation_rmse": result.observation_rmse,
            "notes": result.notes,
        }
        (out_root / "esmda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    raise SystemExit(f"unknown mode: {args.mode}")


def _resolve_samples(cfg: dict[str, Any], args: argparse.Namespace) -> list[SensorSample]:
    from reservoir_backend.pipeline.sensor_io import load_sensor_series, samples_from_config_block

    if args.wells_csv:
        return load_sensor_series(args.wells_csv, args.boundary_csv)
    loaded = samples_from_config_block(cfg)
    if loaded is not None:
        return loaded
    # inline multi-time list under sensors_series
    if "sensors_series" in cfg and isinstance(cfg["sensors_series"], list):
        from copy import deepcopy

        samples = []
        base = cfg
        for block in cfg["sensors_series"]:
            tmp = deepcopy(base)
            tmp["sensors"] = block
            samples.append(sample_from_config(tmp))
        return samples
    return []


if __name__ == "__main__":
    raise SystemExit(main())
