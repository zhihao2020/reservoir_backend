"""Run a small full backend pipeline demo."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.units import permeability_to_m2
from reservoir_backend.fusion.field_fusion import fuse_saturation_fields
from reservoir_backend.inversion.resistivity_archie import ArchieInverter
from reservoir_backend.io.result_manager import ResultManager
from reservoir_backend.solver.capillary_flux import compute_capillary_fluxes
from reservoir_backend.solver.capillary_pressure import capillary_pressure
from reservoir_backend.solver.gravity_flux import compute_gravity_fluxes
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_3d,
    advance_saturation_3d_with_capillary,
    advance_saturation_3d_with_capillary_and_gravity,
    advance_saturation_3d_with_gravity,
)
from reservoir_backend.solver.velocity import compute_darcy_velocity
from examples.run_multisignal_inversion_demo import build_multisignal_inversion


REQUIRED_OUTPUTS = [
    "pressure.npy",
    "sw_inverted.npy",
    "sw_simulated.npy",
    "sw_fused.npy",
    "velocity_x.npy",
    "velocity_y.npy",
    "velocity_z.npy",
    "flux_x.npy",
    "flux_y.npy",
    "flux_z.npy",
    "production_curve.csv",
    "material_balance_report.json",
    "fusion_report.json",
    "solver_report.json",
    "case_summary.json",
]


def run_demo(
    case_id: str = "demo_case",
    results_root: str | Path | None = None,
    use_multisignal: bool = False,
    case_config: dict | None = None,
) -> dict[str, object]:
    """Run the full small backend pipeline and save outputs."""
    root = PROJECT_ROOT / "results" if results_root is None else Path(results_root)
    manager = ResultManager(root)
    case_dir = manager.create_case_dir(case_id)

    grid_config = {} if case_config is None else case_config["grid"]
    grid = Grid3D(
        nx=int(grid_config.get("nx", 6)),
        ny=int(grid_config.get("ny", 5)),
        nz=int(grid_config.get("nz", 3)),
        dx=float(grid_config.get("dx", 1.0)),
        dy=float(grid_config.get("dy", 1.0)),
        dz=float(grid_config.get("dz", 1.0)),
    )
    rock_config = {} if case_config is None else case_config["rock"]
    fluid_config = {} if case_config is None else case_config["fluid"]
    pressure_config = {} if case_config is None else case_config["pressure"]
    saturation_config = {} if case_config is None else case_config["saturation"]
    archie_config = {} if case_config is None else case_config["archie"]
    capillary_config = {} if case_config is None else case_config.get("capillary_pressure", {})
    gravity_config = {} if case_config is None else case_config.get("gravity", {})
    outputs_config = {} if case_config is None else case_config.get("outputs", {})
    initial_saturation_config = {} if case_config is None else case_config.get("initial_saturation", {})

    phi_value = float(rock_config.get("porosity", 0.2))
    permeability = float(rock_config.get("permeability_m2", permeability_to_m2(100.0, "mD")))
    mu = float(fluid_config.get("mu_w", 1.0e-3))
    swi = float(saturation_config.get("swi", archie_config.get("swi", 0.2)))
    sor = float(saturation_config.get("sor", archie_config.get("sor", 0.2)))
    relperm_params = {
        "swi": swi,
        "sor": sor,
        "krw0": float(saturation_config.get("krw0", 1.0)),
        "kro0": float(saturation_config.get("kro0", 1.0)),
        "nw": float(saturation_config.get("nw", 2.0)),
        "no": float(saturation_config.get("no", 2.0)),
        "mu_w": float(fluid_config.get("mu_w", 1.0e-3)),
        "mu_o": float(fluid_config.get("mu_o", 5.0e-3)),
        "injected_sw": float(saturation_config.get("injected_sw", 1.0 - sor)),
    }

    indices = np.indices(grid.shape)
    sw_true_values = np.clip(0.25 + 0.35 * indices[2] / (grid.nx - 1), swi, 1.0 - sor)
    phi = Field3D.from_constant(grid, phi_value, name="porosity", unit="fraction")
    sw_initial = build_initial_saturation_field(grid, initial_saturation_config, swi=swi, sor=sor)
    initial_saturation_type = str(initial_saturation_config.get("type", "constant"))

    multisignal_outputs = None
    if use_multisignal:
        multisignal_outputs = build_multisignal_inversion(grid, sw_true_values, swi=swi, sor=sor)
        sw_inverted = multisignal_outputs["sw_resistivity"]
        observed_sw = multisignal_outputs["sw_signal_fused"]
    else:
        archie = ArchieInverter(
            a=float(archie_config.get("a", 1.0)),
            m=float(archie_config.get("m", 2.0)),
            n=float(archie_config.get("n", 2.0)),
            swi=swi,
            sor=sor,
        )
        rt = archie.forward_resistivity(sw_true_values, rw=float(archie_config.get("rw", 0.25)), phi=phi.values)
        sw_inverted = archie.invert(Field3D(grid, rt, name="Rt", unit="ohm.m"), rw=float(archie_config.get("rw", 0.25)), phi=phi)
        assert isinstance(sw_inverted, Field3D)
        observed_sw = sw_inverted

    pressure_result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=permeability,
        ky=permeability,
        kz=permeability,
        mu=mu,
        dirichlet_boundaries={
            "left": float(pressure_config.get("left_pressure_pa", 10.0e6)),
            "right": float(pressure_config.get("right_pressure_pa", 9.0e6)),
        },
    )
    velocity_result = compute_darcy_velocity(
        grid=grid,
        pressure=pressure_result.pressure,
        kx=permeability,
        ky=permeability,
        kz=permeability,
        mu=mu,
    )

    sw_sim = sw_initial
    production_rows = []
    material_report = {}
    steps = int(saturation_config.get("steps", 3))
    dt = float(saturation_config.get("dt", 1000.0))
    max_cfl = float(saturation_config.get("max_cfl", 1.0))
    capillary_enabled = bool(capillary_config.get("enabled", False))
    capillary_model = str(capillary_config.get("model", "none"))
    gravity_enabled = bool(gravity_config.get("enabled", False))
    combined_transport_enabled = capillary_enabled and gravity_enabled
    last_saturation_report: dict[str, object] = {}
    for step in range(steps):
        if combined_transport_enabled:
            sat_result = advance_saturation_3d_with_capillary_and_gravity(
                grid=grid,
                sw=sw_sim,
                phi=phi,
                flux_x=velocity_result.face_fluxes.flux_x,
                flux_y=velocity_result.face_fluxes.flux_y,
                flux_z=velocity_result.face_fluxes.flux_z,
                dt=dt,
                relperm_params=relperm_params,
                capillary_params=capillary_config,
                gravity_params=gravity_config,
                kx=permeability,
                ky=permeability,
                kz=permeability,
                max_cfl=max_cfl,
            )
        elif capillary_enabled:
            sat_result = advance_saturation_3d_with_capillary(
                grid=grid,
                sw=sw_sim,
                phi=phi,
                flux_x=velocity_result.face_fluxes.flux_x,
                flux_y=velocity_result.face_fluxes.flux_y,
                flux_z=velocity_result.face_fluxes.flux_z,
                dt=dt,
                relperm_params=relperm_params,
                capillary_params=capillary_config,
                kx=permeability,
                ky=permeability,
                kz=permeability,
                max_cfl=max_cfl,
            )
        elif gravity_enabled:
            sat_result = advance_saturation_3d_with_gravity(
                grid=grid,
                sw=sw_sim,
                phi=phi,
                flux_x=velocity_result.face_fluxes.flux_x,
                flux_y=velocity_result.face_fluxes.flux_y,
                flux_z=velocity_result.face_fluxes.flux_z,
                dt=dt,
                relperm_params=relperm_params,
                gravity_params=gravity_config,
                kx=permeability,
                ky=permeability,
                kz=permeability,
                max_cfl=max_cfl,
            )
        else:
            sat_result = advance_saturation_3d(
                grid=grid,
                sw=sw_sim,
                phi=phi,
                flux_x=velocity_result.face_fluxes.flux_x,
                flux_y=velocity_result.face_fluxes.flux_y,
                flux_z=velocity_result.face_fluxes.flux_z,
                dt=dt,
                relperm_params=relperm_params,
                max_cfl=max_cfl,
            )
        sw_sim = sat_result.sw
        last_saturation_report = dict(sat_result.report)
        material_report = {
            "injected_water_volume": sat_result.report["injected_water_volume"],
            "produced_water_volume": sat_result.report["produced_water_volume"],
            "storage_change": sat_result.report["storage_change"],
            "material_balance_error": sat_result.report["material_balance_error"],
        }
        production_rows.append(
            {
                "step": step + 1,
                "time": (step + 1) * dt,
                "water_cut": sat_result.report["water_cut"],
                "injected_water_volume": sat_result.report["injected_water_volume"],
                "produced_water_volume": sat_result.report["produced_water_volume"],
                "storage_change": sat_result.report["storage_change"],
                "material_balance_error": sat_result.report["material_balance_error"],
                "max_cfl": sat_result.report["max_cfl"],
                **(
                    {
                        "max_abs_capillary_flux": sat_result.report["max_abs_capillary_flux"],
                        "max_total_water_flux": sat_result.report["max_total_water_flux"],
                    }
                    if capillary_enabled
                    else {}
                ),
                **(
                    {
                        "max_abs_gravity_flux": sat_result.report["max_abs_gravity_flux"],
                        "max_total_water_flux": sat_result.report["max_total_water_flux"],
                    }
                    if gravity_enabled
                    else {}
                ),
                **(
                    {
                        "combined_transport_enabled": True,
                        "max_effective_flux": sat_result.report["max_effective_flux"],
                    }
                    if combined_transport_enabled
                    else {}
                ),
            }
        )

    capillary_pressure_field = None
    capillary_report: dict[str, object] = {
        "capillary_enabled": False,
        "capillary_model": "none",
        "max_abs_capillary_flux": 0.0,
        "max_advective_flux": 0.0,
        "max_capillary_flux": 0.0,
        "max_total_water_flux": 0.0,
        "capillary_flux_included": False,
        "material_balance_error": float(material_report["material_balance_error"]),
    }
    if capillary_enabled:
        pc_result = capillary_pressure(sw_sim, capillary_config)
        assert isinstance(pc_result, Field3D)
        capillary_pressure_field = pc_result
        cap_flux_x, cap_flux_y, cap_flux_z, cap_flux_report = compute_capillary_fluxes(
            grid=grid,
            sw=sw_sim,
            kx=permeability,
            ky=permeability,
            kz=permeability,
            capillary_params=capillary_config,
            relperm_params=relperm_params,
        )
        capillary_report = {
            "capillary_enabled": True,
            "capillary_model": capillary_model,
            "max_abs_capillary_flux": float(last_saturation_report.get("max_abs_capillary_flux", cap_flux_report["max_abs_capillary_flux"])),
            "max_advective_flux": float(last_saturation_report.get("max_advective_flux", 0.0)),
            "max_capillary_flux": float(last_saturation_report.get("max_capillary_flux", cap_flux_report["max_abs_capillary_flux"])),
            "max_total_water_flux": float(last_saturation_report.get("max_total_water_flux", 0.0)),
            "capillary_flux_included": bool(last_saturation_report.get("capillary_flux_included", True)),
            "material_balance_error": float(material_report["material_balance_error"]),
            "pc_min": float(np.min(capillary_pressure_field.values)),
            "pc_max": float(np.max(capillary_pressure_field.values)),
            "flux_report": cap_flux_report,
        }

    gravity_report: dict[str, object] = {
        "gravity_enabled": False,
        "gravity_flux_included": False,
        "rho_w": float(gravity_config.get("rho_w", 1000.0)),
        "rho_o": float(gravity_config.get("rho_o", 800.0)),
        "density_difference": float(gravity_config.get("rho_w", 1000.0) - gravity_config.get("rho_o", 800.0)),
        "max_abs_gravity_flux": 0.0,
        "max_advective_flux": 0.0,
        "max_gravity_flux": 0.0,
        "max_total_water_flux": 0.0,
        "material_balance_error": float(material_report["material_balance_error"]),
        "max_cfl": float(production_rows[-1]["max_cfl"]),
    }
    grav_flux_x = grav_flux_y = grav_flux_z = None
    if gravity_enabled:
        grav_flux_x, grav_flux_y, grav_flux_z, grav_flux_report = compute_gravity_fluxes(
            grid=grid,
            sw=sw_sim,
            kx=permeability,
            ky=permeability,
            kz=permeability,
            gravity_params=gravity_config,
            relperm_params=relperm_params,
        )
        gravity_report = {
            "gravity_enabled": True,
            "gravity_flux_included": bool(last_saturation_report.get("gravity_flux_included", True)),
            "rho_w": float(last_saturation_report.get("rho_w", gravity_config.get("rho_w", 1000.0))),
            "rho_o": float(last_saturation_report.get("rho_o", gravity_config.get("rho_o", 800.0))),
            "density_difference": float(
                last_saturation_report.get(
                    "density_difference",
                    gravity_config.get("rho_w", 1000.0) - gravity_config.get("rho_o", 800.0),
                )
            ),
            "max_abs_gravity_flux": float(
                last_saturation_report.get("max_abs_gravity_flux", grav_flux_report["max_abs_gravity_flux"])
            ),
            "max_advective_flux": float(last_saturation_report.get("max_advective_flux", 0.0)),
            "max_gravity_flux": float(last_saturation_report.get("max_gravity_flux", grav_flux_report["max_abs_gravity_flux"])),
            "max_total_water_flux": float(last_saturation_report.get("max_total_water_flux", 0.0)),
            "material_balance_error": float(material_report["material_balance_error"]),
            "max_cfl": float(production_rows[-1]["max_cfl"]),
            "flux_report": grav_flux_report,
        }

    combined_report: dict[str, object] = {
        "combined_transport_enabled": combined_transport_enabled,
        "capillary_enabled": capillary_enabled,
        "gravity_enabled": gravity_enabled,
        "capillary_model": capillary_model if capillary_enabled else "none",
        "rho_w": float(gravity_report["rho_w"]),
        "rho_o": float(gravity_report["rho_o"]),
        "density_difference": float(gravity_report["density_difference"]),
        "max_advective_flux": float(last_saturation_report.get("max_advective_flux", 0.0)),
        "max_capillary_flux": float(last_saturation_report.get("max_capillary_flux", capillary_report["max_capillary_flux"])),
        "max_gravity_flux": float(last_saturation_report.get("max_gravity_flux", gravity_report["max_gravity_flux"])),
        "max_abs_capillary_flux": float(last_saturation_report.get("max_abs_capillary_flux", capillary_report["max_abs_capillary_flux"])),
        "max_abs_gravity_flux": float(last_saturation_report.get("max_abs_gravity_flux", gravity_report["max_abs_gravity_flux"])),
        "max_total_water_flux": float(last_saturation_report.get("max_total_water_flux", 0.0)),
        "max_effective_flux": float(last_saturation_report.get("max_effective_flux", 0.0)),
        "max_cfl": float(production_rows[-1]["max_cfl"]),
        "material_balance_error": float(material_report["material_balance_error"]),
        "capillary_flux_included": bool(last_saturation_report.get("capillary_flux_included", capillary_enabled)),
        "gravity_flux_included": bool(last_saturation_report.get("gravity_flux_included", gravity_enabled)),
        "has_nan": bool(last_saturation_report.get("has_nan", False)),
        "has_inf": bool(last_saturation_report.get("has_inf", False)),
        "composer_report": last_saturation_report.get("composer_report", {}),
    }

    sw_fused, fusion_report = fuse_saturation_fields(
        [observed_sw, sw_sim],
        confidence_fields=[
            Field3D.from_constant(grid, 0.8, name="inv_conf"),
            Field3D.from_constant(grid, 0.6, name="sim_conf"),
        ],
        swi=swi,
        sor=sor,
    )

    manager.save_field("pressure", pressure_result.pressure)
    manager.save_field("sw_inverted", sw_inverted)
    if initial_saturation_type != "constant":
        manager.save_field("initial_saturation", sw_initial)
    if multisignal_outputs is not None:
        manager.save_field("sw_resistivity", multisignal_outputs["sw_resistivity"])
        manager.save_field("sw_em", multisignal_outputs["sw_em"])
        manager.save_field("sw_acoustic", multisignal_outputs["sw_acoustic"])
        manager.save_field("sw_signal_fused", multisignal_outputs["sw_signal_fused"])
    manager.save_field("sw_simulated", sw_sim)
    manager.save_field("sw_fused", sw_fused)
    manager.save_field("velocity_x", velocity_result.velocity_x)
    manager.save_field("velocity_y", velocity_result.velocity_y)
    manager.save_field("velocity_z", velocity_result.velocity_z)
    manager.save_npy("flux_x", velocity_result.face_fluxes.flux_x)
    manager.save_npy("flux_y", velocity_result.face_fluxes.flux_y)
    manager.save_npy("flux_z", velocity_result.face_fluxes.flux_z)
    if capillary_enabled and bool(outputs_config.get("save_capillary_pressure", True)):
        assert capillary_pressure_field is not None
        manager.save_field("capillary_pressure", capillary_pressure_field)
    if capillary_enabled and bool(outputs_config.get("save_capillary_flux", True)):
        manager.save_npy("capillary_flux_x", cap_flux_x)
        manager.save_npy("capillary_flux_y", cap_flux_y)
        manager.save_npy("capillary_flux_z", cap_flux_z)
    if capillary_enabled and bool(outputs_config.get("save_reports", True)):
        manager.save_json("capillary_report", capillary_report)
    if gravity_enabled and bool(outputs_config.get("save_gravity_flux", True)):
        assert grav_flux_x is not None and grav_flux_y is not None and grav_flux_z is not None
        manager.save_npy("gravity_flux_x", grav_flux_x)
        manager.save_npy("gravity_flux_y", grav_flux_y)
        manager.save_npy("gravity_flux_z", grav_flux_z)
    if gravity_enabled and bool(outputs_config.get("save_reports", True)):
        manager.save_json("gravity_report", gravity_report)
    if combined_transport_enabled and bool(outputs_config.get("save_reports", True)) and bool(outputs_config.get("save_combined_report", True)):
        manager.save_json("combined_report", combined_report)
    manager.save_csv("production_curve", production_rows)
    manager.save_json("material_balance_report", material_report)
    manager.save_json("fusion_report", fusion_report)
    solver_report = {
        "pressure": pressure_result.report,
        "velocity": velocity_result.report,
        "saturation_last_step": production_rows[-1],
    }
    manager.save_json("solver_report", solver_report)

    outputs = [path.name for path in manager.list_case_outputs(case_id)]
    summary = {
        "case_id": case_id,
        "grid_shape": list(grid.shape),
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "total_cells": grid.total_cells,
        "modules_used": [
            "ArchieInverter",
            *(["ElectromagneticInverter", "AcousticInverter", "SignalFusion"] if use_multisignal else []),
            "PressureSolver3D",
            "DarcyVelocity",
            "SaturationSolver3D",
            *(["CapillaryPressure", "CapillaryFlux", "CapillarySaturation3D"] if capillary_enabled else []),
            *(["GravityFlux", "GravitySaturation3D"] if gravity_enabled else []),
            *(["CombinedCapillaryGravitySaturation3D"] if combined_transport_enabled else []),
            "FieldFusion",
            "ResultManager",
        ],
        "pressure_min": float(np.min(pressure_result.pressure.values)),
        "pressure_max": float(np.max(pressure_result.pressure.values)),
        "sw_inverted_min": float(np.min(sw_inverted.values)),
        "sw_inverted_max": float(np.max(sw_inverted.values)),
        "sw_simulated_min": float(np.min(sw_sim.values)),
        "sw_simulated_max": float(np.max(sw_sim.values)),
        "sw_fused_min": float(np.nanmin(sw_fused.values)),
        "sw_fused_max": float(np.nanmax(sw_fused.values)),
        "use_multisignal": use_multisignal,
        "max_cfl": float(production_rows[-1]["max_cfl"]),
        "material_balance_error": float(material_report["material_balance_error"]),
        "fusion_nan_cells": int(fusion_report["nan_cells_count"]),
        "fusion_clipped_cells": int(fusion_report["clipped_cells"]),
        "capillary_enabled": capillary_enabled,
        "combined_transport_enabled": combined_transport_enabled,
        "capillary_model": capillary_model if capillary_enabled else "none",
        "capillary_pressure_min": None if capillary_pressure_field is None else float(np.min(capillary_pressure_field.values)),
        "capillary_pressure_max": None if capillary_pressure_field is None else float(np.max(capillary_pressure_field.values)),
        "max_abs_capillary_flux": float(combined_report["max_abs_capillary_flux"]),
        "max_capillary_flux": float(combined_report["max_capillary_flux"]),
        "max_total_water_flux": float(combined_report["max_total_water_flux"]),
        "max_effective_flux": float(combined_report["max_effective_flux"]),
        "capillary_flux_included": bool(combined_report["capillary_flux_included"]),
        "initial_saturation_type": initial_saturation_type,
        "output_files": sorted(set(outputs + ["case_summary.json"])),
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    gravity_summary = {
        "gravity_enabled": gravity_enabled,
        "rho_w": float(gravity_report["rho_w"]),
        "rho_o": float(gravity_report["rho_o"]),
        "density_difference": float(gravity_report["density_difference"]),
        "max_abs_gravity_flux": float(combined_report["max_abs_gravity_flux"]),
        "gravity_flux_included": bool(combined_report["gravity_flux_included"]),
    }
    if gravity_enabled:
        summary.update(gravity_summary)
    case_summary = {**summary, **gravity_summary}
    manager.save_case_summary(case_summary)
    manager.validate_required_outputs(case_id, REQUIRED_OUTPUTS)
    return {"case_id": case_id, "case_dir": case_dir, "summary": summary}


def build_initial_saturation_field(
    grid: Grid3D,
    config: dict | None,
    *,
    swi: float,
    sor: float,
) -> Field3D:
    """Build a deterministic initial saturation field from config."""
    cfg = {} if config is None else config
    initial_type = str(cfg.get("type", "constant"))
    lower, upper = float(swi), 1.0 - float(sor)
    if initial_type == "constant":
        values = np.full(grid.shape, float(cfg.get("value", swi)), dtype=float)
    elif initial_type == "step_x":
        low_sw = float(cfg.get("low_sw", swi))
        high_sw = float(cfg.get("high_sw", upper))
        split_fraction = float(cfg.get("split_fraction", 0.5))
        split_index = min(max(int(round(grid.nx * split_fraction)), 1), grid.nx - 1)
        values = np.full(grid.shape, low_sw, dtype=float)
        values[:, :, :split_index] = high_sw
    elif initial_type == "linear_x":
        left_sw = float(cfg.get("left_sw", upper))
        right_sw = float(cfg.get("right_sw", lower))
        line = np.linspace(left_sw, right_sw, grid.nx, dtype=float).reshape(1, 1, grid.nx)
        values = np.broadcast_to(line, grid.shape).copy()
    elif initial_type == "center_blob":
        background_sw = float(cfg.get("background_sw", lower))
        blob_sw = float(cfg.get("blob_sw", upper))
        radius_fraction = float(cfg.get("radius_fraction", 0.25))
        coords = np.indices(grid.shape, dtype=float)
        center = np.array([(grid.nz - 1) / 2.0, (grid.ny - 1) / 2.0, (grid.nx - 1) / 2.0]).reshape(3, 1, 1, 1)
        scale = np.array([max(grid.nz - 1, 1), max(grid.ny - 1, 1), max(grid.nx - 1, 1)]).reshape(3, 1, 1, 1)
        distance = np.sqrt(np.sum(((coords - center) / scale) ** 2, axis=0))
        values = np.full(grid.shape, background_sw, dtype=float)
        values[distance <= radius_fraction] = blob_sw
    else:
        raise ValueError("initial_saturation.type must be constant, step_x, linear_x, or center_blob")
    values = np.clip(values, lower, upper)
    return Field3D(grid=grid, values=values, name="sw_initial", unit="fraction")


def main() -> None:
    """CLI entry point."""
    result = run_demo()
    print(result["case_dir"])


if __name__ == "__main__":
    main()
