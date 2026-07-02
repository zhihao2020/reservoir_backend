"""Profile the Python full pipeline at several grid sizes."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

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
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.saturation_solver import advance_saturation_3d
from reservoir_backend.solver.velocity import compute_darcy_velocity


CASES = {
    "small": {"nx": 6, "ny": 5, "nz": 3, "time_steps": 3},
    "medium": {"nx": 20, "ny": 20, "nz": 5, "time_steps": 2},
    "large-lite": {"nx": 30, "ny": 30, "nz": 10, "time_steps": 1},
}


def profile_case(name: str, spec: dict[str, int], output_root: Path) -> dict[str, Any]:
    """Profile one deterministic pipeline case."""
    tracemalloc.start()
    start_total = time.perf_counter()
    timers = {
        "pressure_runtime_sec": 0.0,
        "flux_runtime_sec": 0.0,
        "saturation_runtime_sec": 0.0,
        "fusion_runtime_sec": 0.0,
        "result_export_runtime_sec": 0.0,
    }
    try:
        grid = Grid3D(nx=spec["nx"], ny=spec["ny"], nz=spec["nz"], dx=1.0, dy=1.0, dz=1.0)
        permeability = permeability_to_m2(100.0, "mD")
        relperm_params = {
            "swi": 0.2,
            "sor": 0.2,
            "krw0": 1.0,
            "kro0": 1.0,
            "nw": 2.0,
            "no": 2.0,
            "mu_w": 1.0e-3,
            "mu_o": 5.0e-3,
        }
        phi = Field3D.from_constant(grid, 0.2, name="porosity", unit="fraction")
        sw_initial = Field3D.from_constant(grid, 0.2, name="sw", unit="fraction")
        archie = ArchieInverter(swi=0.2, sor=0.2)
        true_sw = np.full(grid.shape, 0.35)
        rt = archie.forward_resistivity(true_sw, rw=0.25, phi=phi.values)
        sw_inv = archie.invert(Field3D(grid, rt, name="Rt"), rw=0.25, phi=phi)
        assert isinstance(sw_inv, Field3D)

        start = time.perf_counter()
        pressure = solve_steady_state_pressure_3d(
            grid,
            kx=permeability,
            ky=permeability,
            kz=permeability,
            mu=1.0e-3,
            dirichlet_boundaries={"left": 10.0e6, "right": 9.0e6},
        )
        timers["pressure_runtime_sec"] = time.perf_counter() - start

        start = time.perf_counter()
        velocity = compute_darcy_velocity(grid, pressure.pressure, permeability, permeability, permeability, 1.0e-3)
        timers["flux_runtime_sec"] = time.perf_counter() - start

        sw_sim = sw_initial
        last_report: dict[str, Any] = {}
        start = time.perf_counter()
        for _ in range(spec["time_steps"]):
            sat = advance_saturation_3d(
                grid,
                sw_sim,
                phi,
                velocity.face_fluxes.flux_x,
                velocity.face_fluxes.flux_y,
                velocity.face_fluxes.flux_z,
                dt=1000.0,
                relperm_params=relperm_params,
                max_cfl=1.0,
            )
            sw_sim = sat.sw
            last_report = sat.report
        timers["saturation_runtime_sec"] = time.perf_counter() - start

        start = time.perf_counter()
        sw_fused, fusion_report = fuse_saturation_fields([sw_inv, sw_sim], swi=0.2, sor=0.2)
        timers["fusion_runtime_sec"] = time.perf_counter() - start

        start = time.perf_counter()
        manager = ResultManager(output_root)
        manager.create_case_dir(name)
        manager.save_field("pressure", pressure.pressure)
        manager.save_field("sw_fused", sw_fused)
        manager.save_json("profile_report", {"case": name, "fusion": fusion_report})
        timers["result_export_runtime_sec"] = time.perf_counter() - start

        _, peak = tracemalloc.get_traced_memory()
        total_runtime = time.perf_counter() - start_total
        tracemalloc.stop()
        return {
            "case": name,
            "total_runtime_sec": total_runtime,
            **timers,
            "total_cells": grid.total_cells,
            "time_steps": spec["time_steps"],
            "max_cfl": last_report.get("max_cfl", 0.0),
            "material_balance_error": last_report.get("material_balance_error", 0.0),
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": True,
        }
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "case": name,
            "total_runtime_sec": time.perf_counter() - start_total,
            **timers,
            "total_cells": spec["nx"] * spec["ny"] * spec["nz"],
            "time_steps": spec["time_steps"],
            "max_cfl": None,
            "material_balance_error": None,
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": False,
            "error": str(exc),
        }


def run_profiling(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run all profiling cases and write summary reports."""
    report_dir = PROJECT_ROOT / "profiling_reports" if output_dir is None else Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_root = report_dir / "outputs"
    cases = [profile_case(name, spec, output_root) for name, spec in CASES.items()]
    summary = {
        "cases": cases,
        "success": all(case["success"] for case in cases),
    }
    json_path = report_dir / "performance_summary.json"
    md_path = report_dir / "performance_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_performance_markdown(summary), encoding="utf-8")
    return summary


def _performance_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Performance Summary", "", "| case | cells | total_runtime_sec | pressure | flux | saturation | fusion | export | success |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in summary["cases"]:
        lines.append(
            f"| {case['case']} | {case['total_cells']} | {case['total_runtime_sec']:.6f} | "
            f"{case['pressure_runtime_sec']:.6f} | {case['flux_runtime_sec']:.6f} | "
            f"{case['saturation_runtime_sec']:.6f} | {case['fusion_runtime_sec']:.6f} | "
            f"{case['result_export_runtime_sec']:.6f} | {case['success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    summary = run_profiling()
    print(f"profiling success={summary['success']}")


if __name__ == "__main__":
    main()
