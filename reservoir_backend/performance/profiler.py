"""Lightweight performance profiling for existing Python/NumPy/SciPy paths.

The profiler measures current implementations without changing numerical
algorithms, benchmark runners, or solver behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
import tracemalloc
from typing import Any

import numpy as np

from benchmarks.benchmark_registry import build_benchmark_registry
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.cross_scale.runner import (
    run_lab_field_validation_report,
    run_scale_effect_report,
    run_similarity_report,
)
from reservoir_backend.fusion.field_fusion import weighted_average_fields
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.saturation_solver import DEFAULT_RELPERM_PARAMS, advance_saturation_1d


@dataclass(frozen=True)
class SyntheticCase:
    """Synthetic case descriptor used by the performance baseline."""

    case_id: str
    grid_shape: tuple[int, int, int]
    saturation_nx: int

    @property
    def total_cells(self) -> int:
        nz, ny, nx = self.grid_shape
        return int(nx * ny * nz)


SYNTHETIC_CASES = [
    SyntheticCase("small", (3, 4, 8), 64),
    SyntheticCase("medium", (4, 8, 12), 192),
    SyntheticCase("large", (5, 10, 16), 384),
]


def measure_runtime(stage_name: str, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Measure runtime and Python allocation peak for a callable stage."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
        success = bool(result.get("success", True))
        error = None
    except Exception as exc:  # pragma: no cover - exercised through report failure path
        result = {}
        success = False
        error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "stage_name": stage_name,
        "success": success,
        "runtime_sec": float(runtime),
        "memory_peak_mb": float(peak / (1024.0 * 1024.0)),
        "error": error,
        **_jsonable(result),
    }


def profile_synthetic_case(case: SyntheticCase) -> dict[str, Any]:
    """Profile pressure, saturation, fusion, cross-scale, and registry stages."""
    stages = [
        measure_runtime("pressure", lambda: _pressure_stage(case)),
        measure_runtime("saturation_transport", lambda: _saturation_stage(case)),
        measure_runtime("fusion", lambda: _fusion_stage(case)),
        measure_runtime("cross_scale", lambda: _cross_scale_stage(case)),
        measure_runtime("benchmark_registry", lambda: _registry_stage()),
    ]
    return {
        "case_id": case.case_id,
        "grid_shape": list(case.grid_shape),
        "total_cells": case.total_cells,
        "saturation_nx": case.saturation_nx,
        "stages": stages,
        "success": bool(all(stage["success"] for stage in stages)),
        "total_runtime_sec": float(sum(stage["runtime_sec"] for stage in stages)),
        "total_memory_peak_mb": float(max(stage["memory_peak_mb"] for stage in stages)),
        "has_nan": bool(any(stage.get("has_nan", False) for stage in stages)),
        "has_inf": bool(any(stage.get("has_inf", False) for stage in stages)),
    }


def run_stage_profiles(cases: list[SyntheticCase] | None = None) -> list[dict[str, Any]]:
    """Run all synthetic performance cases."""
    return [profile_synthetic_case(case) for case in (cases or SYNTHETIC_CASES)]


def check_numerical_equivalence() -> dict[str, Any]:
    """Check deterministic equivalence for repeated calls used by the baseline."""
    case = SYNTHETIC_CASES[0]
    pressure_a = _pressure_stage(case)["pressure_checksum"]
    pressure_b = _pressure_stage(case)["pressure_checksum"]
    sat_a = _saturation_stage(case)["saturation_checksum"]
    sat_b = _saturation_stage(case)["saturation_checksum"]
    fusion_a = _fusion_stage(case)["fused_checksum"]
    fusion_b = _fusion_stage(case)["fused_checksum"]
    pressure_error = abs(float(pressure_a) - float(pressure_b))
    saturation_error = abs(float(sat_a) - float(sat_b))
    fusion_error = abs(float(fusion_a) - float(fusion_b))
    max_abs_error = max(pressure_error, saturation_error, fusion_error)
    return {
        "success": bool(max_abs_error <= 1.0e-12),
        "pressure_checksum_error": float(pressure_error),
        "saturation_checksum_error": float(saturation_error),
        "fusion_checksum_error": float(fusion_error),
        "max_abs_error": float(max_abs_error),
    }


def summarize_profiles(case_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate runtime/memory summaries and migration recommendations."""
    all_stages = [stage for case in case_profiles for stage in case["stages"]]
    slowest = max(all_stages, key=lambda stage: stage["runtime_sec"])
    runtime_by_stage: dict[str, float] = {}
    memory_by_stage: dict[str, float] = {}
    for stage in all_stages:
        name = str(stage["stage_name"])
        runtime_by_stage[name] = runtime_by_stage.get(name, 0.0) + float(stage["runtime_sec"])
        memory_by_stage[name] = max(memory_by_stage.get(name, 0.0), float(stage["memory_peak_mb"]))
    max_runtime = float(slowest["runtime_sec"])
    max_memory = float(max(stage["memory_peak_mb"] for stage in all_stages))
    numba_recommended = bool(max_runtime > 2.0)
    cpp_recommended = bool(max_runtime > 5.0 or max_memory > 512.0)
    if not numba_recommended:
        numba_text = "not recommended for the current synthetic baseline; no clear Python kernel bottleneck was observed"
    else:
        numba_text = f"consider numba only for repeated production-scale {slowest['stage_name']} kernels"
    if not cpp_recommended:
        cpp_text = "not recommended for the current synthetic baseline; keep C++ deferred until larger profiling proves need"
    else:
        cpp_text = f"consider C++ only after isolating {slowest['stage_name']} as a production-scale bottleneck"
    return {
        "runtime_summary": {key: float(value) for key, value in sorted(runtime_by_stage.items())},
        "memory_summary": {key: float(value) for key, value in sorted(memory_by_stage.items())},
        "slowest_stage": {
            "case_id": _stage_case_id(case_profiles, slowest),
            "stage_name": slowest["stage_name"],
            "runtime_sec": float(slowest["runtime_sec"]),
            "memory_peak_mb": float(slowest["memory_peak_mb"]),
        },
        "numba_recommended": numba_recommended,
        "cpp_recommended": cpp_recommended,
        "numba_recommendation": numba_text,
        "cpp_recommendation": cpp_text,
    }


def _pressure_stage(case: SyntheticCase) -> dict[str, Any]:
    nz, ny, nx = case.grid_shape
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=10.0, dy=10.0, dz=5.0)
    result = solve_steady_state_pressure_3d(
        grid,
        100.0e-15,
        100.0e-15,
        80.0e-15,
        1.0e-3,
        {"left": 10.0e6, "right": 2.0e6},
    )
    pressure = result.pressure.values
    return {
        "success": True,
        "array_size_bytes": int(pressure.nbytes),
        "solver_iterations": result.report.get("num_iterations"),
        "solver_backend": result.report.get("solver"),
        "residual_norm": float(result.report.get("residual_norm", 0.0)),
        "mass_balance_error": float(result.report.get("mass_balance_error", 0.0)),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "pressure_checksum": float(np.sum(pressure)),
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
    }


def _saturation_stage(case: SyntheticCase) -> dict[str, Any]:
    grid = Grid3D(nx=case.saturation_nx, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    sw = np.full(grid.shape, 0.2, dtype=float)
    flux_x = np.full((1, 1, grid.nx + 1), 1.0e-5, dtype=float)
    result = advance_saturation_1d(
        grid=grid,
        sw=sw,
        phi=0.25,
        flux_x=flux_x,
        dt=100.0,
        relperm_params={**DEFAULT_RELPERM_PARAMS, "injected_sw": 0.8},
        max_cfl=1.0,
    )
    values = result.sw.values
    return {
        "success": bool(result.report["stable"]),
        "array_size_bytes": int(values.nbytes + flux_x.nbytes),
        "max_cfl": float(result.report["max_cfl"]),
        "material_balance_error": float(result.report["material_balance_error"]),
        "saturation_min": float(np.min(values)),
        "saturation_max": float(np.max(values)),
        "saturation_checksum": float(np.sum(values)),
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
    }


def _fusion_stage(case: SyntheticCase) -> dict[str, Any]:
    nz, ny, nx = case.grid_shape
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=10.0, dy=10.0, dz=5.0)
    base = np.linspace(0.2, 0.8, grid.total_cells, dtype=float).reshape(grid.shape)
    field_a = Field3D(grid, base, name="saturation_a", unit="fraction", confidence=np.full(grid.shape, 0.7))
    field_b = Field3D(grid, np.clip(base + 0.05, 0.0, 1.0), name="saturation_b", unit="fraction", confidence=np.full(grid.shape, 0.9))
    fused, report = weighted_average_fields([field_a, field_b], weights=[0.4, 0.6], clip_range=(0.0, 1.0))
    return {
        "success": not bool(report["has_nan"] or report["has_inf"]),
        "array_size_bytes": int(field_a.values.nbytes + field_b.values.nbytes + fused.values.nbytes),
        "fused_min": float(report["fused_min"]),
        "fused_max": float(report["fused_max"]),
        "fused_checksum": float(np.sum(fused.values)),
        "has_nan": bool(report["has_nan"]),
        "has_inf": bool(report["has_inf"]),
    }


def _cross_scale_stage(case: SyntheticCase) -> dict[str, Any]:
    config = _cross_scale_config(case.case_id)
    similarity = run_similarity_report(config)
    scale = run_scale_effect_report(config)
    validation = run_lab_field_validation_report(config)
    return {
        "success": bool(similarity["success"] and scale["success"] and validation["success"]),
        "array_size_bytes": int(0),
        "similarity_score": float(similarity["overall_similarity_score"]),
        "regime_shift_detected": bool(scale["regime_shift_detected"]),
        "rmse": float(validation["rmse"]),
        "mae": float(validation["mae"]),
        "has_nan": bool(similarity.get("has_nan") or scale.get("has_nan") or validation.get("has_nan")),
        "has_inf": bool(similarity.get("has_inf") or scale.get("has_inf") or validation.get("has_inf")),
    }


def _registry_stage() -> dict[str, Any]:
    registry = build_benchmark_registry("accuracy_reports")
    return {
        "success": bool(registry["success"]),
        "array_size_bytes": int(len(str(registry).encode("utf-8"))),
        "num_benchmark_summaries": int(registry["num_benchmark_summaries"]),
        "num_benchmark_cases": int(registry["num_benchmark_cases"]),
        "num_failed_cases": int(registry["num_failed_cases"]),
        "overclaim_warning_count": int(len(registry["overclaim_warnings"])),
        "has_nan": False,
        "has_inf": False,
    }


def _cross_scale_config(case_id: str) -> dict[str, Any]:
    lab = {
        "length_scale_m": 1.0,
        "time_scale_s": 10.0,
        "pressure_scale_pa": 1.0e5,
        "permeability_scale_m2": 1.0e-12,
        "porosity": 0.2,
        "viscosity_pa_s": 1.0e-3,
        "density_kg_m3": 1000.0,
        "velocity_scale_m_s": 1.0e-6,
        "flow_rate_m3_s": 1.0e-9,
        "temperature_scale_k": 300.0,
        "interfacial_tension_n_m": 0.03,
        "diffusivity_m2_s": 1.0e-9,
        "delta_density_kg_m3": 100.0,
        "pressure_drop_pa": 1.0e4,
        "elapsed_time_s": 100.0,
        "mobility_displacing": 2.0,
        "mobility_displaced": 1.0,
    }
    field = dict(lab)
    field.update(
        {
            "length_scale_m": 100.0,
            "time_scale_s": 1000.0,
            "pressure_scale_pa": 2.0e6,
            "permeability_scale_m2": 2.0e-13,
            "porosity": 0.25,
            "velocity_scale_m_s": 1.0e-3,
            "flow_rate_m3_s": 2.0e-5,
            "delta_density_kg_m3": 5000.0,
        }
    )
    return {
        "case_id": f"performance_{case_id}",
        "lab_case": {"descriptor": lab},
        "field_case": {"descriptor": field},
        "curves": [
            {
                "name": "water_cut",
                "lab": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.2, 0.5, 0.8]},
                "field": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.25, 0.45, 0.9]},
            }
        ],
    }


def _stage_case_id(case_profiles: list[dict[str, Any]], stage: dict[str, Any]) -> str:
    for case in case_profiles:
        if stage in case["stages"]:
            return str(case["case_id"])
    return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
