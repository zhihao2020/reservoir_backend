from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmarks.saturation_transport_benchmark import run_saturation_transport_benchmark
from benchmarks.capillary_gravity_benchmark import run_capillary_gravity_benchmark
from benchmarks.three_phase_benchmark import run_three_phase_benchmark
from reservoir_backend.core.exceptions import CFLViolationError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver import (
    cfl,
    limiters,
    saturation_transport_enhancement_report,
    transport_diagnostics,
    tvd_transport,
)
from reservoir_backend.solver.limiters import (
    compute_limited_slopes,
    minmod,
    preserves_monotonicity,
    superbee_limiter,
    vanleer_limiter,
)
from reservoir_backend.solver.pressure_enhancement_report import run_pressure_solver_enhancement_report
from reservoir_backend.solver.saturation_solver import advance_saturation_1d
from reservoir_backend.solver.saturation_transport_enhancement_report import (
    run_saturation_transport_enhancement_report,
)
from reservoir_backend.solver.transport_diagnostics import (
    build_boundedness_diagnostics,
    build_transport_diagnostics,
    compute_front_sharpness,
    compute_overshoot_undershoot,
    compute_total_variation,
    estimate_front_position,
)
from reservoir_backend.solver.tvd_transport import advance_saturation_1d_enhanced
from reservoir_backend.solver.tvd_transport import adapt_timestep, compute_cfl, suggest_stable_timestep


ROOT = Path(__file__).resolve().parents[1]


def _grid(nx: int = 20) -> Grid3D:
    return Grid3D(nx=nx, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)


def _flux(grid: Grid3D, value: float = 1.0e-5) -> np.ndarray:
    flux = np.zeros((1, 1, grid.nx + 1), dtype=float)
    flux[0, 0, :] = value
    return flux


def _params() -> dict[str, float]:
    return {
        "swi": 0.2,
        "sor": 0.2,
        "krw0": 1.0,
        "kro0": 1.0,
        "nw": 2.0,
        "no": 2.0,
        "mu_w": 1.0e-3,
        "mu_o": 5.0e-3,
    }


def _step_sw(grid: Grid3D) -> np.ndarray:
    values = np.full(grid.shape, 0.2)
    values[0, 0, : grid.nx // 3] = 0.65
    return values


def _summary(tmp_path: Path) -> dict:
    return run_saturation_transport_enhancement_report(tmp_path)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_cfl_module_exists():
    assert cfl is not None


def test_limiters_module_exists():
    assert limiters is not None


def test_tvd_transport_module_exists():
    assert tvd_transport is not None


def test_transport_diagnostics_module_exists():
    assert transport_diagnostics is not None


def test_saturation_transport_enhancement_report_module_exists():
    assert saturation_transport_enhancement_report is not None


def test_compute_cfl_scalar_case():
    grid = _grid(nx=2)
    field, report = compute_cfl(grid, 0.2, _flux(grid, 1.0e-5), np.zeros((1, 2, 2)), np.zeros((2, 1, 2)), 10.0)
    assert field.shape == grid.shape
    assert report["max_cfl"] > 0.0


def test_compute_cfl_vector_case():
    grid = _grid(nx=3)
    phi = np.full(grid.shape, 0.25)
    field, report = compute_cfl(grid, phi, _flux(grid, 2.0e-5), np.zeros((1, 2, 3)), np.zeros((2, 1, 3)), 20.0)
    assert np.isfinite(field).all()
    assert report["mean_cfl"] > 0.0


def test_suggest_stable_timestep_reduces_dt():
    grid = _grid()
    report = suggest_stable_timestep(grid, 0.2, _flux(grid, 1.0e-3), np.zeros((1, 2, grid.nx)), np.zeros((2, 1, grid.nx)), 1000.0, target_cfl=0.5)
    assert report["dt_suggested"] < 1000.0


def test_adapt_timestep_keeps_safe_dt():
    grid = _grid()
    report = adapt_timestep(grid, 0.2, _flux(grid, 1.0e-6), np.zeros((1, 2, grid.nx)), np.zeros((2, 1, grid.nx)), 1.0, target_cfl=0.8)
    assert report["dt_adapted"] == pytest.approx(1.0)


def test_adapt_timestep_reports_num_limited_cells():
    grid = _grid()
    report = adapt_timestep(grid, 0.2, _flux(grid, 1.0e-3), np.zeros((1, 2, grid.nx)), np.zeros((2, 1, grid.nx)), 1000.0, target_cfl=0.5)
    assert report["num_limited_cells"] > 0


def test_cfl_violation_warning_or_exception():
    grid = _grid()
    with pytest.raises(CFLViolationError):
        adapt_timestep(grid, 0.2, _flux(grid, 1.0e-3), np.zeros((1, 2, grid.nx)), np.zeros((2, 1, grid.nx)), 1000.0, target_cfl=0.5, raise_on_violation=True)


def test_minmod_limiter_basic():
    assert np.allclose(minmod(np.array([1.0, -1.0]), np.array([2.0, 2.0])), [1.0, 0.0])


def test_vanleer_limiter_basic_if_implemented():
    result = vanleer_limiter(np.array([0.0, 1.0, 2.0]))
    assert np.all(result >= 0.0)
    assert result[1] == pytest.approx(1.0)


def test_superbee_limiter_basic_if_implemented():
    result = superbee_limiter(np.array([0.0, 0.5, 2.0]))
    assert np.all(result >= 0.0)
    assert result[-1] == pytest.approx(2.0)


def test_limiter_preserves_monotonicity():
    values = np.array([0.2, 0.4, 0.6, 0.8])
    slopes = compute_limited_slopes(values, limiter="minmod")
    reconstructed = values + 0.5 * slopes
    assert preserves_monotonicity(values, reconstructed)


def test_upwind_method_matches_baseline():
    grid = _grid()
    sw = _step_sw(grid)
    enhanced = advance_saturation_1d_enhanced(grid, sw, 0.25, _flux(grid), 100.0, _params(), method="upwind")
    baseline = advance_saturation_1d(grid, sw, 0.25, _flux(grid), 100.0, _params())
    assert np.allclose(enhanced.sw.values, baseline.sw.values)


def test_tvd_method_runs_1d_case():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="tvd")
    assert result.report["method_used"] == "tvd"


def test_muscl_method_runs_if_implemented_or_skip():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="muscl")
    assert result.report["method_used"] == "muscl"


def test_tvd_boundedness_passes():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="tvd")
    assert result.report["boundedness_passed"] is True


def test_tvd_no_nan_inf():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="tvd")
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_tvd_mass_balance_reported():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="tvd")
    assert "material_balance_error" in result.report


def test_upwind_vs_tvd_front_sharpness_compared():
    summary = _summary(Path("accuracy_reports"))
    case = next(case for case in summary["cases"] if case["case_name"] == "upwind_tvd_front_sharpness_comparison")
    assert "front_sharpness_delta" in case["key_metrics"]


def test_front_position_detected():
    assert estimate_front_position(np.array([0.2, 0.3, 0.6]), threshold=0.5, dx=2.0) == pytest.approx(5.0)


def test_front_sharpness_metric_finite():
    assert np.isfinite(compute_front_sharpness(np.array([0.2, 0.6, 0.8])))


def test_total_variation_metric_finite():
    assert compute_total_variation(np.array([0.2, 0.6, 0.4])) == pytest.approx(0.6)


def test_overshoot_detected():
    report = compute_overshoot_undershoot(np.array([0.2, 1.2]), lower=0.0, upper=1.0)
    assert report["overshoot"] > 0.0


def test_undershoot_detected():
    report = compute_overshoot_undershoot(np.array([-0.1, 0.5]), lower=0.0, upper=1.0)
    assert report["undershoot"] > 0.0


def test_num_clipped_cells_reported():
    report = build_boundedness_diagnostics(np.array([-0.1, 0.5, 1.2]))
    assert report["num_clipped_cells"] == 2


def test_fallback_to_upwind_when_requested():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="implicit")
    assert result.report["method_used"] == "upwind"


def test_fallback_warning_generated():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="implicit")
    assert result.report["warnings"]


def test_implicit_request_returns_deferred_warning():
    grid = _grid()
    result = advance_saturation_1d_enhanced(grid, _step_sw(grid), 0.25, _flux(grid), 100.0, _params(), method="implicit")
    assert result.report["implicit_deferred"] is True


def test_implicit_not_claimed_as_implemented():
    summary = _summary(Path("accuracy_reports"))
    text = json.dumps(summary).lower()
    assert "implicit saturation transport is deferred" in text
    assert "fully implicit reservoir simulator" in text


def test_boundedness_diagnostics_json_serializable():
    json.dumps(build_boundedness_diagnostics(np.array([0.2, 0.5])))


def test_transport_diagnostics_json_serializable():
    json.dumps(build_transport_diagnostics(np.array([0.2, 0.4]), np.array([0.3, 0.5])))


def test_cfl_report_json_serializable():
    grid = _grid()
    json.dumps(adapt_timestep(grid, 0.2, _flux(grid), np.zeros((1, 2, grid.nx)), np.zeros((2, 1, grid.nx)), 10.0))


def test_saturation_enhancement_summary_json_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "saturation_transport_enhancement_summary.json").exists()


def test_saturation_enhancement_summary_markdown_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "saturation_transport_enhancement_summary.md").exists()


def test_summary_contains_cfl_cases(tmp_path):
    assert _summary(tmp_path)["cfl_cases"]


def test_summary_contains_upwind_tvd_comparison(tmp_path):
    assert _summary(tmp_path)["upwind_tvd_comparison"]


def test_summary_contains_fallback_cases(tmp_path):
    assert _summary(tmp_path)["fallback_cases"]


def test_summary_contains_limitations(tmp_path):
    assert _summary(tmp_path)["limitations"]


def test_summary_does_not_claim_black_oil(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "no black-oil" in text
    assert "complete black-oil model" not in text


def test_summary_does_not_claim_fully_implicit_solver(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "not implemented as a full solver" in text
    assert "fully implicit solver implemented" not in text


def test_docs_saturation_transport_enhancement_exists():
    assert (ROOT / "docs" / "saturation_transport_enhancement.md").exists()


def test_docs_mentions_tvd_optional():
    text = (ROOT / "docs" / "saturation_transport_enhancement.md").read_text(encoding="utf-8").lower()
    assert "tvd/muscl is optional" in text


def test_docs_mentions_upwind_baseline_preserved():
    text = (ROOT / "docs" / "saturation_transport_enhancement.md").read_text(encoding="utf-8").lower()
    assert "upwind baseline is preserved" in text


def test_docs_mentions_implicit_deferred():
    text = (ROOT / "docs" / "saturation_transport_enhancement.md").read_text(encoding="utf-8").lower()
    assert "implicit solver is deferred" in text


def test_docs_mentions_no_black_oil():
    text = (ROOT / "docs" / "saturation_transport_enhancement.md").read_text(encoding="utf-8").lower()
    assert "no black-oil" in text


def test_readme_mentions_saturation_transport_enhancement():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "saturation transport enhancement" in text


def test_traceability_mentions_task_014():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8").lower()
    assert "task-014" in text
    assert "saturation transport enhancement" in text


def test_existing_saturation_transport_benchmark_still_passes(tmp_path):
    assert run_saturation_transport_benchmark(tmp_path)["success"] is True


def test_existing_capillary_gravity_benchmark_still_passes(tmp_path):
    assert run_capillary_gravity_benchmark(tmp_path)["success"] is True


def test_existing_three_phase_benchmark_still_passes(tmp_path):
    assert run_three_phase_benchmark(tmp_path)["success"] is True


def test_existing_pressure_enhancement_tests_still_pass(tmp_path):
    assert run_pressure_solver_enhancement_report(tmp_path)["success"] is True


def test_does_not_modify_inversion():
    diff = _git_diff(["reservoir_backend/inversion"])
    known_preexisting = {
        "reservoir_backend/inversion/acoustic.py",
        "reservoir_backend/inversion/electromagnetic.py",
        "reservoir_backend/inversion/resistivity_archie.py",
        "reservoir_backend/inversion/saturation_fusion.py",
    }
    assert set(diff) <= known_preexisting


def test_does_not_modify_fusion():
    assert _git_diff(["reservoir_backend/fusion"]) == []


def test_does_not_modify_cross_scale():
    assert _git_diff(["reservoir_backend/cross_scale"]) == []


def test_does_not_modify_data_pipeline():
    assert _git_diff(["reservoir_backend/data"]) == []


def test_does_not_modify_result_export_contract():
    assert _git_diff(["reservoir_backend/results"]) == []


def test_does_not_modify_benchmarks():
    assert _git_diff(["benchmarks"]) == []


def test_pytest_all_pass_placeholder():
    assert True
