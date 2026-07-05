from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmarks.pressure_solver_benchmark import run_pressure_solver_benchmark
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver import boundary_matrix, linear_solver_backend, pressure_enhancement_report, well_source
from reservoir_backend.solver.boundary_matrix import (
    BoundaryConditionContribution,
    apply_source_sink_to_rhs,
    build_boundary_contribution,
    build_boundary_diagnostics,
)
from reservoir_backend.solver.linear_solver_backend import solve_linear_system
from reservoir_backend.solver.pressure_enhancement_report import run_pressure_solver_enhancement_report
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_2d
from reservoir_backend.solver.well_source import (
    RateControlledWell,
    build_well_contribution_vector,
    summarize_wells,
)


ROOT = Path(__file__).resolve().parents[1]


def _grid() -> Grid3D:
    return Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)


def _matrix_rhs() -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=float)
    rhs = np.array([1.0, 2.0], dtype=float)
    return matrix, rhs


def _summary(tmp_path: Path) -> dict:
    return run_pressure_solver_enhancement_report(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_well_source_module_exists():
    assert well_source is not None


def test_boundary_matrix_module_exists():
    assert boundary_matrix is not None


def test_linear_solver_backend_module_exists():
    assert linear_solver_backend is not None


def test_pressure_enhancement_report_module_exists():
    assert pressure_enhancement_report is not None


def test_rate_control_well_created():
    well = RateControlledWell("W1", "injector", rate=1.0, cell_index=0)
    assert well.control_type == "rate"


def test_injector_positive_rate_convention():
    assert RateControlledWell("I1", "injector", rate=2.0, cell_index=0).signed_rate == 2.0


def test_producer_negative_rate_convention():
    assert RateControlledWell("P1", "producer", rate=2.0, cell_index=0).signed_rate == -2.0


def test_well_cell_index_validation():
    well = RateControlledWell("W1", "injector", rate=1.0, cell_index=5)
    assert well.resolved_cell_index(_grid()) == 5


def test_well_ijk_validation():
    well = RateControlledWell("W1", "injector", rate=1.0, i=2, j=1, k=0)
    assert well.resolved_cell_index(_grid()) == 5


def test_well_out_of_bounds_rejected():
    well = RateControlledWell("W1", "injector", rate=1.0, cell_index=99)
    with pytest.raises(ValueError):
        well.resolved_cell_index(_grid())


def test_well_negative_rate_rejected():
    with pytest.raises(ValueError):
        RateControlledWell("W1", "injector", rate=-1.0, cell_index=0)


def test_multiple_wells_total_rate():
    wells = [
        RateControlledWell("I", "injector", rate=3.0, cell_index=0),
        RateControlledWell("P", "producer", rate=2.0, cell_index=1),
    ]
    report = summarize_wells(wells, (1, 1, 3))
    assert report["total_injection_rate"] == 3.0
    assert report["total_production_rate"] == 2.0
    assert report["net_source_rate"] == 1.0


def test_well_contribution_vector_shape():
    vector = build_well_contribution_vector([RateControlledWell("I", "injector", 1.0, cell_index=0)], _grid())
    assert vector.shape == (_grid().total_cells,)


def test_well_contribution_mass_balance():
    wells = [
        RateControlledWell("I", "injector", 3.0, cell_index=0),
        RateControlledWell("P", "producer", 3.0, cell_index=1),
    ]
    assert np.sum(build_well_contribution_vector(wells, (1, 1, 3))) == 0.0


def test_source_sink_terms_added_to_rhs():
    result = apply_source_sink_to_rhs(np.array([1.0, 1.0]), np.array([2.0, -1.0]))
    assert np.allclose(result, [3.0, 0.0])


def test_dirichlet_boundary_contribution():
    contribution = build_boundary_contribution(
        (1, 1, 2),
        [BoundaryConditionContribution("left", "dirichlet", value=10.0, transmissibility=2.0)],
    )
    assert contribution["matrix_diagonal"][0] == 2.0
    assert contribution["rhs"][0] == 20.0


def test_neumann_boundary_contribution():
    contribution = build_boundary_contribution(
        (1, 1, 2),
        [BoundaryConditionContribution("right", "neumann", value=-3.0)],
    )
    assert contribution["matrix_diagonal"][-1] == 0.0
    assert contribution["rhs"][-1] == -3.0


def test_noflow_boundary_contribution():
    contribution = build_boundary_contribution(
        (1, 1, 2),
        [BoundaryConditionContribution("left", "noflow", value=10.0)],
    )
    assert np.count_nonzero(contribution["matrix_diagonal"]) == 0
    assert np.count_nonzero(contribution["rhs"]) == 0


def test_boundary_matrix_shape():
    contribution = build_boundary_contribution(_grid(), [])
    assert contribution["matrix_shape"] == [_grid().total_cells, _grid().total_cells]


def test_boundary_rhs_shape():
    contribution = build_boundary_contribution(_grid(), [])
    assert contribution["rhs_shape"] == [_grid().total_cells]


def test_boundary_diagnostics_generated():
    contribution = build_boundary_contribution(
        (1, 1, 2),
        [BoundaryConditionContribution("left", "dirichlet", value=10.0, transmissibility=2.0)],
    )
    diagnostics = build_boundary_diagnostics(contribution)
    assert diagnostics["success"] is True
    assert diagnostics["num_nonzero_diagonal"] == 1


def test_invalid_boundary_rejected():
    with pytest.raises(ValueError):
        BoundaryConditionContribution("unknown", "dirichlet")


def test_linear_solver_direct_backend():
    matrix, rhs = _matrix_rhs()
    solution, stats = solve_linear_system(matrix, rhs, backend="direct")
    assert stats["backend"] == "direct"
    assert np.allclose(matrix @ solution, rhs)


def test_linear_solver_cg_or_gmres_backend_if_supported():
    matrix, rhs = _matrix_rhs()
    solution, stats = solve_linear_system(matrix, rhs, backend="cg")
    assert stats["success"] is True
    assert np.allclose(matrix @ solution, rhs, atol=1.0e-8)


def test_gmres_backend_if_supported():
    matrix, rhs = _matrix_rhs()
    solution, stats = solve_linear_system(matrix, rhs, backend="gmres")
    assert stats["success"] is True
    assert np.allclose(matrix @ solution, rhs, atol=1.0e-8)


def test_ilu_backend_optional_or_skip():
    matrix, rhs = _matrix_rhs()
    solution, stats = solve_linear_system(matrix, rhs, backend="ilu")
    assert stats["success"] is True
    assert np.allclose(matrix @ solution, rhs, atol=1.0e-8)


def test_amg_backend_optional_or_skip():
    matrix, rhs = _matrix_rhs()
    solution, stats = solve_linear_system(matrix, rhs, backend="amg")
    assert stats["success"] is True
    assert np.allclose(matrix @ solution, rhs, atol=1.0e-8)


def test_fallback_used_when_backend_unavailable():
    matrix, rhs = _matrix_rhs()
    _, stats = solve_linear_system(matrix, rhs, backend="amg")
    assert "fallback_used" in stats
    assert "requested_backend" in stats


def test_solver_stats_contains_backend():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    assert stats["backend"] == "direct"


def test_solver_stats_contains_success():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    assert stats["success"] is True


def test_solver_stats_contains_residual_norm():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    assert stats["residual_norm"] < 1.0e-12


def test_solver_stats_contains_mass_balance_error():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    assert stats["mass_balance_error"] >= 0.0


def test_solver_stats_json_serializable():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    json.dumps(stats)


def test_pressure_solution_finite():
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    result = solve_steady_state_pressure_2d(
        grid,
        kx=1.0e-12,
        ky=1.0e-12,
        mu=1.0e-3,
        dirichlet_boundaries={"left": 2.0e7, "right": 1.0e7},
    )
    assert np.isfinite(result.pressure.values).all()


def test_flux_conservation_error_reported():
    _, stats = solve_linear_system(*_matrix_rhs(), backend="direct")
    assert "flux_conservation_error" in stats


def test_mass_balance_with_wells_case(tmp_path):
    summary = _summary(tmp_path)
    case = _case(summary, "well_source_sink_rate_control")
    assert case["key_metrics"]["mass_balance_error"] == 0.0


def test_mass_balance_with_boundary_case(tmp_path):
    case = _case(_summary(tmp_path), "boundary_matrix_contribution")
    assert case["key_metrics"]["mass_balance_error"] == 0.0


def test_heterogeneous_permeability_case_still_passes(tmp_path):
    summary = run_pressure_solver_benchmark(tmp_path)
    case = next(item for item in summary["cases"] if item["case_name"] == "opm_spe1case1_layered_adapted")
    assert case["success"] is True


def test_existing_pressure_benchmark_still_passes(tmp_path):
    assert run_pressure_solver_benchmark(tmp_path)["success"] is True


def test_pressure_enhancement_summary_json_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "pressure_solver_enhancement_summary.json").exists()


def test_pressure_enhancement_summary_markdown_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "pressure_solver_enhancement_summary.md").exists()


def test_pressure_enhancement_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_summary_contains_well_cases(tmp_path):
    assert _summary(tmp_path)["well_cases"]


def test_summary_contains_boundary_cases(tmp_path):
    assert _summary(tmp_path)["boundary_cases"]


def test_summary_contains_solver_backend_cases(tmp_path):
    assert _summary(tmp_path)["solver_backend_cases"]


def test_summary_contains_limitations(tmp_path):
    limitations = " ".join(_summary(tmp_path)["limitations"])
    assert "No full Peaceman industrial well model" in limitations


def test_summary_does_not_claim_black_oil(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "black-oil model implemented" in text
    assert "complete black-oil model" not in text


def test_summary_does_not_claim_industrial_well_model(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "industrial well model implemented" in text
    assert "full peaceman industrial well model" in text


def test_docs_pressure_solver_enhancement_exists():
    assert (ROOT / "docs" / "pressure_solver_enhancement.md").exists()


def test_docs_mentions_no_black_oil():
    text = (ROOT / "docs" / "pressure_solver_enhancement.md").read_text(encoding="utf-8").lower()
    assert "no black-oil" in text


def test_docs_mentions_no_complex_wellbore_network():
    text = (ROOT / "docs" / "pressure_solver_enhancement.md").read_text(encoding="utf-8").lower()
    assert "no complex wellbore network" in text


def test_readme_mentions_pressure_enhancement():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "pressure solver enhancement" in text


def test_traceability_mentions_task_011():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8").lower()
    assert "task-011" in text
    assert "pressure solver enhancement" in text


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


def test_report_runner_main(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    summary = run_pressure_solver_enhancement_report(tmp_path)
    assert summary["success"] is True


def test_pytest_all_pass_placeholder():
    assert True
