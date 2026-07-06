from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from reservoir_backend.core.field import Field3D
from reservoir_backend.simulation import impes, production
from reservoir_backend.simulation.impes import (
    IMPESConfig,
    compute_mobility_fields,
    create_synthetic_waterflood_case,
    run_impes_simulation,
    run_impes_step,
)
from reservoir_backend.simulation.impes_report import run_impes_report


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> IMPESConfig:
    return create_synthetic_waterflood_case()


@pytest.fixture(scope="module")
def run_result(config: IMPESConfig):
    return run_impes_simulation(config)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_simulation_package_exists():
    import reservoir_backend.simulation as simulation

    assert hasattr(simulation, "run_impes_simulation")


def test_impes_module_exists():
    assert impes is not None


def test_production_module_exists():
    assert production is not None


def test_impes_report_module_exists():
    assert (ROOT / "reservoir_backend" / "simulation" / "impes_report.py").exists()


def test_synthetic_case_config_valid(config: IMPESConfig):
    assert config.grid.nx > 1
    assert config.grid.ny > 1
    assert config.grid.nz > 1
    assert config.num_steps > 1
    assert config.pressure_boundaries["left"] > config.pressure_boundaries["right"]


def test_compute_mobility_fields_shape(config: IMPESConfig):
    mobility = compute_mobility_fields(config.initial_sw, config.relperm_params)
    assert mobility["lambda_t"].shape == config.grid.shape
    assert mobility["fw"].shape == config.grid.shape


def test_compute_mobility_fields_finite(config: IMPESConfig):
    mobility = compute_mobility_fields(config.initial_sw, config.relperm_params)
    assert np.isfinite(mobility["lambda_t"]).all()
    assert np.isfinite(mobility["fw"]).all()


def test_single_step_coupling_runs(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert step.pressure.values.shape == config.grid.shape
    assert step.sw.values.shape == config.grid.shape
    assert step.face_fluxes.flux_x.shape == (config.grid.nz, config.grid.ny, config.grid.nx + 1)


def test_single_step_pressure_finite(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert np.isfinite(step.pressure.values).all()
    assert float(np.max(step.pressure.values)) > float(np.min(step.pressure.values))


def test_single_step_flux_finite(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert np.isfinite(step.face_fluxes.flux_x).all()
    assert np.max(np.abs(step.face_fluxes.flux_x)) > 0.0


def test_single_step_saturation_bounded(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert float(np.min(step.sw.values)) >= config.relperm_params["swi"]
    assert float(np.max(step.sw.values)) <= 1.0 - config.relperm_params["sor"]


def test_single_step_cfl_report(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert step.cfl_report["stable"] is True
    assert step.cfl_report["max_cfl"] <= config.max_cfl
    assert "suggested_stable_dt" in step.cfl_report


def test_single_step_material_balance_reported(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    assert "material_balance_error" in step.saturation_report
    assert step.mass_balance_error >= 0.0


def test_single_step_production_summary(config: IMPESConfig):
    step = run_impes_step(config=config, sw=config.initial_sw)
    summary = step.production_summary
    assert summary["total_liquid_rate"] > 0.0
    assert summary["oil_rate"] >= 0.0
    assert 0.0 <= summary["water_cut"] <= 1.0


def test_multi_step_waterflood_runs(run_result):
    assert run_result.summary["success"] is True
    assert len(run_result.steps) == run_result.config.num_steps


def test_multi_step_pressure_flux_saturation_records(run_result):
    first = run_result.steps[0]
    assert isinstance(first.pressure, Field3D)
    assert first.face_fluxes.flux_y.shape[1] == run_result.config.grid.ny + 1
    assert isinstance(first.sw, Field3D)


def test_mobility_updates_across_steps(run_result):
    initial = compute_mobility_fields(run_result.config.initial_sw, run_result.config.relperm_params)
    final = compute_mobility_fields(run_result.steps[-1].sw.values, run_result.config.relperm_params)
    assert not np.allclose(initial["lambda_t"], final["lambda_t"])


def test_production_curve_generated(run_result):
    curve = run_result.production_curve
    assert len(curve) == run_result.config.num_steps
    assert {"time", "water_cut", "water_rate", "oil_rate"} <= set(curve[0])


def test_water_cut_curve_nonnegative(run_result):
    water_cuts = [entry["water_cut"] for entry in run_result.production_curve]
    assert all(0.0 <= value <= 1.0 for value in water_cuts)
    assert water_cuts[-1] > 0.0


def test_breakthrough_time_detected(run_result):
    assert run_result.breakthrough_time is not None
    assert run_result.breakthrough_time > 0.0


def test_detect_breakthrough_time_none_before_threshold():
    assert production.detect_breakthrough_time([1.0, 2.0], [0.0, 0.001], threshold=0.1) is None


def test_detect_breakthrough_time_first_crossing():
    assert production.detect_breakthrough_time([1.0, 2.0, 3.0], [0.0, 0.2, 0.3], threshold=0.1) == 2.0


def test_summary_keys(run_result):
    required = {
        "success",
        "num_steps",
        "max_cfl",
        "max_mass_balance_error",
        "production_curve",
        "breakthrough_time",
        "limitations",
    }
    assert required <= set(run_result.summary)


def test_summary_json_serializable(run_result):
    json.dumps(run_result.summary)


def test_impes_report_generates_json(tmp_path: Path):
    summary = run_impes_report(tmp_path)
    assert (tmp_path / "impes_loop_summary.json").exists()
    assert summary["success"] is True


def test_impes_report_generates_markdown(tmp_path: Path):
    run_impes_report(tmp_path)
    text = (tmp_path / "impes_loop_summary.md").read_text(encoding="utf-8")
    assert "IMPES Sequential Loop Summary" in text
    assert "Production Curve" in text


def test_report_mentions_non_claims(tmp_path: Path):
    run_impes_report(tmp_path)
    text = (tmp_path / "impes_loop_summary.md").read_text(encoding="utf-8")
    assert "No fully implicit simulator" in text
    assert "No black-oil" in text
    assert "No complex well-control" in text


def test_docs_impes_sequential_loop_exists():
    assert (ROOT / "docs" / "impes_sequential_loop.md").exists()


def test_docs_mentions_pressure_saturation_loop():
    text = (ROOT / "docs" / "impes_sequential_loop.md").read_text(encoding="utf-8")
    assert "pressure -> flux -> saturation -> mobility -> pressure" in text


def test_docs_mentions_no_fully_implicit():
    text = (ROOT / "docs" / "impes_sequential_loop.md").read_text(encoding="utf-8")
    assert "No fully implicit simulator" in text


def test_docs_mentions_no_black_oil():
    text = (ROOT / "docs" / "impes_sequential_loop.md").read_text(encoding="utf-8")
    assert "No black-oil" in text


def test_readme_mentions_impes_loop():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "IMPES" in text


def test_module_matrix_mentions_impes():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "IMPES" in text


def test_traceability_mentions_f3_04():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "F3-04" in text


def test_no_pressure_solver_core_modification():
    assert _git_diff(["reservoir_backend/solver/pressure_solver.py"]) == []


def test_no_saturation_solver_core_modification():
    assert _git_diff(["reservoir_backend/solver/saturation_solver.py"]) == []


def test_does_not_modify_forbidden_modules():
    assert _git_diff(
        [
            "reservoir_backend/inversion",
            "reservoir_backend/fusion",
            "reservoir_backend/cross_scale",
            "reservoir_backend/data",
            "reservoir_backend/results",
            "benchmarks",
            "references",
            "config",
        ]
    ) == []


def test_existing_project_case_report_still_runs(tmp_path: Path):
    from reservoir_backend.project.case_report import run_project_case_management_report

    summary = run_project_case_management_report(tmp_path, root=ROOT)
    assert summary["success"] is True


def test_existing_pressure_enhancement_report_still_runs(tmp_path: Path):
    from reservoir_backend.solver.pressure_enhancement_report import run_pressure_solver_enhancement_report

    summary = run_pressure_solver_enhancement_report(tmp_path)
    assert summary["success"] is True


def test_existing_saturation_transport_enhancement_report_still_runs(tmp_path: Path):
    from reservoir_backend.solver.saturation_transport_enhancement_report import (
        run_saturation_transport_enhancement_report,
    )

    summary = run_saturation_transport_enhancement_report(tmp_path)
    assert summary["success"] is True
