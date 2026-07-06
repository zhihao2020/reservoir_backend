from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from reservoir_backend.fusion import dynamic_field_fusion, synthetic_twin, synthetic_twin_report
from reservoir_backend.fusion.dynamic_field_fusion import (
    build_synthetic_twin_fusion_summary,
    check_shape_time_consistency,
    fuse_dynamic_field_records,
    fuse_production_series_records,
    fuse_static_field_records,
)
from reservoir_backend.fusion.synthetic_twin import (
    DynamicFieldRecord,
    ProductionSeriesRecord,
    StaticFieldRecord,
    SyntheticTwinMetadata,
)
from reservoir_backend.fusion.synthetic_twin_report import (
    build_synthetic_twin_fixture,
    run_synthetic_twin_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> SyntheticTwinMetadata:
    return SyntheticTwinMetadata(
        twin_id="twin",
        case_id="case",
        run_id="run",
        created_at="2026-07-06T00:00:00+00:00",
        grid_shape=(1, 2, 3),
        time_steps=(0.0, 1.0),
        source_name="tests",
        metadata={"purpose": "unit"},
    )


def _truth_static() -> np.ndarray:
    return np.full((1, 2, 3), 0.25)


def _truth_dynamic() -> np.ndarray:
    return np.stack([np.full((1, 2, 3), 0.2), np.full((1, 2, 3), 0.4)])


def _static_records() -> list[StaticFieldRecord]:
    truth = _truth_static()
    return [
        StaticFieldRecord("porosity", truth + 0.02, "fraction", "source_low", confidence=0.2, truth=truth, provenance={"rank": 1}),
        StaticFieldRecord("porosity", truth + 0.005, "fraction", "source_high", confidence=0.8, truth=truth, provenance={"rank": 2}),
    ]


def _dynamic_records() -> list[DynamicFieldRecord]:
    truth = _truth_dynamic()
    return [
        DynamicFieldRecord("saturation", truth - 0.03, (0.0, 1.0), "fraction", "sim", confidence=0.3, truth=truth),
        DynamicFieldRecord("saturation", truth + 0.01, (0.0, 1.0), "fraction", "obs", confidence=0.9, truth=truth),
    ]


def _production_records() -> list[ProductionSeriesRecord]:
    truth = np.array([0.0, 0.2])
    return [
        ProductionSeriesRecord("water_cut", [0.0, 1.0], truth - 0.02, "fraction", "sim", confidence=0.2, truth=truth),
        ProductionSeriesRecord("water_cut", [0.0, 1.0], truth + 0.01, "fraction", "obs", confidence=0.8, truth=truth),
    ]


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_synthetic_twin_module_exists():
    assert synthetic_twin is not None


def test_dynamic_field_fusion_module_exists():
    assert dynamic_field_fusion is not None


def test_synthetic_twin_report_module_exists():
    assert synthetic_twin_report is not None


def test_metadata_to_dict_json_serializable():
    json.dumps(_metadata().to_dict())


def test_metadata_rejects_empty_id():
    with pytest.raises(ValueError):
        SyntheticTwinMetadata("", "case", "run", "now", (1, 1, 1), (0.0,), "source")


def test_static_field_record_preserves_source_confidence_mask_truth():
    mask = np.ones((1, 2, 3), dtype=bool)
    record = StaticFieldRecord("porosity", _truth_static(), "fraction", "source", confidence=0.5, mask=mask, truth=_truth_static())
    assert record.source == "source"
    assert record.confidence.shape == record.shape
    assert record.mask.shape == record.shape
    assert record.truth.shape == record.shape


def test_dynamic_field_record_preserves_time_steps():
    record = _dynamic_records()[0]
    assert record.time_steps == (0.0, 1.0)
    assert record.shape == (2, 1, 2, 3)


def test_dynamic_field_record_rejects_time_mismatch():
    with pytest.raises(ValueError):
        DynamicFieldRecord("pressure", np.zeros((2, 1, 2, 3)), (0.0,), "Pa", "source")


def test_production_series_record_preserves_metadata():
    record = ProductionSeriesRecord("water_cut", [0.0, 1.0], [0.0, 0.2], "fraction", "prod", provenance={"kind": "observed"})
    assert record.to_dict()["provenance"]["kind"] == "observed"


def test_production_series_record_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ProductionSeriesRecord("water_cut", [0.0, 1.0], [0.0], "fraction", "prod")


def test_shape_time_consistency_passes():
    report = check_shape_time_consistency(_metadata(), _static_records(), _dynamic_records(), _production_records())
    assert report["success"] is True


def test_shape_time_consistency_detects_static_shape_mismatch():
    bad = [StaticFieldRecord("porosity", np.ones((2, 2)), "fraction", "bad")]
    report = check_shape_time_consistency(_metadata(), bad, [], [])
    assert report["success"] is False


def test_shape_time_consistency_detects_dynamic_time_mismatch():
    metadata = SyntheticTwinMetadata("t", "c", "r", "now", (1, 2, 3), (0.0, 2.0), "source")
    report = check_shape_time_consistency(metadata, [], _dynamic_records(), [])
    assert report["success"] is False


def test_static_field_fusion_runs():
    result = fuse_static_field_records(_static_records())
    assert "porosity" in result
    assert result["porosity"]["source_count"] == 2


def test_static_field_fusion_confidence_dominates():
    result = fuse_static_field_records(_static_records())["porosity"]
    assert result["fused_max"] < 0.27
    assert result["diagnostics"]["weighting_policy"] == "confidence"


def test_static_field_truth_error_reported():
    result = fuse_static_field_records(_static_records())["porosity"]
    assert result["truth_error"]["rmse"] is not None
    assert result["truth_error"]["mae"] is not None


def test_dynamic_field_fusion_runs():
    result = fuse_dynamic_field_records(_dynamic_records())
    assert "saturation" in result
    assert result["saturation"]["shape"] == [2, 1, 2, 3]


def test_dynamic_field_bounds_checked():
    result = fuse_dynamic_field_records(_dynamic_records())["saturation"]
    assert result["diagnostics"]["bounds_violations"] == 0


def test_dynamic_field_truth_error_reported():
    result = fuse_dynamic_field_records(_dynamic_records())["saturation"]
    assert result["truth_error"]["max_abs_error"] is not None


def test_dynamic_field_mask_propagates_to_warning():
    records = _dynamic_records()
    mask = np.ones(records[0].shape, dtype=bool)
    mask[:, :, :, 0] = False
    masked = DynamicFieldRecord("saturation", records[0].values, records[0].time_steps, "fraction", "masked", confidence=0.5, mask=mask)
    result = fuse_dynamic_field_records([masked])["saturation"]
    assert result["diagnostics"]["has_nan"] is True
    assert result["diagnostics"]["warnings"]


def test_production_series_fusion_runs():
    result = fuse_production_series_records(_production_records())
    assert "water_cut" in result
    assert result["water_cut"]["shape"] == [2]


def test_production_series_bounds_checked():
    result = fuse_production_series_records(_production_records())["water_cut"]
    assert result["diagnostics"]["bounds_violations"] == 0


def test_production_series_truth_error_reported():
    result = fuse_production_series_records(_production_records())["water_cut"]
    assert result["truth_error"]["rmse"] is not None


def test_build_synthetic_twin_summary_success():
    summary = build_synthetic_twin_fusion_summary(
        metadata=_metadata(),
        static_records=_static_records(),
        dynamic_records=_dynamic_records(),
        production_records=_production_records(),
    )
    assert summary.to_dict()["success"] is True


def test_summary_contains_static_dynamic_production():
    data = build_synthetic_twin_fusion_summary(
        metadata=_metadata(),
        static_records=_static_records(),
        dynamic_records=_dynamic_records(),
        production_records=_production_records(),
    ).to_dict()
    assert data["diagnostics"]["num_static_fields"] == 1
    assert data["diagnostics"]["num_dynamic_fields"] == 1
    assert data["diagnostics"]["num_production_series"] == 1


def test_summary_preserves_provenance_sources():
    data = build_synthetic_twin_fusion_summary(
        metadata=_metadata(),
        static_records=_static_records(),
        dynamic_records=_dynamic_records(),
        production_records=_production_records(),
    ).to_dict()
    assert "source_high" in data["provenance"]["sources"]
    assert "obs" in data["provenance"]["sources"]


def test_summary_json_serializable():
    data = build_synthetic_twin_fusion_summary(
        metadata=_metadata(),
        static_records=_static_records(),
        dynamic_records=_dynamic_records(),
        production_records=_production_records(),
    ).to_dict()
    json.dumps(data)


def test_summary_rejects_inconsistent_shape():
    bad = [StaticFieldRecord("porosity", np.ones((2, 2)), "fraction", "bad")]
    with pytest.raises(ValueError):
        build_synthetic_twin_fusion_summary(metadata=_metadata(), static_records=bad, dynamic_records=[], production_records=[])


def test_fixture_builds():
    metadata, static_records, dynamic_records, production_records = build_synthetic_twin_fixture()
    assert metadata.twin_id
    assert static_records
    assert dynamic_records
    assert production_records


def test_report_runner_generates_json(tmp_path: Path):
    summary = run_synthetic_twin_report(tmp_path)
    assert summary["success"] is True
    assert (tmp_path / "fusion_synthetic_twin_summary.json").exists()


def test_report_runner_generates_markdown(tmp_path: Path):
    run_synthetic_twin_report(tmp_path)
    text = (tmp_path / "fusion_synthetic_twin_summary.md").read_text(encoding="utf-8")
    assert "Synthetic Twin Dynamic Field Fusion Summary" in text


def test_report_contains_limitations(tmp_path: Path):
    summary = run_synthetic_twin_report(tmp_path)
    assert any("No history matching" in item for item in summary["limitations"])
    assert any("No EnKF" in item for item in summary["limitations"])


def test_report_does_not_claim_closed_loop(tmp_path: Path):
    run_synthetic_twin_report(tmp_path)
    text = (tmp_path / "fusion_synthetic_twin_summary.md").read_text(encoding="utf-8")
    assert "No closed-loop digital twin control" in text


def test_static_permeability_positive_fixture():
    _, static_records, _, _ = build_synthetic_twin_fixture()
    permeability = [record for record in static_records if record.field_name == "permeability"]
    result = fuse_static_field_records(permeability)["permeability"]
    assert result["fused_min"] > 0.0


def test_dynamic_pressure_fixture_finite():
    _, _, dynamic_records, _ = build_synthetic_twin_fixture()
    pressure = [record for record in dynamic_records if record.field_name == "pressure"]
    result = fuse_dynamic_field_records(pressure)["pressure"]
    assert result["diagnostics"]["has_inf"] is False


def test_dynamic_saturation_fixture_bounded():
    _, _, dynamic_records, _ = build_synthetic_twin_fixture()
    saturation = [record for record in dynamic_records if record.field_name == "saturation"]
    result = fuse_dynamic_field_records(saturation)["saturation"]
    assert result["diagnostics"]["bounds_violations"] == 0


def test_docs_fusion_synthetic_twin_exists():
    assert (ROOT / "docs" / "fusion_synthetic_twin.md").exists()


def test_docs_mentions_no_history_matching():
    text = (ROOT / "docs" / "fusion_synthetic_twin.md").read_text(encoding="utf-8")
    assert "No history matching" in text


def test_docs_mentions_no_enkf_esmda():
    text = (ROOT / "docs" / "fusion_synthetic_twin.md").read_text(encoding="utf-8")
    assert "No EnKF / ES-MDA" in text


def test_readme_mentions_synthetic_twin():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "synthetic twin" in text


def test_module_matrix_mentions_synthetic_twin():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "synthetic twin" in text


def test_traceability_mentions_f4_04():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "F4-04" in text


def test_no_solver_modification():
    assert _git_diff(["reservoir_backend/solver"]) == []


def test_no_inversion_modification():
    assert _git_diff(["reservoir_backend/inversion"]) == []


def test_no_cross_scale_modification():
    assert _git_diff(["reservoir_backend/cross_scale"]) == []


def test_no_data_modification():
    assert _git_diff(["reservoir_backend/data"]) == []


def test_no_benchmark_modification():
    assert _git_diff(["benchmarks"]) == []


def test_no_reference_modification():
    assert _git_diff(["references"]) == []


def test_existing_uncertainty_report_still_runs(tmp_path: Path):
    from reservoir_backend.fusion.uncertainty_report import run_parameter_fusion_uncertainty_report

    assert run_parameter_fusion_uncertainty_report(tmp_path)["success"] is True


def test_existing_impes_report_still_runs(tmp_path: Path):
    from reservoir_backend.simulation.impes_report import run_impes_report

    assert run_impes_report(tmp_path)["success"] is True


def test_existing_project_case_report_still_runs(tmp_path: Path):
    from reservoir_backend.project.case_report import run_project_case_management_report

    assert run_project_case_management_report(tmp_path, root=ROOT)["success"] is True
