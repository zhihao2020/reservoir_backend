from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from reservoir_backend.data.qc import run_qc_pipeline
from reservoir_backend.data.reader import read_experimental_data
from reservoir_backend.data.resample import align_fields_to_grid_shape


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "experimental_data"


def _manifest() -> dict:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _fixture(fixture_id: str) -> dict:
    return next(item for item in _manifest()["fixtures"] if item["fixture_id"] == fixture_id)


def _metadata(item: dict) -> dict:
    return json.loads((FIXTURE_ROOT / item["metadata_path"]).read_text(encoding="utf-8"))


def _expected(item: dict) -> dict:
    return json.loads((FIXTURE_ROOT / item["expected_summary_path"]).read_text(encoding="utf-8"))


def _actual_summary(item: dict) -> dict:
    metadata = _metadata(item)
    dataset = read_experimental_data(FIXTURE_ROOT / item["input_path"], source_name=metadata["source_name"])
    report = run_qc_pipeline(dataset, required_fields=metadata.get("required_fields", [])).report
    resample_summary = {}
    if "target_shape" in metadata:
        aligned, resample_summary = align_fields_to_grid_shape(
            run_qc_pipeline(dataset, required_fields=metadata.get("required_fields", [])).dataset,
            tuple(metadata["target_shape"]),
        )
        report = run_qc_pipeline(aligned, required_fields=metadata.get("required_fields", [])).report
    return {
        "fixture_id": item["fixture_id"],
        "success": report["success"],
        "num_rows": report["num_rows"],
        "shape": report["shape"],
        "fields_detected": report["fields_detected"],
        "fields_missing": report["fields_missing"],
        "unit_warnings": report["unit_warnings"],
        "num_nan": report["num_nan"],
        "num_inf": report["num_inf"],
        "num_missing": report["num_missing"],
        "bounds_violations": report["bounds_violations"],
        "outlier_flags": report["num_outliers"],
        "duplicate_time_count": report["duplicate_time_count"],
        "duplicate_coordinate_count": report["duplicate_coordinate_count"],
        "resample_summary": resample_summary,
        "expected_recommendations": report["recommendations"],
        "warnings": report["warnings"],
        "source_name": report["source_name"],
    }


def test_fixture_directory_exists():
    assert FIXTURE_ROOT.exists()


def test_manifest_exists():
    assert (FIXTURE_ROOT / "manifest.json").exists()


def test_manifest_json_serializable():
    json.dumps(_manifest())


def test_manifest_has_at_least_five_fixtures():
    assert len(_manifest()["fixtures"]) >= 5


def test_each_fixture_has_input_metadata_expected_summary():
    for item in _manifest()["fixtures"]:
        assert (FIXTURE_ROOT / item["input_path"]).exists()
        assert (FIXTURE_ROOT / item["metadata_path"]).exists()
        assert (FIXTURE_ROOT / item["expected_summary_path"]).exists()


def test_valid_csv_core_fields_passes():
    item = _fixture("valid_csv_core_fields")
    assert _actual_summary(item)["success"] is True


def test_valid_json_multimodal_fields_passes():
    item = _fixture("valid_json_multimodal_fields")
    assert _actual_summary(item)["success"] is True


def test_valid_npz_grid_fields_passes():
    item = _fixture("valid_npz_grid_fields")
    actual = _actual_summary(item)
    assert actual["success"] is True
    assert actual["shape"] == [1, 2, 3]


def test_invalid_missing_required_fields_fails():
    item = _fixture("invalid_missing_required_fields")
    actual = _actual_summary(item)
    assert actual["success"] is False
    assert set(actual["fields_missing"]) == {"porosity", "permeability"}


def test_invalid_units_and_bounds_generates_expected_errors():
    item = _fixture("invalid_units_and_bounds")
    actual = _actual_summary(item)
    assert actual["success"] is False
    assert actual["unit_warnings"]
    assert {"porosity", "saturation", "permeability", "resistivity", "confidence", "variance"} <= set(actual["bounds_violations"])


def test_duplicate_time_or_coordinates_detected():
    item = _fixture("duplicate_time_or_coordinates")
    actual = _actual_summary(item)
    assert actual["duplicate_time_count"] == 1
    assert actual["duplicate_coordinate_count"] == 1


def test_nan_inf_missing_values_detected():
    item = _fixture("nan_inf_missing_values")
    actual = _actual_summary(item)
    assert actual["num_nan"] == 2
    assert actual["num_inf"] == 1
    assert actual["num_missing"] == 2


def test_metadata_preserved():
    item = _fixture("valid_json_multimodal_fields")
    dataset = read_experimental_data(FIXTURE_ROOT / item["input_path"])
    assert dataset.metadata["fixture_id"] == "valid_json_multimodal_fields"


def test_source_name_preserved():
    item = _fixture("valid_csv_core_fields")
    metadata = _metadata(item)
    dataset = read_experimental_data(FIXTURE_ROOT / item["input_path"], source_name=metadata["source_name"])
    assert dataset.source_name == metadata["source_name"]


def test_expected_summary_schema_valid():
    required = {
        "success",
        "num_rows",
        "shape",
        "fields_detected",
        "fields_missing",
        "unit_warnings",
        "num_nan",
        "num_inf",
        "num_missing",
        "bounds_violations",
        "outlier_flags",
        "expected_recommendations",
    }
    for item in _manifest()["fixtures"]:
        assert required <= set(_expected(item))


def test_expected_qc_behavior_matches_actual():
    for item in _manifest()["fixtures"]:
        actual = _actual_summary(item)
        expected = _expected(item)
        for key in (
            "success",
            "num_rows",
            "shape",
            "fields_detected",
            "fields_missing",
            "unit_warnings",
            "num_nan",
            "num_inf",
            "num_missing",
            "bounds_violations",
            "outlier_flags",
            "duplicate_time_count",
            "duplicate_coordinate_count",
        ):
            assert actual[key] == expected[key], (item["fixture_id"], key)


def test_all_supported_formats_covered():
    formats = {item["format"] for item in _manifest()["fixtures"]}
    assert {"csv", "json", "npz"} <= formats


def test_experimental_data_docs_exist():
    assert (ROOT / "docs" / "experimental_data_pipeline.md").exists()
    assert (ROOT / "docs" / "data_schema.md").exists()
    assert (ROOT / "docs" / "API_AND_DATA_CONTRACT.md").exists()


def test_data_schema_documents_core_fields():
    text = (ROOT / "docs" / "data_schema.md").read_text(encoding="utf-8")
    assert "porosity" in text
    assert "permeability" in text
    assert "resistivity" in text
    assert "variance" in text


def test_pipeline_documents_qc_and_units():
    text = (ROOT / "docs" / "experimental_data_pipeline.md").read_text(encoding="utf-8")
    assert "QC" in text
    assert "porosity" in text
    assert "pressure" in text


def test_docs_experimental_pipeline_still_mentions_qc():
    text = (ROOT / "docs" / "experimental_data_pipeline.md").read_text(encoding="utf-8")
    assert "QC Flow" in text
    assert "Fixture Catalog" in text


def test_readme_mentions_experimental_data_fixtures():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "tests/fixtures/experimental_data" in text


def test_manifest_should_pass_matches_expected():
    for item in _manifest()["fixtures"]:
        assert item["should_pass"] == _expected(item)["success"]


def test_manifest_expected_warnings_are_present():
    for item in _manifest()["fixtures"]:
        combined = " ".join(_expected(item)["warnings"] + _expected(item)["unit_warnings"])
        for warning in item["expected_warnings"]:
            assert warning in combined


def test_manifest_expected_errors_are_present():
    for item in _manifest()["fixtures"]:
        expected = _expected(item)
        for error in item["expected_errors"]:
            if error == "fields_missing":
                assert expected["fields_missing"]
            elif error == "bounds_violations":
                assert expected["bounds_violations"]
            elif error == "unit_warnings":
                assert expected["unit_warnings"]
            else:
                assert expected[error] > 0


def test_valid_fixtures_have_no_bounds_violations():
    for fixture_id in ["valid_csv_core_fields", "valid_json_multimodal_fields", "valid_npz_grid_fields"]:
        assert _expected(_fixture(fixture_id))["bounds_violations"] == {}


def test_invalid_fixtures_are_marked_invalid():
    for fixture_id in ["invalid_missing_required_fields", "invalid_units_and_bounds", "nan_inf_missing_values"]:
        assert _expected(_fixture(fixture_id))["success"] is False


def test_npz_fixture_is_binary_and_readable():
    item = _fixture("valid_npz_grid_fields")
    dataset = read_experimental_data(FIXTURE_ROOT / item["input_path"])
    assert dataset.input_format == "npz"
    assert np.asarray(dataset.fields["porosity"].values).size == 6


def test_pytest_all_pass_anchor():
    assert True
