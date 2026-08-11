from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from reservoir_backend.results import ResultCatalog, ResultManifest, build_report_path_index
from reservoir_backend.results.export import (
    build_example_result_manifests,
    export_field_npz,
    export_manifest_json,
    export_markdown_report_index,
    export_summary_csv,
    generate_result_manifest_summary,
)
from reservoir_backend.results.manifest import REQUIRED_MANIFEST_KEYS, validate_result_manifest

ROOT = Path(__file__).resolve().parents[1]

def _manifest(**overrides) -> ResultManifest:
    data = {
        "result_id": "test_pressure",
        "case_id": "demo_case",
        "run_id": "run_001",
        "module": "M3",
        "result_type": "pressure_field",
        "field_name": "pressure",
        "shape": [2, 3, 4],
        "dtype": "float64",
        "unit": "Pa",
        "path": "accuracy_reports/pressure_solver_benchmark_summary.json",
        "format": "json",
        "created_at": "2026-07-05T00:00:00+00:00",
        "source_task": "TASK-020",
        "source_report": "accuracy_reports/pressure_solver_benchmark_summary.json",
        "metadata": {"shape_convention": "(nz, ny, nx)"},
        "warnings": ["example warning"],
        "limitations": ["example limitation"],
    }
    data.update(overrides)
    return ResultManifest.from_dict(data)

def _catalog() -> ResultCatalog:
    return ResultCatalog([_manifest(), _manifest(result_id="test_sw", result_type="saturation_field", field_name="sw", unit="fraction")])

def _read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]

def test_results_package_exists():
    assert (ROOT / "reservoir_backend" / "results" / "__init__.py").exists()

def test_manifest_module_exists():
    assert (ROOT / "reservoir_backend" / "results" / "manifest.py").exists()

def test_catalog_module_exists():
    assert (ROOT / "reservoir_backend" / "results" / "catalog.py").exists()

def test_export_module_exists():
    assert (ROOT / "reservoir_backend" / "results" / "export.py").exists()

def test_report_index_module_exists():
    assert (ROOT / "reservoir_backend" / "results" / "report_index.py").exists()

def test_result_manifest_json_serializable():
    json.dumps(_manifest().to_dict())

def test_result_manifest_required_keys():
    data = _manifest().to_dict()
    assert set(REQUIRED_MANIFEST_KEYS) <= set(data)

def test_invalid_manifest_missing_required_key_rejected():
    data = _manifest().to_dict()
    data.pop("unit")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_result_manifest(data)

def test_invalid_manifest_shape_rejected():
    data = _manifest().to_dict()
    data["shape"] = ["bad"]
    with pytest.raises(ValueError, match="shape entries"):
        validate_result_manifest(data)

def test_result_catalog_add_list_find():
    catalog = ResultCatalog()
    catalog.add(_manifest())
    catalog.add(_manifest(result_id="test_sw", result_type="saturation_field", field_name="sw"))
    assert len(catalog.list()) == 2
    assert catalog.find(result_id="test_pressure")[0]["field_name"] == "pressure"
    assert catalog.find(result_type="saturation_field")[0]["field_name"] == "sw"

def test_result_catalog_duplicate_result_id_rejected():
    catalog = ResultCatalog([_manifest()])
    with pytest.raises(ValueError, match="duplicate result_id"):
        catalog.add(_manifest())

def test_result_catalog_validate_paths():
    validation = _catalog().validate_paths(ROOT)
    assert validation["success"] is True
    assert validation["num_missing_paths"] == 0

def test_missing_report_path_generates_warning():
    catalog = ResultCatalog([_manifest(path="accuracy_reports/missing_result_contract_file.json")])
    validation = catalog.validate_paths(ROOT)
    assert validation["success"] is False
    assert validation["warnings"]

def test_report_path_index_default_paths():
    index = build_report_path_index(root=ROOT)
    assert index["num_reports"] >= 9
    assert index["num_missing_reports"] == 0

def test_report_path_index_missing_path_generates_warning(tmp_path):
    index = build_report_path_index(["missing.json"], root=tmp_path)
    assert index["success"] is False
    assert index["warnings"]

def test_json_manifest_export_generated(tmp_path):
    path = export_manifest_json(_catalog(), tmp_path / "manifest.json")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["num_results"] == 2

def test_csv_summary_export_generated(tmp_path):
    path = export_summary_csv(_catalog(), tmp_path / "summary.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["shape"] == "2x3x4"

def test_npz_field_export_generated(tmp_path):
    path = export_field_npz({"pressure": np.ones((2, 3)), "sw": np.zeros((2, 3))}, tmp_path / "fields.npz")
    data = np.load(path)
    assert data["pressure"].shape == (2, 3)
    assert data["sw"].shape == (2, 3)

def test_markdown_report_index_generated(tmp_path):
    index = build_report_path_index(root=ROOT)
    path = export_markdown_report_index(_catalog(), index, tmp_path / "index.md")
    text = path.read_text(encoding="utf-8")
    assert "Result Catalog" in text
    assert "Report Path Index" in text

def test_pressure_field_manifest_example():
    item = next(entry for entry in build_example_result_manifests() if entry["result_type"] == "pressure_field")
    assert item["field_name"] == "pressure"
    assert item["unit"] == "Pa"

def test_saturation_field_manifest_example():
    item = next(entry for entry in build_example_result_manifests() if entry["result_type"] == "saturation_field")
    assert item["field_name"] == "sw"
    assert item["unit"] == "fraction"

def test_parameter_fusion_manifest_example():
    item = next(entry for entry in build_example_result_manifests() if entry["result_type"] == "parameter_fusion_report")
    assert item["module"] == "M5"
    assert "parameter_fusion_benchmark_summary.json" in item["path"]

def test_experimental_data_qc_manifest_example():
    item = next(entry for entry in build_example_result_manifests() if entry["result_type"] == "experimental_data_qc")
    assert item["module"] == "M1"
    assert item["source_task"] == "TASK-008"

def test_benchmark_registry_manifest_example():
    item = next(entry for entry in build_example_result_manifests() if entry["result_type"] == "benchmark_registry")
    assert item["module"] == "M8"
    assert "benchmark_registry_summary.json" in item["path"]

def test_field_units_preserved():
    assert _manifest(unit="m2").to_dict()["unit"] == "m2"

def test_shape_conventions_preserved():
    assert _manifest(shape=[1, 2, 3]).to_dict()["shape"] == [1, 2, 3]

def test_dtype_preserved():
    assert _manifest(dtype="float32").to_dict()["dtype"] == "float32"

def test_metadata_preserved():
    assert _manifest(metadata={"source": "unit-test"}).to_dict()["metadata"]["source"] == "unit-test"

def test_warnings_preserved():
    assert _manifest(warnings=["warning-a"]).to_dict()["warnings"] == ["warning-a"]

def test_limitations_preserved():
    assert _manifest(limitations=["limitation-a"]).to_dict()["limitations"] == ["limitation-a"]

def test_frontend_field_contract_exists():
    assert (ROOT / "docs" / "frontend_field_contract.md").exists()

def test_frontend_field_contract_mentions_no_frontend():
    assert "No frontend implementation" in _read_text("docs/API_AND_DATA_CONTRACT.md")

def test_frontend_field_contract_mentions_no_udp():
    assert "No UDP implementation" in _read_text("docs/API_AND_DATA_CONTRACT.md")

def test_frontend_field_contract_mentions_no_rest_api():
    assert "No REST API implementation" in _read_text("docs/API_AND_DATA_CONTRACT.md")

def test_docs_result_manifest_exists():
    text = _read_text("docs/API_AND_DATA_CONTRACT.md")
    assert "Manifest Schema" in text
    assert "required fields" in text.lower()

def test_docs_result_export_pipeline_exists():
    text = _read_text("docs/result_export_pipeline.md")
    assert "Report Path Index" in text
    assert "No solver rewrite" in text

def test_readme_mentions_result_manifest():
    text = _read_text("README.md")
    assert "result manifest" in text.lower()
    assert "frontend field contract" in text.lower()

def test_traceability_mentions_result_export():
    text = _read_text("specs/10_requirement_traceability.md")
    assert "result export and frontend field contract" in text

def test_module_matrix_mentions_result_contract():
    text = _read_text("STATUS.md")
    assert "结果" in text

def test_result_manifest_summary_generated(tmp_path):
    summary = generate_result_manifest_summary(tmp_path, root=ROOT)
    assert summary["success"] is True
    assert (tmp_path / "result_manifest_summary.json").exists()
    assert (tmp_path / "result_manifest_summary.md").exists()

def test_result_manifest_summary_json_serializable(tmp_path):
    json.dumps(generate_result_manifest_summary(tmp_path, root=ROOT))

def test_result_manifest_summary_contains_export_formats(tmp_path):
    summary = generate_result_manifest_summary(tmp_path, root=ROOT)
    assert {"json", "csv", "npz", "markdown"} <= set(summary["export_formats_supported"])

def test_result_manifest_summary_preserves_report_paths(tmp_path):
    summary = generate_result_manifest_summary(tmp_path, root=ROOT)
    paths = {entry["path"] for entry in summary["report_path_index"]["reports"]}
    assert "accuracy_reports/benchmark_registry_summary.json" in paths
    assert "accuracy_reports/experimental_data_qc_summary.json" in paths

def test_pytest_all_pass_anchor():
    assert True
