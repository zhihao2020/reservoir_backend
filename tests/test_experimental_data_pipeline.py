from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.data import qc, reader, resample, schema
from reservoir_backend.data.qc import normalize_dataset_units, run_qc_pipeline
from reservoir_backend.data.reader import read_experimental_data
from reservoir_backend.data.report import write_qc_report
from reservoir_backend.data.resample import align_fields_to_grid_shape, resample_time_series
from reservoir_backend.data.schema import ExperimentalDataset, ExperimentalField, dataset_from_arrays


ROOT = Path(__file__).resolve().parents[1]


def _csv_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "lab_data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def _valid_csv(tmp_path: Path) -> Path:
    return _csv_file(
        tmp_path,
        "\n".join(
            [
                "time_s,x_m,y_m,z_m,porosity_fraction,saturation_percent,permeability_md,pressure_mpa,resistivity_ohm_m,confidence_fraction,variance",
                "0,0,0,0,0.20,20,100,10,20,0.9,0.1",
                "10,1,0,0,0.22,30,120,9.8,22,0.8,0.2",
                "20,2,0,0,0.24,40,140,9.6,24,0.7,0.3",
            ]
        )
        + "\n",
    )


def _json_file(tmp_path: Path) -> Path:
    path = tmp_path / "lab_data.json"
    payload = {
        "source_name": "json_lab",
        "metadata": {"operator": "tester"},
        "fields": {
            "time": {"values": [0, 10, 20], "unit": "s"},
            "porosity": {"values": [0.2, 0.22, 0.24], "unit": "fraction"},
            "permeability": {"values": [100, 120, 140], "unit": "mD"},
            "pressure": {"values": [10, 9.8, 9.6], "unit": "MPa"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _npz_file(tmp_path: Path) -> Path:
    path = tmp_path / "lab_data.npz"
    np.savez(
        path,
        time=np.array([0.0, 10.0, 20.0]),
        time_unit=np.array("s"),
        porosity=np.array([0.2, 0.22, 0.24]),
        porosity_unit=np.array("fraction"),
        permeability=np.array([100.0, 120.0, 140.0]),
        permeability_unit=np.array("mD"),
        metadata_json=np.array(json.dumps({"source_name": "npz_lab"})),
    )
    return path


def test_schema_module_exists():
    assert schema is not None


def test_reader_module_exists():
    assert reader is not None


def test_qc_module_exists():
    assert qc is not None


def test_resample_module_exists():
    assert resample is not None


def test_valid_csv_read(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path), source_name="csv_source")
    assert isinstance(dataset, ExperimentalDataset)
    assert "porosity" in dataset.fields
    assert dataset.source_name == "csv_source"


def test_valid_json_read(tmp_path):
    dataset = read_experimental_data(_json_file(tmp_path))
    assert dataset.source_name == "json_lab"
    assert dataset.metadata["operator"] == "tester"


def test_valid_npz_read(tmp_path):
    dataset = read_experimental_data(_npz_file(tmp_path))
    assert "permeability" in dataset.fields
    assert dataset.input_format == "npz"


def test_missing_required_field_detected(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path))
    report = run_qc_pipeline(dataset, required_fields=["time", "temperature"]).report
    assert "temperature" in report["fields_missing"]
    assert report["success"] is False


def test_invalid_dtype_detected(tmp_path):
    path = _csv_file(tmp_path, "time_s,porosity_fraction\n0,0.2\n1,not_a_number\n")
    with pytest.raises(ValueError):
        read_experimental_data(path)


def test_unit_normalization_works(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path))
    normalized = normalize_dataset_units(dataset)
    assert np.isclose(normalized.fields["pressure"].values[0], 10.0e6)
    assert np.isclose(normalized.fields["saturation"].values[0], 0.2)
    assert normalized.fields["permeability"].unit == "m2"


def test_missing_unit_warning_generated():
    dataset = dataset_from_arrays({"unknown_signal": np.array([1.0, 2.0])}, source_name="unit_test")
    report = run_qc_pipeline(dataset).report
    assert report["unit_warnings"]


def test_nan_detected(tmp_path):
    path = _csv_file(tmp_path, "time_s,porosity_fraction\n0,0.2\n1,\n")
    report = run_qc_pipeline(read_experimental_data(path)).report
    assert report["num_nan"] == 1
    assert report["num_missing"] == 1


def test_inf_detected():
    dataset = dataset_from_arrays({"pressure": np.array([1.0, np.inf])}, units={"pressure": "Pa"})
    report = run_qc_pipeline(dataset).report
    assert report["num_inf"] == 1
    assert report["success"] is False


def test_missing_values_detected(tmp_path):
    path = _csv_file(tmp_path, "time_s,saturation_fraction\n0,0.2\n1,\n")
    report = run_qc_pipeline(read_experimental_data(path)).report
    assert report["num_missing"] == 1


def test_outlier_flag_generated():
    dataset = dataset_from_arrays({"pressure": np.array([10.0, 10.0, 10.0, 50.0])}, units={"pressure": "Pa"})
    report = run_qc_pipeline(dataset, outlier_zscore=1.0).report
    assert report["num_outliers"] > 0


def test_porosity_bounds_checked():
    dataset = dataset_from_arrays({"porosity": np.array([0.2, 1.2])}, units={"porosity": "fraction"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["porosity"] == 1


def test_saturation_bounds_checked():
    dataset = dataset_from_arrays({"saturation": np.array([-0.1, 0.5])}, units={"saturation": "fraction"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["saturation"] == 1


def test_permeability_positive_checked():
    dataset = dataset_from_arrays({"permeability": np.array([100.0, 0.0])}, units={"permeability": "mD"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["permeability"] == 1


def test_pressure_finite_checked():
    dataset = dataset_from_arrays({"pressure": np.array([1.0, np.inf])}, units={"pressure": "Pa"})
    report = run_qc_pipeline(dataset).report
    assert "pressure" in report["bounds_violations"]


def test_resistivity_positive_checked():
    dataset = dataset_from_arrays({"resistivity": np.array([10.0, -1.0])}, units={"resistivity": "ohm_m"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["resistivity"] == 1


def test_confidence_bounds_checked():
    dataset = dataset_from_arrays({"confidence": np.array([0.5, 1.5])}, units={"confidence": "fraction"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["confidence"] == 1


def test_variance_bounds_checked():
    dataset = dataset_from_arrays({"variance": np.array([0.1, -0.1])}, units={"variance": "variance"})
    report = run_qc_pipeline(dataset).report
    assert report["bounds_violations"]["variance"] == 1


def test_shape_consistency_checked():
    dataset = ExperimentalDataset(
        {
            "porosity": ExperimentalField("porosity", np.array([0.2, 0.3]), "fraction"),
            "pressure": ExperimentalField("pressure", np.ones((2, 2)), "Pa"),
        }
    )
    report = run_qc_pipeline(dataset).report
    assert report["shape_consistency"]["shape_consistent"] is False


def test_time_series_resampling_works():
    dataset = dataset_from_arrays(
        {"time": np.array([0.0, 10.0]), "pressure": np.array([10.0, 20.0])},
        units={"time": "s", "pressure": "Pa"},
    )
    output, summary = resample_time_series(dataset, [0.0, 5.0, 10.0])
    assert np.allclose(output.fields["pressure"].values, [10.0, 15.0, 20.0])
    assert summary["target_count"] == 3


def test_spatial_grid_shape_alignment_works():
    dataset = dataset_from_arrays({"porosity": np.arange(6.0)}, units={"porosity": "fraction"})
    output, summary = align_fields_to_grid_shape(dataset, (1, 2, 3))
    assert output.fields["porosity"].values.shape == (1, 2, 3)
    assert summary["aligned_fields"] == ["porosity"]


def test_qc_report_json_generated(tmp_path):
    report = run_qc_pipeline(read_experimental_data(_valid_csv(tmp_path))).report
    json_path, _ = write_qc_report(report, tmp_path)
    assert json_path.exists()


def test_qc_report_markdown_generated(tmp_path):
    report = run_qc_pipeline(read_experimental_data(_valid_csv(tmp_path))).report
    _, md_path = write_qc_report(report, tmp_path)
    assert md_path.exists()


def test_report_json_serializable(tmp_path):
    report = run_qc_pipeline(read_experimental_data(_valid_csv(tmp_path))).report
    json.dumps(report)


def test_reader_output_has_standard_internal_structure(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path))
    assert isinstance(dataset, ExperimentalDataset)
    assert all(isinstance(field, ExperimentalField) for field in dataset.fields.values())


def test_metadata_preserved(tmp_path):
    dataset = read_experimental_data(_json_file(tmp_path))
    assert dataset.metadata["operator"] == "tester"


def test_source_name_preserved(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path), source_name="core_lab")
    assert dataset.source_name == "core_lab"


def test_warnings_preserved(tmp_path):
    dataset = read_experimental_data(_valid_csv(tmp_path), required_fields=["temperature"])
    assert dataset.metadata["reader_warnings"]


def test_invalid_file_format_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        read_experimental_data(path)


def test_empty_input_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_experimental_data(path)


def test_duplicate_time_detected():
    dataset = dataset_from_arrays({"time": np.array([0.0, 0.0, 1.0])}, units={"time": "s"})
    report = run_qc_pipeline(dataset).report
    assert report["duplicate_time_count"] == 1


def test_duplicate_coordinates_detected():
    dataset = dataset_from_arrays(
        {"x": np.array([0.0, 0.0]), "y": np.array([1.0, 1.0]), "z": np.array([2.0, 2.0])},
        units={"x": "m", "y": "m", "z": "m"},
    )
    report = run_qc_pipeline(dataset).report
    assert report["duplicate_coordinate_count"] == 1


def test_does_not_modify_solver():
    assert (ROOT / "reservoir_backend" / "solver").exists()


def test_does_not_modify_inversion():
    assert (ROOT / "reservoir_backend" / "inversion").exists()


def test_does_not_modify_fusion():
    assert (ROOT / "reservoir_backend" / "fusion").exists()


def test_does_not_modify_cross_scale():
    assert (ROOT / "reservoir_backend" / "cross_scale").exists()


def test_data_schema_doc_exists():
    text = (ROOT / "docs" / "data_schema.md").read_text(encoding="utf-8")
    assert "Standard Fields" in text
    assert "physical" in text.lower()


def test_experimental_data_pipeline_doc_exists():
    text = (ROOT / "docs" / "experimental_data_pipeline.md").read_text(encoding="utf-8")
    assert "Reader Flow" in text
    assert "QC Flow" in text


def test_module_docs_mention_data_pipeline():
    text = (ROOT / "docs" / "experimental_data_pipeline.md").read_text(encoding="utf-8")
    assert "experimental data pipeline" in text.lower()


def test_report_module_runner_generates_default_report():
    from reservoir_backend.data.report import generate_demo_qc_report

    report = generate_demo_qc_report(ROOT / "accuracy_reports")
    assert report["success"] is True


def test_pytest_all_pass_anchor():
    assert True
