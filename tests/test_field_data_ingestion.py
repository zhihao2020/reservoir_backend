from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.field_data.ingestion import (
    build_case_input_summary,
    duplicate_values,
    read_pressure_history,
    read_production_history,
    read_property_field,
    read_schedule_csv,
    read_well_table,
    validate_field_records,
    validate_time_ordering,
    write_field_data_ingestion_report,
)
from reservoir_backend.field_data.report import generate_demo_field_data_ingestion_report


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    well_table = _write_csv(
        tmp_path / "well_table.csv",
        ["well_id", "well_type", "i", "j", "k", "status"],
        [["I1", "injector", 0, 0, 0, "open"], ["P1", "producer", 4, 0, 0, "open"]],
    )
    production = _write_csv(
        tmp_path / "production_history.csv",
        ["well_id", "time", "oil_rate", "water_rate", "gas_rate"],
        [["P1", 0.0, 10.0, 0.0, 0.0], ["P1", 1.0, 9.0, 1.0, 0.0]],
    )
    pressure = _write_csv(tmp_path / "pressure_history.csv", ["well_id", "time", "pressure", "unit"], [["P1", 0.0, 10.0, "MPa"], ["P1", 1.0, 9.8, "MPa"]])
    schedule = _write_csv(
        tmp_path / "schedule.csv",
        ["well_id", "time", "control_type", "target", "unit", "status"],
        [["I1", 0.0, "rate", 100.0, "m3/day", "open"], ["P1", 0.0, "bhp", 9.0, "MPa", "open"]],
    )
    property_field = tmp_path / "property_field.npz"
    np.savez(
        property_field,
        porosity=np.array([0.2, 0.25]),
        porosity_unit=np.array("fraction"),
        permeability=np.array([100.0, 120.0]),
        permeability_unit=np.array("mD"),
    )
    return {
        "well_table": well_table,
        "production_history": production,
        "pressure_history": pressure,
        "schedule": schedule,
        "property_field": property_field,
    }


def test_well_table_schema(tmp_path: Path) -> None:
    rows = read_well_table(_fixtures(tmp_path)["well_table"])
    assert rows[0]["well_type"] == "injector"
    assert rows[0]["i"] == 0.0


def test_production_history_schema(tmp_path: Path) -> None:
    rows = read_production_history(_fixtures(tmp_path)["production_history"])
    assert rows[1]["water_rate"] == 1.0


def test_pressure_history_schema(tmp_path: Path) -> None:
    rows = read_pressure_history(_fixtures(tmp_path)["pressure_history"])
    assert rows[0]["pressure"] == 10.0


def test_schedule_csv_schema(tmp_path: Path) -> None:
    rows = read_schedule_csv(_fixtures(tmp_path)["schedule"])
    assert {row["control_type"] for row in rows} == {"rate", "bhp"}


def test_property_field_npz_qc(tmp_path: Path) -> None:
    result = read_property_field(_fixtures(tmp_path)["property_field"], required_fields=["porosity", "permeability"])
    assert result["qc_report"]["success"] is True


def test_unit_validation(tmp_path: Path) -> None:
    result = read_property_field(_fixtures(tmp_path)["property_field"], required_fields=["porosity", "permeability"])
    fields = result["dataset"].fields
    assert fields["porosity"].unit == "fraction"
    assert fields["permeability"].unit == "m2"


def test_missing_field_errors(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad_wells.csv", ["well_id", "well_type"], [["I1", "injector"]])
    with pytest.raises(ValueError, match="missing required fields"):
        read_well_table(path)


def test_time_ordering_validation() -> None:
    report = validate_time_ordering([{"well_id": "P1", "time": 1.0}, {"well_id": "P1", "time": 0.0}])
    assert report["success"] is False
    assert report["errors"]


def test_time_ordering_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad_prod.csv", ["well_id", "time", "oil_rate", "water_rate"], [["P1", 1.0, 1.0, 0.0], ["P1", 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="time ordering"):
        read_production_history(path)


def test_duplicate_well_id_detection(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "dup_wells.csv",
        ["well_id", "well_type", "i", "j", "k"],
        [["I1", "injector", 0, 0, 0], ["I1", "producer", 1, 0, 0]],
    )
    with pytest.raises(ValueError, match="duplicate well id"):
        read_well_table(path)


def test_duplicate_values_helper() -> None:
    assert duplicate_values(["A", "B", "A", "B"]) == ["A", "B"]


def test_invalid_control_type_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad_schedule.csv", ["well_id", "time", "control_type", "target", "unit", "status"], [["I1", 0.0, "foo", 1.0, "m3/day", "open"]])
    with pytest.raises(ValueError, match="invalid control_type"):
        read_schedule_csv(path)


def test_invalid_well_type_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad_well_type.csv", ["well_id", "well_type", "i", "j", "k"], [["I1", "observer", 0, 0, 0]])
    with pytest.raises(ValueError, match="invalid well_type"):
        read_well_table(path)


def test_nonfinite_numeric_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        validate_field_records([{"well_id": "P1", "time": "inf", "oil_rate": 1.0, "water_rate": 0.0}], "production_history")


def test_qc_summary_generation(tmp_path: Path) -> None:
    summary = build_case_input_summary(**_fixtures(tmp_path))
    assert summary["success"] is True
    assert summary["property_field_qc"]["fields_detected"] == ["permeability", "porosity"]


def test_case_input_summary_generation(tmp_path: Path) -> None:
    summary = build_case_input_summary(**_fixtures(tmp_path))
    assert summary["well_table"]["num_wells"] == 2
    assert summary["schedule"]["control_types"] == ["bhp", "rate"]


def test_report_json_markdown_generated(tmp_path: Path) -> None:
    summary = build_case_input_summary(**_fixtures(tmp_path))
    paths = write_field_data_ingestion_report(summary, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert "Field Data Ingestion Summary" in Path(paths["markdown"]).read_text(encoding="utf-8")


def test_report_json_serializable(tmp_path: Path) -> None:
    summary = build_case_input_summary(**_fixtures(tmp_path))
    json.dumps(summary)


def test_demo_report_runner(tmp_path: Path) -> None:
    summary = generate_demo_field_data_ingestion_report(tmp_path)
    assert summary["success"] is True
    assert Path(summary["report_json_path"]).exists()


def test_las_eclipse_resqml_are_not_claimed(tmp_path: Path) -> None:
    summary = build_case_input_summary(**_fixtures(tmp_path))
    text = "\n".join(summary["limitations"])
    assert "LAS, Eclipse deck, and RESQML are roadmap items only." in text
    assert "No database service." in text
