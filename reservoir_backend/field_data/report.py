"""Report runner for IND-002 field data ingestion."""

from __future__ import annotations

import csv
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from reservoir_backend.field_data.ingestion import build_case_input_summary, write_field_data_ingestion_report


def generate_demo_field_data_ingestion_report(output_dir: str | Path = "accuracy_reports") -> dict[str, object]:
    """Generate a small deterministic field-data ingestion report."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="field_data_ingestion_demo_") as temp_root:
        paths = _write_demo_inputs(Path(temp_root))
        summary = build_case_input_summary(**paths)
    report_paths = write_field_data_ingestion_report(summary, root)
    summary["report_json_path"] = report_paths["json"]
    summary["report_markdown_path"] = report_paths["markdown"]
    return summary


def _write_demo_inputs(root: Path) -> dict[str, Path]:
    well_table = root / "well_table.csv"
    production = root / "production_history.csv"
    pressure = root / "pressure_history.csv"
    schedule = root / "schedule.csv"
    property_field = root / "property_field.npz"
    _write_rows(
        well_table,
        ["well_id", "well_type", "i", "j", "k", "status"],
        [
            ["I1", "injector", 0, 0, 0, "open"],
            ["P1", "producer", 5, 0, 0, "open"],
        ],
    )
    _write_rows(
        production,
        ["well_id", "time", "oil_rate", "water_rate", "gas_rate"],
        [["P1", 0.0, 10.0, 0.0, 0.0], ["P1", 1.0, 9.5, 0.5, 0.0]],
    )
    _write_rows(pressure, ["well_id", "time", "pressure", "unit"], [["P1", 0.0, 10.0, "MPa"], ["P1", 1.0, 9.8, "MPa"]])
    _write_rows(
        schedule,
        ["well_id", "time", "control_type", "target", "unit", "status"],
        [["I1", 0.0, "rate", 100.0, "m3/day", "open"], ["P1", 0.0, "bhp", 9.0, "MPa", "open"]],
    )
    np.savez(
        property_field,
        porosity=np.array([0.2, 0.25, 0.3], dtype=float),
        porosity_unit=np.array("fraction"),
        permeability=np.array([100.0, 120.0, 140.0], dtype=float),
        permeability_unit=np.array("mD"),
    )
    return {
        "well_table": well_table,
        "production_history": production,
        "pressure_history": pressure,
        "schedule": schedule,
        "property_field": property_field,
    }


def _write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    summary = generate_demo_field_data_ingestion_report()
    print(json.dumps({"success": summary["success"], "report": summary["report_json_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
