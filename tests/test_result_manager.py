from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.io.result_manager import ResultManager


def test_create_case_dir(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    case_dir = manager.create_case_dir("case_a")
    assert case_dir.exists()
    assert case_dir.name == "case_a"


def test_save_and_load_npy(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    array = np.arange(6).reshape(2, 3)
    path = manager.save_npy("array", array)
    assert np.array_equal(np.load(path), array)


def test_save_json_report(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    path = manager.save_json("report", {"value": np.float64(1.5)})
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["value"] == 1.5


def test_save_csv_production_curve(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    path = manager.save_csv("production_curve", [{"step": 1, "water_cut": 0.2}])
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["step"] == "1"
    assert rows[0]["water_cut"] == "0.2"


def test_save_field3d(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    field = Field3D.from_constant(grid, 3.0)
    path = manager.save_field("field", field)
    assert np.allclose(np.load(path), field.values)


def test_case_summary_saved(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    path = manager.save_case_summary({"case_id": "case_a", "success": True})
    assert path.name == "case_summary.json"
    assert json.loads(path.read_text(encoding="utf-8"))["success"] is True


def test_list_case_outputs(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    manager.save_json("a", {"x": 1})
    outputs = manager.list_case_outputs("case_a")
    assert [path.name for path in outputs] == ["a.json"]


def test_validate_required_outputs_success(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    manager.save_json("a", {"x": 1})
    assert manager.validate_required_outputs("case_a", ["a.json"]) is True


def test_validate_required_outputs_missing_file_raises(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    with pytest.raises(FileNotFoundError):
        manager.validate_required_outputs("case_a", ["missing.json"])


def test_result_manager_does_not_overwrite_other_case(tmp_path) -> None:
    manager = ResultManager(tmp_path)
    manager.create_case_dir("case_a")
    manager.save_json("report", {"case": "a"})
    manager.create_case_dir("case_b")
    manager.save_json("report", {"case": "b"})
    assert json.loads((tmp_path / "case_a" / "report.json").read_text())["case"] == "a"
    assert json.loads((tmp_path / "case_b" / "report.json").read_text())["case"] == "b"
