from pathlib import Path

import json

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case


def test_validate_lab_case(tmp_path: Path) -> None:
    code = main(["validate", "examples/lab/lab_30cm.yaml", "--output", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "validate.json").is_file()
    report = json.loads((tmp_path / "validate.json").read_text(encoding="utf-8"))
    assert report["parameterization_class"] == "RegionParameterization"
    assert report["n_theta"] == 2
    twin = load_case("examples/lab/lab_30cm.yaml")
    assert all(p.cell_ids.size == twin.grid.nz for p in twin.ports)
