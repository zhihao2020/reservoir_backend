from pathlib import Path

import json

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case


def test_validate_lab_v1_dev(tmp_path: Path) -> None:
    code = main(["validate", "examples/lab_v1/case_dev.yaml", "--output", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "validate.json").is_file()
    report = json.loads((tmp_path / "validate.json").read_text(encoding="utf-8"))
    assert report["parameterization_class"] == "LogCfTmfParameterization"
    assert report["n_theta"] == 2
    twin = load_case("examples/lab_v1/case_dev.yaml")
    assert twin.uses_dpdp()
    assert twin.physics.fluid is not None


def test_forecast_requires_posterior(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(SystemExit, match="does not invert"):
        main(["forecast", "examples/lab_v1/case_dev.yaml", "--output", str(tmp_path)])
