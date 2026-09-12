from pathlib import Path

import json
import pytest

from reservoir_backend.cli.main import cmd_run, cmd_simulate, main
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
    with pytest.raises(SystemExit, match="does not invert"):
        main(["forecast", "examples/lab_v1/case_dev.yaml", "--output", str(tmp_path)])


def test_run_is_simulate_alias() -> None:
    assert cmd_run is cmd_simulate


def test_run_help_is_forward_only(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--self-check" not in text
    assert "--demo" not in text
    assert "case" in text.lower()


def test_run_cmg_habit_example(tmp_path: Path) -> None:
    code = main(["run", "examples/run/case.yaml", "--output", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["n_steps"] >= 1
    assert len(report["times_s"]) >= 1
    assert (tmp_path / "simulate.json").is_file()
    assert (tmp_path / "pressure.npy").is_file()


def test_run_existing_lab_cf(tmp_path: Path) -> None:
    code = main(["run", "examples/lab/lab_cf.yaml", "--output", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["n_steps"] >= 1
    twin = load_case("examples/lab/lab_cf.yaml")
    assert twin.uses_dpdp()
    assert [p.name for p in twin.ports] == ["INJ", "PROD"]
