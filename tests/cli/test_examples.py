from pathlib import Path

import pytest

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case

LAB_V1 = Path("examples/lab_v1")
LAB_CF = Path("examples/lab/lab_cf.yaml")


def test_lab_v1_product_is_dpdp_cf_tmf() -> None:
    twin = load_case(LAB_V1 / "case.yaml")
    assert twin.uses_dpdp()
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "LogCfTmfParameterization"
    assert twin.inverse.algorithm == "esmda"
    assert not twin.experiment.observations


def test_lab_v1_five_file_contract() -> None:
    assert (LAB_V1 / "case.yaml").is_file()
    assert (LAB_V1 / "pvt.yaml").is_file()
    assert (LAB_V1 / "wells.yaml").is_file()
    assert (LAB_V1 / "controls.csv").is_file()
    twin = load_case(LAB_V1 / "case_dev.yaml")
    assert [p.name for p in twin.ports] == ["INJ", "PROD"]
    assert twin.physics.fluid is not None


def test_apply_without_demo_or_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no observations"):
        main(["apply", str(LAB_CF), "--output", str(tmp_path / "out")])
