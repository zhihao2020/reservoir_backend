"""Literature EXAMPLE fluid + tiny run. Not a Jiyang GEM card."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case
from reservoir_backend.io.eos_load import load_eos_card
from reservoir_backend.io.well_history import WELL_HISTORY_COLUMNS, read_well_history_csv

_ROOT = Path(__file__).resolve().parents[2]
_FLUID = _ROOT / "examples" / "compositional" / "fixtures" / "literature_c1_nc10.yaml"
_CASE = _ROOT / "examples" / "compositional" / "literature_tiny.yaml"
_PVT = _ROOT / "examples" / "lab_v1" / "pvt.yaml"


def test_literature_marker_refuses_jiyang_label() -> None:
    text = _FLUID.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert data["marker"] == "literature"
    assert data["marker"] != "jiyang"
    assert "jiyang" not in str(data["marker"]).lower()
    assert "济阳" not in str(data.get("marker", ""))
    assert "NOT a Jiyang GEM card" in text
    assert "NOT 济阳/现场" in text
    assert "NOT for CMG-GEM product nRMSE" in text
    case_text = _CASE.read_text(encoding="utf-8")
    case = yaml.safe_load(case_text)
    assert case["marker"] == "literature"
    assert case["marker"] != "jiyang"
    assert "NOT a Jiyang GEM card" in case_text
    assert "NOT for CMG-GEM product nRMSE" in case_text
    assert "inverse" not in case


def test_literature_fluid_load_matches_published_c1_nc10() -> None:
    eos = load_eos_card(_FLUID)
    published = load_eos_card(_PVT)
    assert eos.nc == 2
    assert eos.names == ("C1", "nC10")
    np.testing.assert_allclose(eos.tc, published.tc)
    np.testing.assert_allclose(eos.pc, published.pc)
    np.testing.assert_allclose(eos.omega, published.omega)
    np.testing.assert_allclose(eos.mw, published.mw)
    np.testing.assert_allclose(eos.kij, published.kij)


def test_literature_tiny_case_loads() -> None:
    twin = load_case(_CASE)
    assert twin.physics.model == "compositional"
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.eos.names == ("C1", "nC10")
    assert [p.name for p in twin.ports] == ["INJ", "PROD"]
    kinds = {(o.sensor_name, o.kind) for o in twin.experiment.observations}
    assert ("INJ", "bhp") in kinds
    assert ("PROD", "bhp") in kinds


def test_missing_real_gem_card_still_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        _CASE.read_text(encoding="utf-8").replace(
            "file: fixtures/literature_c1_nc10.yaml",
            "gem_deck: missing_jiyang.gem",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found|refuse invented"):
        load_case(p)


def test_literature_tiny_run_writes_well_history(tmp_path: Path) -> None:
    code = main(["run", str(_CASE), "--output", str(tmp_path)])
    assert code == 0
    csv_path = tmp_path / "well_history.csv"
    json_path = tmp_path / "well_history.json"
    assert csv_path.is_file()
    assert json_path.is_file()
    rows = read_well_history_csv(csv_path)
    assert rows
    assert list(rows[0].keys()) == list(WELL_HISTORY_COLUMNS)
    assert {r["well"] for r in rows} >= {"INJ", "PROD"}
