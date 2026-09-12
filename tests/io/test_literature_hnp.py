"""Literature EXAMPLE CO2 huff-n-puff. Not a Jiyang GEM card."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case
from reservoir_backend.io.eos_load import load_eos_card
from reservoir_backend.io.well_history import WELL_HISTORY_COLUMNS, read_well_history_csv

_ROOT = Path(__file__).resolve().parents[2]
_FLUID = _ROOT / "examples" / "compositional" / "fixtures" / "literature_c1_nc10_co2.yaml"
_PUBLIC = _ROOT / "examples" / "compositional" / "fixtures" / "comp_c1c10co2.yaml"
_CASE = _ROOT / "examples" / "compositional" / "literature_hnp_1inj4prod.yaml"


def _assert_refuses_jiyang(text: str, data: dict) -> None:
    marker = str(data["marker"])
    assert marker == "literature"
    assert marker != "jiyang"
    assert "jiyang" not in marker.lower()
    assert "济阳" not in marker
    assert "现场" not in marker
    assert "Jiyang" not in marker
    assert "NOT a Jiyang GEM card" in text
    assert "NOT 济阳/现场" in text
    assert "NOT for CMG-GEM product nRMSE" in text


def test_literature_hnp_marker_refuses_jiyang_label() -> None:
    fluid_text = _FLUID.read_text(encoding="utf-8")
    fluid = yaml.safe_load(fluid_text)
    _assert_refuses_jiyang(fluid_text, fluid)
    case_text = _CASE.read_text(encoding="utf-8")
    case = yaml.safe_load(case_text)
    _assert_refuses_jiyang(case_text, case)
    assert "PLACEHOLDER" in case_text
    assert "literature-only" in case_text
    assert "inverse" not in case
    assert str(case.get("notes", "")).startswith("literature")


def test_literature_hnp_fluid_matches_public_c1_nc10_co2() -> None:
    eos = load_eos_card(_FLUID)
    published = load_eos_card(_PUBLIC)
    assert eos.nc == 3
    assert eos.names == ("CO2", "C1", "nC10")
    np.testing.assert_allclose(eos.tc, published.tc)
    np.testing.assert_allclose(eos.pc, published.pc)
    np.testing.assert_allclose(eos.omega, published.omega)
    np.testing.assert_allclose(eos.mw, published.mw)
    np.testing.assert_allclose(eos.kij, published.kij)


def test_literature_hnp_case_is_1inj4prod_inject_soak_produce() -> None:
    twin = load_case(_CASE)
    assert twin.physics.model == "compositional"
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.eos.names == ("CO2", "C1", "nC10")
    np.testing.assert_allclose(twin.physics.fluid.z_inj, [0.95, 0.04, 0.01], atol=1.0e-12)
    assert float(twin.physics.fluid.z_inj[0]) > 0.9
    assert [p.name for p in twin.ports] == ["INJ", "PROD1", "PROD2", "PROD3", "PROD4"]
    assert twin.ports[0].role == "injector"
    assert all(p.role == "producer" for p in twin.ports[1:])
    inj = next(c for c in twin.experiment.controls if c.port_name == "INJ" and c.kind == "rate")
    prod = next(c for c in twin.experiment.controls if c.port_name == "PROD1" and c.kind == "pressure")
    p_init = float(twin.physics.p_init)
    # inject: CO2-rich rate on, producers at p_init
    assert inj.value_at(0.5) > 0.0
    assert prod.value_at(0.5) == pytest.approx(p_init)
    # soak: injector off, producers still at p_init
    assert inj.value_at(1.5) == pytest.approx(0.0)
    assert prod.value_at(1.5) == pytest.approx(p_init)
    # produce: injector off, producers drawn down
    assert inj.value_at(3.0) == pytest.approx(0.0)
    assert prod.value_at(3.0) < p_init
    kinds = {(o.sensor_name, o.kind) for o in twin.experiment.observations}
    assert ("INJ", "bhp") in kinds
    assert ("INJ", "q_inj") in kinds
    assert ("PROD1", "bhp") in kinds
    case_text = _CASE.read_text(encoding="utf-8")
    assert "inject" in case_text.lower()
    assert "soak" in case_text.lower()
    assert "produce" in case_text.lower()


def test_missing_real_gem_card_still_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        _CASE.read_text(encoding="utf-8").replace(
            "file: fixtures/literature_c1_nc10_co2.yaml",
            "gem_deck: missing_jiyang.gem",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found|refuse invented"):
        load_case(p)


def test_literature_hnp_run_writes_well_history(tmp_path: Path) -> None:
    code = main(["run", str(_CASE), "--output", str(tmp_path)])
    assert code == 0
    csv_path = tmp_path / "well_history.csv"
    json_path = tmp_path / "well_history.json"
    assert csv_path.is_file()
    assert json_path.is_file()
    rows = read_well_history_csv(csv_path)
    assert rows
    assert list(rows[0].keys()) == list(WELL_HISTORY_COLUMNS)
    assert {r["well"] for r in rows} == {"INJ", "PROD1", "PROD2", "PROD3", "PROD4"}
    payload = json_path.read_text(encoding="utf-8")
    assert "nrmse" in payload.lower()
    assert "jiyang" not in payload.lower()
