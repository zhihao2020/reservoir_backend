"""Published EOS card loader. No invented Jiyang criticals."""

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.io.eos_load import load_eos_card

_ROOT = Path(__file__).resolve().parents[2]
_YAML = _ROOT / "examples" / "compositional" / "fixtures" / "comp_c1c10co2.yaml"
_INC = _ROOT / "examples" / "compositional" / "fixtures" / "comp_c1c10co2.inc"


def test_load_yaml_opm_1d_comp_numbers() -> None:
    eos = load_eos_card(_YAML)
    assert eos.nc == 3
    assert eos.names == ("CO2", "C1", "nC10")
    np.testing.assert_allclose(eos.tc, [304.128, 190.564, 617.7])
    np.testing.assert_allclose(eos.pc, np.array([73.773, 45.992, 21.03]) * 1.0e5)
    fl = flash_tp(eos, 8.0e6, 350.0, np.array([0.2, 0.5, 0.3]))
    rec = fl.vapor_frac * fl.y + (1.0 - fl.vapor_frac) * fl.x
    np.testing.assert_allclose(rec, np.array([0.2, 0.5, 0.3]), atol=1.0e-7)


def test_load_keyword_inc_matches_yaml() -> None:
    a = load_eos_card(_YAML)
    b = load_eos_card(_INC)
    np.testing.assert_allclose(a.tc, b.tc)
    np.testing.assert_allclose(a.pc, b.pc)
    np.testing.assert_allclose(a.omega, b.omega)
    np.testing.assert_allclose(a.mw, b.mw)


def test_missing_card_refuses() -> None:
    with pytest.raises(ValueError, match="not found|refuse"):
        load_eos_card(Path("examples/compositional/fixtures/no_such_card.yaml"))


def test_incomplete_yaml_refuses(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("names: [C1]\ntc_k: [190.0]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs Tc|refuse"):
        load_eos_card(p)


def test_case_yaml_loads_published_card(tmp_path: Path) -> None:
    from reservoir_backend.io.case import load_case

    yml = tmp_path / "card_case.yaml"
    yml.write_text(
        """
geometry: {size_m: [2, 2, 1]}
grid: {type: cartesian, nx: 2, ny: 2, nz: 1}
physics:
  model: compositional
  fluid: {file: examples/compositional/fixtures/comp_c1c10co2.yaml}
  z_init: [0.2, 0.5, 0.3]
  z_inj: [1.0, 0.0, 0.0]
  p_init: 1.0e7
  dt_init: 1.0
  dt_max: 2.0
rock: {porosity: 0.2}
ports:
  - {name: INJ, role: injector, control: rate, ijk: [[1,1,1]]}
  - {name: PROD, role: producer, control: pressure, ijk: [[2,2,1]]}
experiment:
  controls:
    - {port: INJ, kind: rate, times: [0, 1], values: [0.01, 0.01]}
    - {port: PROD, kind: pressure, times: [0, 1], values: [9.0e6, 9.0e6]}
""",
        encoding="utf-8",
    )
    twin = load_case(yml)
    assert twin.physics.fluid.eos.nc == 3
    assert twin.physics.fluid.eos.names[0] == "CO2"
