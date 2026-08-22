"""EXAMPLE GEM-like text card → YAML EOS deck. Not FIM, not field-validated."""

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.eos import (
    DEFAULT_EXAMPLE_FLUID_YAML,
    DEFAULT_EXAMPLE_GEM_CARD,
    EXAMPLE_C1_C7PLUS_CO2_GEM,
    load_eos_mixture_gem,
    load_eos_mixture_yaml,
    parse_gem_card,
    resolve_gem_deck,
)

_TIGHT = dict(rtol=1e-12, atol=0.0)


def test_example_gem_card_round_trips_to_yaml_mixture() -> None:
    assert DEFAULT_EXAMPLE_GEM_CARD.is_file()
    text = DEFAULT_EXAMPLE_GEM_CARD.read_text(encoding="utf-8")
    assert "EXAMPLE" in text
    assert "NOT a Jiyang" in text
    assert "*EOS" in text and "*COMP" in text
    deck = parse_gem_card(DEFAULT_EXAMPLE_GEM_CARD)
    assert [row["name"] for row in deck["components"]] == ["C1", "CO2"]
    assert "Tc" in deck["components"][0] and "Pc" in deck["components"][0]
    assert "omega" in deck["components"][0]
    gem = load_eos_mixture_gem(DEFAULT_EXAMPLE_GEM_CARD)
    yaml_mix = load_eos_mixture_yaml(DEFAULT_EXAMPLE_FLUID_YAML).subset(["C1", "CO2"])
    assert gem.names == yaml_mix.names == ("C1", "CO2")
    np.testing.assert_allclose(gem.Tc, yaml_mix.Tc, **_TIGHT)
    np.testing.assert_allclose(gem.Pc, yaml_mix.Pc, **_TIGHT)
    np.testing.assert_allclose(gem.omega, yaml_mix.omega, **_TIGHT)
    assert gem.Mw is not None and yaml_mix.Mw is not None
    np.testing.assert_allclose(gem.Mw, yaml_mix.Mw, **_TIGHT)
    np.testing.assert_allclose(gem.kij, yaml_mix.kij, **_TIGHT)
    assert "EXAMPLE" in gem.marker
    assert "NOT a Jiyang" in gem.marker


def test_eight_comp_gem_matches_yaml_mixture() -> None:
    """Full 8-comp EXAMPLE .gem uses the same Tc/Pc/ω/kij as the YAML deck."""
    assert EXAMPLE_C1_C7PLUS_CO2_GEM.is_file()
    text = EXAMPLE_C1_C7PLUS_CO2_GEM.read_text(encoding="utf-8")
    assert "EXAMPLE" in text
    assert "NOT a Jiyang" in text
    assert "*EOS" in text and "*COMP" in text
    gem = load_eos_mixture_gem(EXAMPLE_C1_C7PLUS_CO2_GEM)
    yaml_mix = load_eos_mixture_yaml(DEFAULT_EXAMPLE_FLUID_YAML)
    assert gem.names == yaml_mix.names
    np.testing.assert_allclose(gem.Tc, yaml_mix.Tc, **_TIGHT)
    np.testing.assert_allclose(gem.Pc, yaml_mix.Pc, **_TIGHT)
    np.testing.assert_allclose(gem.omega, yaml_mix.omega, **_TIGHT)
    np.testing.assert_allclose(gem.kij, yaml_mix.kij, **_TIGHT)
    gem_hz = gem.subset(["C1", "CO2"])
    yaml_hz = yaml_mix.subset(["C1", "CO2"])
    np.testing.assert_allclose(gem_hz.Tc, yaml_hz.Tc, **_TIGHT)
    np.testing.assert_allclose(gem_hz.Pc, yaml_hz.Pc, **_TIGHT)
    np.testing.assert_allclose(gem_hz.omega, yaml_hz.omega, **_TIGHT)
    np.testing.assert_allclose(gem_hz.kij, yaml_hz.kij, **_TIGHT)


def test_missing_gem_card_file_errors() -> None:
    missing = Path("/tmp/does_not_exist_example.gem")
    with pytest.raises(FileNotFoundError, match="EXAMPLE GEM card not found"):
        load_eos_mixture_gem(missing)
    with pytest.raises(FileNotFoundError, match="refusing to invent GEM/Jiyang criticals"):
        resolve_gem_deck(missing)


def test_missing_gem_tcrit_errors(tmp_path: Path) -> None:
    raw = DEFAULT_EXAMPLE_GEM_CARD.read_text(encoding="utf-8")
    bad = tmp_path / "no_tcrit.gem"
    bad.write_text(raw.replace("*TCRIT", "*SKIP_TCRIT"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required keyword \*TCRIT"):
        load_eos_mixture_gem(bad)
    with pytest.raises(ValueError, match="Refusing to invent criticals"):
        parse_gem_card(bad)
