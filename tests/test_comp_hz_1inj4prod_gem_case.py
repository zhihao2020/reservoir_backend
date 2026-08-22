"""EXAMPLE HZ 1+4 case via fluid.gem_deck. Not FIM, not a Jiyang card."""

import csv
import io

import numpy as np

from reservoir_backend.comp.case_run import (
    DEFAULT_CASE,
    GEM_CASE,
    format_metrics,
    load_case_mixture,
    load_case_yaml,
    main,
)

_TIGHT = dict(rtol=1e-12, atol=0.0)


def test_gem_case_yaml_uses_gem_deck_not_eos_yaml() -> None:
    cfg = load_case_yaml(GEM_CASE)
    assert cfg["marker"] == "EXAMPLE"
    assert cfg["pattern"] == "hz_1inj4prod"
    assert cfg["n_cycles"] == 2
    assert cfg["fluid"]["gem_deck"] == "example_c1_c7plus_co2.gem"
    assert "eos_yaml" not in cfg["fluid"]
    text = GEM_CASE.read_text(encoding="utf-8")
    assert "not a Jiyang" in text.lower() or "NOT a Jiyang" in text
    yaml_cfg = load_case_yaml(DEFAULT_CASE)
    assert yaml_cfg["fluid"]["components"] == cfg["fluid"]["components"]
    assert yaml_cfg["grid"] == cfg["grid"]
    assert yaml_cfg["schedule"] == cfg["schedule"]


def test_gem_case_mixture_matches_yaml_case() -> None:
    gem = load_case_mixture(load_case_yaml(GEM_CASE)["fluid"])
    yaml_mix = load_case_mixture(load_case_yaml(DEFAULT_CASE)["fluid"])
    assert gem.names == yaml_mix.names == ("C1", "CO2")
    np.testing.assert_allclose(gem.Tc, yaml_mix.Tc, **_TIGHT)
    np.testing.assert_allclose(gem.Pc, yaml_mix.Pc, **_TIGHT)
    np.testing.assert_allclose(gem.omega, yaml_mix.omega, **_TIGHT)
    np.testing.assert_allclose(gem.kij, yaml_mix.kij, **_TIGHT)


def test_gem_case_run_prints_metrics_and_fields(tmp_path) -> None:
    """Run the gem_deck entry; same HZ 1+4 two-cycle path writes metrics/fields."""
    buf = io.StringIO()
    csv_path = tmp_path / "fields.csv"
    metrics = main([str(GEM_CASE), "--fields", str(csv_path)], stdout=buf)
    text = buf.getvalue()
    assert "EXAMPLE case: hz_1inj4prod_two_cycle_gem" in text
    assert "not a Jiyang GEM card" in text
    assert "cycle 1 inject ||R||" in text
    assert "cycle 2 produce ||R||" in text
    assert "accepted nsteps" in text
    assert metrics["n_cycles"] == 2
    assert metrics["accepted_steps"] >= 2
    assert metrics["underflow"] is False
    assert "inject_R" in metrics["cycles"][0]
    assert "produce_R" in metrics["cycles"][0]
    assert format_metrics(metrics) in text
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "p" in rows[0] and "z_CO2" in rows[0]
    assert len(rows) == 15
    assert metrics["fields_csv"] == str(csv_path)
