"""EXAMPLE EOS YAML deck: load, marker, missing file/Tc. Not FIM, not GEM."""

import ast
from pathlib import Path

import pytest
import yaml

from reservoir_backend.eos import (
    DEFAULT_EXAMPLE_FLUID_YAML,
    EXAMPLE_LIBRARY_MARKER,
    example_eight_component_mixture,
    example_feed_z,
    load_eos_mixture_yaml,
    load_feed_z_yaml,
    resolve_fluid_yaml,
)


def test_example_yaml_loads_and_keeps_marker() -> None:
    mix = load_eos_mixture_yaml(DEFAULT_EXAMPLE_FLUID_YAML)
    assert DEFAULT_EXAMPLE_FLUID_YAML.is_file()
    assert "EXAMPLE" in DEFAULT_EXAMPLE_FLUID_YAML.read_text(encoding="utf-8")
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert "NOT a Jiyang GEM card" in mix.marker
    assert mix.names == ("C1", "C2", "C3", "nC4", "nC5", "nC6", "example_C7plus", "CO2")
    assert float(mix.Tc[mix.names.index("C1")]) == 190.564
    adapter = example_eight_component_mixture()
    assert adapter.marker == mix.marker
    assert adapter.names == mix.names
    z = example_feed_z()
    assert z.shape == (8,)
    assert abs(float(z.sum()) - 1.0) < 1e-12


def test_missing_eos_yaml_file_errors() -> None:
    missing = Path("/tmp/does_not_exist_example_eos.yaml")
    with pytest.raises(FileNotFoundError, match="refusing to invent GEM/Jiyang criticals"):
        load_eos_mixture_yaml(missing)
    with pytest.raises(FileNotFoundError, match="refusing to invent GEM/Jiyang criticals"):
        resolve_fluid_yaml(missing)


def test_missing_tc_errors(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_EXAMPLE_FLUID_YAML.read_text(encoding="utf-8"))
    del raw["components"][0]["Tc"]
    bad = tmp_path / "no_tc.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field Tc"):
        load_eos_mixture_yaml(bad)
    with pytest.raises(ValueError, match="Refusing to invent criticals"):
        load_eos_mixture_yaml(bad)


def test_missing_feed_z_errors(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_EXAMPLE_FLUID_YAML.read_text(encoding="utf-8"))
    del raw["feed_z"]
    bad = tmp_path / "no_feed.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field feed_z"):
        load_feed_z_yaml(bad)


def test_eos_loader_does_not_import_fi_or_references() -> None:
    root = Path(__file__).resolve().parents[1] / "reservoir_backend" / "eos"
    for rel in ("load.py", "example_library.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not any(n.startswith("references") or n == "references" for n in names)
        assert not any(
            "solver.fi" in n or n == "reservoir_backend.solver" or n.startswith("reservoir_backend.solver.")
            for n in names
        )
