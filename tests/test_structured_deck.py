"""Tests for the project-local structured deck loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.io.structured_deck import load_structured_deck

REPO_ROOT = Path(__file__).resolve().parents[1]
SPE1 = REPO_ROOT / "references" / "upstream" / "opm-tests" / "spe1" / "SPE1CASE1.DATA"
WATER = REPO_ROOT / "references" / "upstream" / "opm-tests" / "water-1ph" / "WATER2F.DATA"


@pytest.mark.skipif(not SPE1.is_file(), reason="opm-tests submodule not initialized")
def test_load_spe1_structured_geometry_and_properties() -> None:
    bundle = load_structured_deck(SPE1)
    assert bundle.grid.nx == 10
    assert bundle.grid.ny == 10
    assert bundle.grid.nz == 3
    assert np.allclose(bundle.grid.spacing_i, 1000.0)
    assert np.allclose(bundle.grid.spacing_j, 1000.0)
    assert np.allclose(bundle.grid.spacing_k, [20.0, 30.0, 50.0])
    assert bundle.porosity_field is not None
    assert np.allclose(bundle.porosity_field, 0.3)
    assert bundle.permeability_x_md is not None
    assert np.allclose(bundle.permeability_x_md[0], 500.0)
    assert np.allclose(bundle.permeability_x_md[1], 50.0)
    assert np.allclose(bundle.permeability_x_md[2], 200.0)


@pytest.mark.skipif(not SPE1.is_file(), reason="opm-tests submodule not initialized")
def test_length_unit_conversion_optional() -> None:
    native = load_structured_deck(SPE1, convert_length_ft_to_m=False)
    metric = load_structured_deck(SPE1, convert_length_ft_to_m=True)
    assert np.allclose(metric.grid.spacing_i, native.grid.spacing_i * 0.3048)


@pytest.mark.skipif(not WATER.is_file(), reason="opm-tests submodule not initialized")
def test_load_water_case_dimensions() -> None:
    bundle = load_structured_deck(WATER)
    assert bundle.grid.nx >= 1
    assert bundle.grid.ny >= 1
    assert bundle.grid.nz >= 1


def test_missing_deck_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.DATA"
    with pytest.raises(FileNotFoundError, match="submodule"):
        load_structured_deck(missing)
