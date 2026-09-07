from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import (
    ContrastParameterization,
    RegionParameterization,
)
from reservoir_backend.io.case import load_case
from reservoir_backend.io.parameterization_cfg import parameterization_from_cfg


def _grid() -> CartesianGrid:
    return CartesianGrid.uniform((0.4, 0.2, 0.3), (0.1, 0.1, 0.1))


def test_unknown_parameterization_kind_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown inverse.parameterization"):
        parameterization_from_cfg(_grid(), {"inverse": {"parameterization": "mystery"}}, tmp_path)


def test_contrast_is_selectable(tmp_path: Path) -> None:
    param = parameterization_from_cfg(
        _grid(),
        {"inverse": {"parameterization": "contrast", "region_axis": "z", "n_regions": 2}},
        tmp_path,
    )
    assert isinstance(param, ContrastParameterization)
    assert param.n_params == 2


def test_coarse_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown inverse.parameterization"):
        parameterization_from_cfg(
            _grid(), {"inverse": {"parameterization": "coarse_field", "coarse_n": [2, 1, 3]}}, tmp_path
        )


def test_region_axis_x_differs_from_z(tmp_path: Path) -> None:
    px = parameterization_from_cfg(
        _grid(), {"inverse": {"parameterization": "region", "region_axis": "x", "n_regions": 2}}, tmp_path
    )
    pz = parameterization_from_cfg(
        _grid(), {"inverse": {"parameterization": "region", "region_axis": "z", "n_regions": 2}}, tmp_path
    )
    assert isinstance(px, RegionParameterization)
    assert isinstance(pz, RegionParameterization)
    assert not np.array_equal(px.region_id, pz.region_id)


def test_region_map_npy(tmp_path: Path) -> None:
    rid = np.arange(_grid().n_cells, dtype=np.int64) % 3
    np.save(tmp_path / "regions.npy", rid)
    param = parameterization_from_cfg(
        _grid(), {"inverse": {"parameterization": "region", "region_map": "regions.npy"}}, tmp_path
    )
    assert np.array_equal(param.region_id, rid)


def test_lab_v1_has_two_cf_tmf_parameters() -> None:
    twin = load_case("examples/lab_v1/case.yaml")
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "LogCfTmfParameterization"


def test_high_region_flips_contrast_body(tmp_path: Path) -> None:
    top = parameterization_from_cfg(
        _grid(),
        {"inverse": {"parameterization": "contrast", "region_axis": "z", "n_regions": 2, "high_region": 1}},
        tmp_path,
    )
    bot = parameterization_from_cfg(
        _grid(),
        {"inverse": {"parameterization": "contrast", "region_axis": "z", "n_regions": 2, "high_region": 0}},
        tmp_path,
    )
    assert not np.array_equal(top.region_id, bot.region_id)
    assert int(np.max(top.region_id)) == 1


def test_lab_cf_fixture_is_joint_cf_tmf() -> None:
    twin = load_case("examples/lab/lab_cf.yaml")
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "LogCfTmfParameterization"
    assert twin.uses_dpdp()


def test_forbidden_inverse_keys_error() -> None:
    from reservoir_backend.io.case import inverse_spec_from_cfg

    with pytest.raises(ValueError, match="inverse keys not accepted"):
        inverse_spec_from_cfg({"n_workers": 8, "parameterization": "region"})


def test_ensemble_keys_are_accepted() -> None:
    from reservoir_backend.io.case import inverse_spec_from_cfg

    spec = inverse_spec_from_cfg(
        {"parameterization": "region", "algorithm": "esmda", "ensemble_size": 10, "assimilation_steps": 3, "seed": 2}
    )
    assert spec.algorithm == "esmda"
    assert spec.ensemble_size == 10
    assert spec.assimilation_steps == 3
    assert spec.seed == 2

