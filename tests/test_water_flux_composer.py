from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.core.exceptions import FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.io.config_loader import load_case_config
from reservoir_backend.solver.water_flux_composer import (
    compose_water_fluxes_3d,
    compute_effective_flux_for_cfl,
)


def test_compose_advective_only_returns_advective() -> None:
    adv = _adv()
    water = compose_water_fluxes_3d(*adv)[:3]
    for result, expected in zip(water, adv, strict=True):
        assert np.allclose(result, expected)


def test_compose_with_capillary() -> None:
    adv, cap = _adv(), _cap()
    water = compose_water_fluxes_3d(*adv, *cap, include_capillary=True)[:3]
    for result, a, c in zip(water, adv, cap, strict=True):
        assert np.allclose(result, a + c)


def test_compose_with_gravity() -> None:
    adv, grav = _adv(), _grav()
    water = compose_water_fluxes_3d(*adv, grav_flux_x=grav[0], grav_flux_y=grav[1], grav_flux_z=grav[2], include_gravity=True)[:3]
    for result, a, g in zip(water, adv, grav, strict=True):
        assert np.allclose(result, a + g)


def test_compose_with_capillary_and_gravity() -> None:
    adv, cap, grav = _adv(), _cap(), _grav()
    water = compose_water_fluxes_3d(
        *adv,
        *cap,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_capillary=True,
        include_gravity=True,
    )[:3]
    for result, a, c, g in zip(water, adv, cap, grav, strict=True):
        assert np.allclose(result, a + c + g)


def test_effective_flux_advective_only() -> None:
    adv = _signed_adv()
    eff = compute_effective_flux_for_cfl(*adv)
    for result, a in zip(eff, adv, strict=True):
        assert np.allclose(result, np.abs(a))


def test_effective_flux_with_capillary() -> None:
    adv, cap = _signed_adv(), _cap()
    eff = compute_effective_flux_for_cfl(*adv, *cap, include_capillary=True)
    for result, a, c in zip(eff, adv, cap, strict=True):
        assert np.allclose(result, np.abs(a) + np.abs(c))


def test_effective_flux_with_gravity() -> None:
    adv, grav = _signed_adv(), _grav()
    eff = compute_effective_flux_for_cfl(
        *adv,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_gravity=True,
    )
    for result, a, g in zip(eff, adv, grav, strict=True):
        assert np.allclose(result, np.abs(a) + np.abs(g))


def test_effective_flux_with_capillary_and_gravity() -> None:
    adv, cap, grav = _signed_adv(), _cap(), _grav()
    eff = compute_effective_flux_for_cfl(
        *adv,
        *cap,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_capillary=True,
        include_gravity=True,
    )
    for result, a, c, g in zip(eff, adv, cap, grav, strict=True):
        assert np.allclose(result, np.abs(a) + np.abs(c) + np.abs(g))


def test_report_keys() -> None:
    *_, report = compose_water_fluxes_3d(*_adv(), *_cap(), include_capillary=True)
    keys = {
        "include_capillary",
        "include_gravity",
        "max_advective_flux",
        "max_capillary_flux",
        "max_gravity_flux",
        "max_total_water_flux",
        "max_effective_flux",
        "has_nan",
        "has_inf",
        "flux_shape_x",
        "flux_shape_y",
        "flux_shape_z",
    }
    assert keys.issubset(report)


def test_shape_mismatch_raises() -> None:
    adv, cap = _adv(), _cap()
    bad_cap_x = np.zeros((1, 1, 1))
    with pytest.raises(FieldShapeError):
        compose_water_fluxes_3d(adv[0], adv[1], adv[2], bad_cap_x, cap[1], cap[2], include_capillary=True)


def test_missing_capillary_flux_when_enabled_raises() -> None:
    adv = _adv()
    with pytest.raises(ValueError, match="capillary fluxes are required"):
        compose_water_fluxes_3d(*adv, include_capillary=True)


def test_missing_gravity_flux_when_enabled_raises() -> None:
    adv = _adv()
    with pytest.raises(ValueError, match="gravity fluxes are required"):
        compose_water_fluxes_3d(*adv, include_gravity=True)


def test_nan_flux_raises() -> None:
    adv = list(_adv())
    adv[0] = adv[0].copy()
    adv[0][0, 0, 0] = np.nan
    with pytest.raises(InvalidPhysicalValueError):
        compose_water_fluxes_3d(*adv)


def test_inf_flux_raises() -> None:
    adv = list(_adv())
    adv[1] = adv[1].copy()
    adv[1][0, 0, 0] = np.inf
    with pytest.raises(InvalidPhysicalValueError):
        compose_water_fluxes_3d(*adv)


def test_zero_optional_flux() -> None:
    adv = _adv()
    zeros = tuple(np.zeros_like(array) for array in adv)
    water = compose_water_fluxes_3d(*adv, *zeros, include_capillary=True)[:3]
    for result, expected in zip(water, adv, strict=True):
        assert np.allclose(result, expected)


def test_flux_shapes_preserved() -> None:
    adv = _adv()
    water = compose_water_fluxes_3d(*adv)[:3]
    for result, expected in zip(water, adv, strict=True):
        assert result.shape == expected.shape


def test_composer_does_not_modify_inputs() -> None:
    adv, cap, grav = _adv(), _cap(), _grav()
    originals = tuple(array.copy() for array in (*adv, *cap, *grav))
    compose_water_fluxes_3d(
        *adv,
        *cap,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_capillary=True,
        include_gravity=True,
    )
    for array, original in zip((*adv, *cap, *grav), originals, strict=True):
        assert np.allclose(array, original)


def test_combined_design_doc_mentions_composer() -> None:
    text = Path("specs/11_combined_capillary_gravity_design.md").read_text(encoding="utf-8")
    assert "029_combined_flux_composer" in text
    assert "water flux composer" in text
    assert "conservative effective flux for CFL" in text


def test_existing_capillary_gravity_together_accepted_when_flags_consistent() -> None:
    config = load_case_config("config/combined_case.yaml")
    assert config["capillary_pressure"]["enabled"] is True
    assert config["gravity"]["enabled"] is True


def test_inconsistent_capillary_gravity_flags_still_raise(tmp_path: Path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_capillary"] = False
    path = tmp_path / "combined_inconsistent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="capillary_pressure.enabled=true requires saturation.use_capillary=true"):
        load_case_config(path)


def test_repeatability() -> None:
    adv, cap, grav = _signed_adv(), _cap(), _grav()
    first = compose_water_fluxes_3d(
        *adv,
        *cap,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_capillary=True,
        include_gravity=True,
    )
    second = compose_water_fluxes_3d(
        *adv,
        *cap,
        grav_flux_x=grav[0],
        grav_flux_y=grav[1],
        grav_flux_z=grav[2],
        include_capillary=True,
        include_gravity=True,
    )
    for a, b in zip(first[:3], second[:3], strict=True):
        assert np.allclose(a, b)
    assert first[3] == second[3]


def _adv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((2, 3, 5), 1.0),
        np.full((2, 4, 4), 2.0),
        np.full((3, 3, 4), 3.0),
    )


def _signed_adv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = _adv()
    return -x, y, -z


def _cap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((2, 3, 5), 0.1),
        np.full((2, 4, 4), -0.2),
        np.full((3, 3, 4), 0.3),
    )


def _grav() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((2, 3, 5)),
        np.zeros((2, 4, 4)),
        np.full((3, 3, 4), -0.4),
    )
