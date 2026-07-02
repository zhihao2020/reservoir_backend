"""Utilities for composing advective, capillary, and gravity water fluxes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, InvalidPhysicalValueError


def compose_water_fluxes_3d(
    adv_flux_x: ArrayLike,
    adv_flux_y: ArrayLike,
    adv_flux_z: ArrayLike,
    cap_flux_x: ArrayLike | None = None,
    cap_flux_y: ArrayLike | None = None,
    cap_flux_z: ArrayLike | None = None,
    grav_flux_x: ArrayLike | None = None,
    grav_flux_y: ArrayLike | None = None,
    grav_flux_z: ArrayLike | None = None,
    include_capillary: bool = False,
    include_gravity: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Compose total water face fluxes from enabled flux components."""
    components = validate_flux_shapes_3d(
        adv_flux_x,
        adv_flux_y,
        adv_flux_z,
        cap_flux_x,
        cap_flux_y,
        cap_flux_z,
        grav_flux_x,
        grav_flux_y,
        grav_flux_z,
        include_capillary=include_capillary,
        include_gravity=include_gravity,
    )
    adv_x, adv_y, adv_z = components["adv"]
    cap_x, cap_y, cap_z = components["cap"]
    grav_x, grav_y, grav_z = components["grav"]
    water_x = adv_x + cap_x + grav_x
    water_y = adv_y + cap_y + grav_y
    water_z = adv_z + cap_z + grav_z
    eff_x, eff_y, eff_z = compute_effective_flux_for_cfl(
        adv_x,
        adv_y,
        adv_z,
        cap_x,
        cap_y,
        cap_z,
        grav_x,
        grav_y,
        grav_z,
        include_capillary=include_capillary,
        include_gravity=include_gravity,
    )
    report = build_combined_flux_report(
        adv_x,
        adv_y,
        adv_z,
        water_x,
        water_y,
        water_z,
        eff_x,
        eff_y,
        eff_z,
        cap_x,
        cap_y,
        cap_z,
        grav_x,
        grav_y,
        grav_z,
        include_capillary=include_capillary,
        include_gravity=include_gravity,
    )
    return water_x, water_y, water_z, report


def compute_effective_flux_for_cfl(
    adv_flux_x: ArrayLike,
    adv_flux_y: ArrayLike,
    adv_flux_z: ArrayLike,
    cap_flux_x: ArrayLike | None = None,
    cap_flux_y: ArrayLike | None = None,
    cap_flux_z: ArrayLike | None = None,
    grav_flux_x: ArrayLike | None = None,
    grav_flux_y: ArrayLike | None = None,
    grav_flux_z: ArrayLike | None = None,
    include_capillary: bool = False,
    include_gravity: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return conservative effective fluxes for explicit CFL checks."""
    components = validate_flux_shapes_3d(
        adv_flux_x,
        adv_flux_y,
        adv_flux_z,
        cap_flux_x,
        cap_flux_y,
        cap_flux_z,
        grav_flux_x,
        grav_flux_y,
        grav_flux_z,
        include_capillary=include_capillary,
        include_gravity=include_gravity,
    )
    adv_x, adv_y, adv_z = components["adv"]
    cap_x, cap_y, cap_z = components["cap"]
    grav_x, grav_y, grav_z = components["grav"]
    return (
        np.abs(adv_x) + np.abs(cap_x) + np.abs(grav_x),
        np.abs(adv_y) + np.abs(cap_y) + np.abs(grav_y),
        np.abs(adv_z) + np.abs(cap_z) + np.abs(grav_z),
    )


def build_combined_flux_report(
    adv_flux_x: ArrayLike,
    adv_flux_y: ArrayLike,
    adv_flux_z: ArrayLike,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    effective_flux_x: ArrayLike,
    effective_flux_y: ArrayLike,
    effective_flux_z: ArrayLike,
    cap_flux_x: ArrayLike | None = None,
    cap_flux_y: ArrayLike | None = None,
    cap_flux_z: ArrayLike | None = None,
    grav_flux_x: ArrayLike | None = None,
    grav_flux_y: ArrayLike | None = None,
    grav_flux_z: ArrayLike | None = None,
    include_capillary: bool = False,
    include_gravity: bool = False,
) -> dict[str, object]:
    """Build a report for composed water fluxes."""
    adv = _max_abs_arrays(adv_flux_x, adv_flux_y, adv_flux_z)
    cap = _max_abs_arrays(cap_flux_x, cap_flux_y, cap_flux_z) if include_capillary else 0.0
    grav = _max_abs_arrays(grav_flux_x, grav_flux_y, grav_flux_z) if include_gravity else 0.0
    total = _max_abs_arrays(water_flux_x, water_flux_y, water_flux_z)
    effective = _max_abs_arrays(effective_flux_x, effective_flux_y, effective_flux_z)
    all_arrays = [
        np.asarray(adv_flux_x, dtype=float),
        np.asarray(adv_flux_y, dtype=float),
        np.asarray(adv_flux_z, dtype=float),
        np.asarray(water_flux_x, dtype=float),
        np.asarray(water_flux_y, dtype=float),
        np.asarray(water_flux_z, dtype=float),
        np.asarray(effective_flux_x, dtype=float),
        np.asarray(effective_flux_y, dtype=float),
        np.asarray(effective_flux_z, dtype=float),
    ]
    return {
        "include_capillary": bool(include_capillary),
        "include_gravity": bool(include_gravity),
        "max_advective_flux": float(adv),
        "max_capillary_flux": float(cap),
        "max_gravity_flux": float(grav),
        "max_total_water_flux": float(total),
        "max_effective_flux": float(effective),
        "has_nan": bool(any(np.isnan(array).any() for array in all_arrays)),
        "has_inf": bool(any(np.isinf(array).any() for array in all_arrays)),
        "flux_shape_x": tuple(np.asarray(water_flux_x, dtype=float).shape),
        "flux_shape_y": tuple(np.asarray(water_flux_y, dtype=float).shape),
        "flux_shape_z": tuple(np.asarray(water_flux_z, dtype=float).shape),
    }


def validate_flux_shapes_3d(
    adv_flux_x: ArrayLike,
    adv_flux_y: ArrayLike,
    adv_flux_z: ArrayLike,
    cap_flux_x: ArrayLike | None = None,
    cap_flux_y: ArrayLike | None = None,
    cap_flux_z: ArrayLike | None = None,
    grav_flux_x: ArrayLike | None = None,
    grav_flux_y: ArrayLike | None = None,
    grav_flux_z: ArrayLike | None = None,
    include_capillary: bool = False,
    include_gravity: bool = False,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """Validate flux component shapes and return normalized arrays."""
    adv = (
        _finite_array(adv_flux_x, "adv_flux_x"),
        _finite_array(adv_flux_y, "adv_flux_y"),
        _finite_array(adv_flux_z, "adv_flux_z"),
    )
    if include_capillary:
        _require_all_present((cap_flux_x, cap_flux_y, cap_flux_z), "capillary")
        cap = (
            _finite_array(cap_flux_x, "cap_flux_x"),
            _finite_array(cap_flux_y, "cap_flux_y"),
            _finite_array(cap_flux_z, "cap_flux_z"),
        )
    else:
        cap = tuple(np.zeros_like(array, dtype=float) for array in adv)
    if include_gravity:
        _require_all_present((grav_flux_x, grav_flux_y, grav_flux_z), "gravity")
        grav = (
            _finite_array(grav_flux_x, "grav_flux_x"),
            _finite_array(grav_flux_y, "grav_flux_y"),
            _finite_array(grav_flux_z, "grav_flux_z"),
        )
    else:
        grav = tuple(np.zeros_like(array, dtype=float) for array in adv)

    for label, arrays in {"capillary": cap, "gravity": grav}.items():
        for direction, base, optional in zip(("x", "y", "z"), adv, arrays, strict=True):
            if optional.shape != base.shape:
                raise FieldShapeError(
                    f"{label} flux_{direction} shape {optional.shape} does not match advective shape {base.shape}"
                )
    return {"adv": adv, "cap": cap, "grav": grav}


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError(f"{name} must be finite")
    return array.copy()


def _require_all_present(values: tuple[ArrayLike | None, ArrayLike | None, ArrayLike | None], label: str) -> None:
    if any(value is None for value in values):
        raise ValueError(f"{label} fluxes are required when include_{label}=True")


def _max_abs_arrays(*arrays: ArrayLike | None) -> float:
    finite_arrays = [np.asarray(array, dtype=float) for array in arrays if array is not None]
    if not finite_arrays:
        return 0.0
    return float(max(np.max(np.abs(array)) for array in finite_arrays))
