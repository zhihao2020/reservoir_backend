"""Oil-water Corey relative permeability and fractional flow models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D


def effective_saturation(sw: float | ArrayLike | Field3D, swi: float, sor: float) -> float | NDArray[np.float64]:
    """Return clipped effective water saturation `Se`."""
    validate_saturation_params(swi, sor)
    sw_values = _values(sw)
    se = (sw_values - float(swi)) / (1.0 - float(swi) - float(sor))
    result = np.clip(se, 0.0, 1.0)
    return _scalar_or_array(result)


def corey_relative_permeability(
    sw: float | ArrayLike | Field3D,
    swi: float,
    sor: float,
    krw0: float,
    kro0: float,
    nw: float,
    no: float,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return Corey water and oil relative permeability `(krw, kro)`."""
    _validate_corey_params(krw0, kro0, nw, no)
    se = np.asarray(effective_saturation(sw, swi, sor), dtype=float)
    krw = float(krw0) * se ** float(nw)
    kro = float(kro0) * (1.0 - se) ** float(no)
    return _scalar_or_array(krw), _scalar_or_array(kro)


def water_mobility(krw: float | ArrayLike, mu_w: float) -> float | NDArray[np.float64]:
    """Return water mobility `lambda_w = krw / mu_w`."""
    validate_viscosity(mu_w, 1.0)
    values = _nonnegative_finite(krw, "krw")
    result = values / float(mu_w)
    return _scalar_or_array(result)


def oil_mobility(kro: float | ArrayLike, mu_o: float) -> float | NDArray[np.float64]:
    """Return oil mobility `lambda_o = kro / mu_o`."""
    validate_viscosity(1.0, mu_o)
    values = _nonnegative_finite(kro, "kro")
    result = values / float(mu_o)
    return _scalar_or_array(result)


def fractional_flow_water(
    sw: float | ArrayLike | Field3D,
    swi: float,
    sor: float,
    krw0: float,
    kro0: float,
    nw: float,
    no: float,
    mu_w: float,
    mu_o: float,
) -> float | NDArray[np.float64]:
    """Return water fractional flow `fw = lambda_w / (lambda_w + lambda_o)`."""
    validate_viscosity(mu_w, mu_o)
    krw, kro = corey_relative_permeability(sw, swi, sor, krw0, kro0, nw, no)
    lambda_w = np.asarray(water_mobility(krw, mu_w), dtype=float)
    lambda_o = np.asarray(oil_mobility(kro, mu_o), dtype=float)
    denominator = lambda_w + lambda_o
    fw = np.divide(
        lambda_w,
        denominator,
        out=np.zeros_like(lambda_w, dtype=float),
        where=denominator > 0.0,
    )
    fw = np.clip(fw, 0.0, 1.0)
    return _scalar_or_array(fw)


def validate_saturation_params(swi: float, sor: float) -> None:
    """Validate residual water and oil saturations."""
    swi_value = float(swi)
    sor_value = float(sor)
    if (
        not np.isfinite(swi_value)
        or not np.isfinite(sor_value)
        or swi_value < 0.0
        or sor_value < 0.0
        or swi_value >= 1.0
        or sor_value >= 1.0
        or swi_value + sor_value >= 1.0
    ):
        raise InvalidPhysicalValueError("swi and sor must be finite, non-negative, and sum to less than 1")


def validate_viscosity(mu_w: float, mu_o: float) -> None:
    """Validate water and oil viscosity values."""
    mu_w_value = float(mu_w)
    mu_o_value = float(mu_o)
    if not np.isfinite(mu_w_value) or mu_w_value <= 0.0:
        raise InvalidPhysicalValueError("mu_w must be a positive finite value")
    if not np.isfinite(mu_o_value) or mu_o_value <= 0.0:
        raise InvalidPhysicalValueError("mu_o must be a positive finite value")


def _validate_corey_params(krw0: float, kro0: float, nw: float, no: float) -> None:
    for name, value in {"krw0": krw0, "kro0": kro0}.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise InvalidPhysicalValueError(f"{name} must be finite and non-negative")
    for name, value in {"nw": nw, "no": no}.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise InvalidPhysicalValueError(f"{name} must be a positive finite value")


def _values(sw: float | ArrayLike | Field3D) -> NDArray[np.float64]:
    values = sw.values if isinstance(sw, Field3D) else sw
    array = np.asarray(values, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError("sw must be finite")
    return array


def _nonnegative_finite(value: float | ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any() or (array < 0.0).any():
        raise InvalidPhysicalValueError(f"{name} must be finite and non-negative")
    return array


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if value.shape == ():
        return float(value)
    return value
