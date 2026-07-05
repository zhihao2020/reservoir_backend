"""Pressure-field diagnostic utilities.

These helpers do not assemble or solve pressure systems. They only inspect
pressure and flux arrays produced by existing solvers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.solver.velocity import FaceFluxes


def compute_pressure_statistics(pressure: Field3D | ArrayLike) -> dict:
    """Return finite pressure summary statistics."""
    values = _values(pressure)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "pressure_min": None,
            "pressure_max": None,
            "pressure_mean": None,
            "pressure_std": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["pressure contains NaN or Inf"],
        }
    return {
        "pressure_min": float(np.min(values)),
        "pressure_max": float(np.max(values)),
        "pressure_mean": float(np.mean(values)),
        "pressure_std": float(np.std(values)),
        "has_nan": False,
        "has_inf": False,
        "warnings": [],
    }


def check_pressure_finite(pressure: Field3D | ArrayLike) -> dict:
    """Report whether pressure contains NaN or Inf."""
    values = _values(pressure)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    return {"finite": not (has_nan or has_inf), "has_nan": has_nan, "has_inf": has_inf}


def compute_pressure_error_metrics(
    pressure: Field3D | ArrayLike,
    reference_pressure: Field3D | ArrayLike,
) -> dict:
    """Return L2/Linf pressure error metrics."""
    values = _values(pressure)
    reference = _values(reference_pressure)
    if values.shape != reference.shape:
        raise ValueError("pressure and reference_pressure must have matching shapes")
    diff = values - reference
    l2 = float(np.linalg.norm(diff.ravel()))
    ref_l2 = float(np.linalg.norm(reference.ravel()))
    rel = l2 / max(ref_l2, 1.0e-30)
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "l2_error": l2,
        "relative_l2_error": float(rel),
        "linf_error": float(np.max(np.abs(diff))),
        "has_nan": bool(np.isnan(diff).any()),
        "has_inf": bool(np.isinf(diff).any()),
    }


def compute_flux_conservation_metrics(
    face_fluxes: FaceFluxes | tuple[ArrayLike, ArrayLike, ArrayLike],
    source_sink: ArrayLike | None = None,
) -> dict:
    """Compute cell-wise finite-volume flux imbalance.

    Positive source_sink denotes injection. The returned imbalance is
    `net_outflow - source_sink`.
    """
    fx, fy, fz = _flux_arrays(face_fluxes)
    if fx.ndim != 3 or fy.ndim != 3 or fz.ndim != 3:
        raise ValueError("flux arrays must be 3D")
    nz, ny, nx_plus = fx.shape
    nx = nx_plus - 1
    if fy.shape != (nz, ny + 1, nx) or fz.shape != (nz + 1, ny, nx):
        raise ValueError("flux shapes are inconsistent")
    divergence = (
        fx[:, :, 1:] - fx[:, :, :-1]
        + fy[:, 1:, :] - fy[:, :-1, :]
        + fz[1:, :, :] - fz[:-1, :, :]
    )
    if source_sink is None:
        source = np.zeros_like(divergence)
    else:
        source = np.asarray(source_sink, dtype=float)
        if source.shape != divergence.shape:
            raise ValueError("source_sink shape must match cell shape")
    imbalance = divergence - source
    return {
        "max_flux_imbalance": float(np.max(np.abs(imbalance))),
        "mean_abs_flux_imbalance": float(np.mean(np.abs(imbalance))),
        "total_flux_imbalance": float(np.sum(imbalance)),
        "has_nan": bool(np.isnan(imbalance).any()),
        "has_inf": bool(np.isinf(imbalance).any()),
    }


def compute_mass_balance_residual(
    inflow: float,
    outflow: float,
    source_sink: float = 0.0,
) -> float:
    """Return signed mass-balance residual for scalar flow totals."""
    return float(float(inflow) + float(source_sink) - float(outflow))


def build_pressure_diagnostics_report(
    pressure: Field3D | ArrayLike,
    reference_pressure: Field3D | ArrayLike | None = None,
    face_fluxes: FaceFluxes | tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    source_sink: ArrayLike | None = None,
    mass_balance_error: float | None = None,
) -> dict:
    """Build a JSON-serializable pressure diagnostics report."""
    stats = compute_pressure_statistics(pressure)
    if reference_pressure is None:
        errors = {
            "max_abs_error": None,
            "l2_error": None,
            "relative_l2_error": None,
            "linf_error": None,
        }
    else:
        errors = compute_pressure_error_metrics(pressure, reference_pressure)

    if face_fluxes is None:
        flux = {"max_flux_imbalance": None}
    else:
        flux = compute_flux_conservation_metrics(face_fluxes, source_sink=source_sink)

    has_nan = bool(stats["has_nan"] or errors.get("has_nan", False) or flux.get("has_nan", False))
    has_inf = bool(stats["has_inf"] or errors.get("has_inf", False) or flux.get("has_inf", False))
    return {
        "success": not (has_nan or has_inf),
        "pressure_min": stats["pressure_min"],
        "pressure_max": stats["pressure_max"],
        "pressure_mean": stats["pressure_mean"],
        "pressure_std": stats["pressure_std"],
        "max_abs_error": errors["max_abs_error"],
        "l2_error": errors["l2_error"],
        "relative_l2_error": errors["relative_l2_error"],
        "linf_error": errors["linf_error"],
        "max_flux_imbalance": flux["max_flux_imbalance"],
        "mass_balance_error": None if mass_balance_error is None else float(mass_balance_error),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": list(stats.get("warnings", [])),
    }


def _values(value: Field3D | ArrayLike) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        return np.asarray(value.values, dtype=float)
    return np.asarray(value, dtype=float)


def _flux_arrays(face_fluxes: FaceFluxes | tuple[ArrayLike, ArrayLike, ArrayLike]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(face_fluxes, FaceFluxes):
        return (
            np.asarray(face_fluxes.flux_x, dtype=float),
            np.asarray(face_fluxes.flux_y, dtype=float),
            np.asarray(face_fluxes.flux_z, dtype=float),
        )
    fx, fy, fz = face_fluxes
    return np.asarray(fx, dtype=float), np.asarray(fy, dtype=float), np.asarray(fz, dtype=float)
