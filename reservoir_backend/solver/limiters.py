"""Slope limiters for optional high-resolution saturation transport."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def minmod(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the minmod-limited slope for two differences."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    same_sign = left * right > 0.0
    limited = np.where(same_sign, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
    return np.asarray(limited, dtype=float)


def vanleer_limiter(r: ArrayLike) -> NDArray[np.float64]:
    """Van Leer limiter phi(r) = (r + |r|) / (1 + |r|)."""
    ratio = np.asarray(r, dtype=float)
    return np.where(np.isfinite(ratio), (ratio + np.abs(ratio)) / (1.0 + np.abs(ratio)), 0.0)


def superbee_limiter(r: ArrayLike) -> NDArray[np.float64]:
    """Superbee limiter."""
    ratio = np.asarray(r, dtype=float)
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * ratio, 1.0), np.minimum(ratio, 2.0)))


def compute_limited_slopes(values: ArrayLike, limiter: str = "minmod") -> NDArray[np.float64]:
    """Compute cell-centered limited slopes for a 1D array."""
    line = np.asarray(values, dtype=float)
    if line.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(line).all():
        raise ValueError("values must be finite")
    slopes = np.zeros_like(line, dtype=float)
    if line.size < 3:
        return slopes
    backward = line[1:-1] - line[:-2]
    forward = line[2:] - line[1:-1]
    name = limiter.lower().replace("_", "-")
    if name == "minmod":
        slopes[1:-1] = minmod(backward, forward)
    elif name in {"vanleer", "van-leer"}:
        ratio = _safe_ratio(backward, forward)
        phi = vanleer_limiter(ratio)
        slopes[1:-1] = phi * forward
    elif name == "superbee":
        ratio = _safe_ratio(backward, forward)
        phi = superbee_limiter(ratio)
        slopes[1:-1] = phi * forward
    else:
        raise ValueError(f"unsupported limiter: {limiter}")
    return slopes


def preserves_monotonicity(values: ArrayLike, reconstructed: ArrayLike, tolerance: float = 1.0e-12) -> bool:
    """Return whether reconstructed values stay within original min/max bounds."""
    original = np.asarray(values, dtype=float)
    recon = np.asarray(reconstructed, dtype=float)
    if not np.isfinite(original).all() or not np.isfinite(recon).all():
        return False
    return bool(np.min(recon) >= np.min(original) - tolerance and np.max(recon) <= np.max(original) + tolerance)


def _safe_ratio(numerator: NDArray[np.float64], denominator: NDArray[np.float64]) -> NDArray[np.float64]:
    ratio = np.zeros_like(numerator, dtype=float)
    mask = np.abs(denominator) > 1.0e-30
    ratio[mask] = numerator[mask] / denominator[mask]
    return ratio
