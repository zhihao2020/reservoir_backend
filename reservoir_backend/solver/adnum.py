"""Local dense automatic differentiation for FIM residuals (product name).

Each quantity carries value + derivatives w.r.t. the cell's three primary
unknowns (p, Sw, x). Face fluxes combine neighbour AD numbers to fill
off-diagonal Jacobian blocks. Pure numpy — no JAX/CasADi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CellAD:
    """Vectorized dual number: value (n,) and derivs (n, n_prim)."""

    __slots__ = ("value", "derivs")

    def __init__(self, value: NDArray[np.float64] | float, derivs: NDArray[np.float64] | None = None, *, n_prim: int = 3):
        v = np.asarray(value, dtype=float).ravel()
        self.value = v
        if derivs is None:
            self.derivs = np.zeros((v.size, int(n_prim)), dtype=float)
        else:
            d = np.asarray(derivs, dtype=float)
            if d.ndim == 1:
                d = d.reshape(-1, 1)
            if d.shape[0] != v.size:
                raise ValueError(f"derivs rows {d.shape[0]} != value size {v.size}")
            self.derivs = d

    @property
    def n(self) -> int:
        return int(self.value.size)

    @property
    def n_prim(self) -> int:
        return int(self.derivs.shape[1])

    @classmethod
    def independent(cls, value: NDArray[np.float64] | float, slot: int, *, n_prim: int = 3) -> CellAD:
        v = np.asarray(value, dtype=float).ravel()
        d = np.zeros((v.size, int(n_prim)), dtype=float)
        d[:, int(slot)] = 1.0
        return cls(v, d)

    @classmethod
    def constant(cls, value: NDArray[np.float64] | float, *, n: int | None = None, n_prim: int = 3) -> CellAD:
        v = np.asarray(value, dtype=float).ravel()
        if n is not None and v.size == 1:
            v = np.full(int(n), float(v[0]), dtype=float)
        return cls(v, np.zeros((v.size, int(n_prim)), dtype=float))

    def copy(self) -> CellAD:
        return CellAD(self.value.copy(), self.derivs.copy())

    def _bin(self, other, op_v, op_d):
        if isinstance(other, CellAD):
            return CellAD(op_v(self.value, other.value), op_d(self, other))
        o = float(other)
        return CellAD(op_v(self.value, o), op_d(self, o))

    def __add__(self, other):
        if isinstance(other, CellAD):
            return CellAD(self.value + other.value, self.derivs + other.derivs)
        return CellAD(self.value + float(other), self.derivs.copy())

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CellAD):
            return CellAD(self.value - other.value, self.derivs - other.derivs)
        return CellAD(self.value - float(other), self.derivs.copy())

    def __rsub__(self, other):
        if isinstance(other, CellAD):
            return CellAD(other.value - self.value, other.derivs - self.derivs)
        return CellAD(float(other) - self.value, -self.derivs)

    def __mul__(self, other):
        if isinstance(other, CellAD):
            return CellAD(
                self.value * other.value,
                self.derivs * other.value[:, None] + other.derivs * self.value[:, None],
            )
        o = float(other)
        return CellAD(self.value * o, self.derivs * o)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, CellAD):
            den = np.maximum(np.abs(other.value), 1.0e-30) * np.sign(other.value + 1.0e-30)
            # safer: use other.value directly with clip
            ov = other.value
            inv = 1.0 / np.where(np.abs(ov) < 1.0e-30, np.sign(ov) * 1.0e-30 + (ov == 0) * 1.0e-30, ov)
            val = self.value * inv
            d = self.derivs * inv[:, None] - other.derivs * (self.value * inv * inv)[:, None]
            return CellAD(val, d)
        o = float(other)
        inv = 1.0 / (o if abs(o) > 1.0e-30 else (1.0e-30 if o >= 0 else -1.0e-30))
        return CellAD(self.value * inv, self.derivs * inv)

    def __rtruediv__(self, other):
        o = float(other)
        inv = 1.0 / np.where(np.abs(self.value) < 1.0e-30, np.sign(self.value) * 1.0e-30 + (self.value == 0) * 1.0e-30, self.value)
        val = o * inv
        d = -self.derivs * (o * inv * inv)[:, None]
        return CellAD(val, d)

    def __neg__(self):
        return CellAD(-self.value, -self.derivs)

    def __pow__(self, power):
        p = float(power)
        val = np.power(self.value, p)
        # d(x^p)/dx = p x^{p-1}
        base = np.where(np.abs(self.value) < 1.0e-30, 1.0e-30, self.value)
        dval = p * np.power(base, p - 1.0)
        return CellAD(val, self.derivs * dval[:, None])


def where(cond: NDArray[np.bool_], a: CellAD | float, b: CellAD | float) -> CellAD:
    c = np.asarray(cond, dtype=bool).ravel()
    if isinstance(a, CellAD):
        av, ad = a.value, a.derivs
        n_prim = a.derivs.shape[1]
        n = a.n
    else:
        n = int(c.size)
        n_prim = b.derivs.shape[1] if isinstance(b, CellAD) else 3
        av = np.full(n, float(a), dtype=float)
        ad = np.zeros((n, n_prim), dtype=float)
    if isinstance(b, CellAD):
        bv, bd = b.value, b.derivs
    else:
        bv = np.full(n, float(b), dtype=float)
        bd = np.zeros((n, n_prim), dtype=float)
    val = np.where(c, av, bv)
    d = np.where(c[:, None], ad, bd)
    return CellAD(val, d)


def maximum(a: CellAD | float, b: CellAD | float) -> CellAD:
    if isinstance(a, CellAD):
        av = a.value
        other = b.value if isinstance(b, CellAD) else float(b)
        return where(av >= other, a, b)
    return where(float(a) >= (b.value if isinstance(b, CellAD) else float(b)), a, b)


def minimum(a: CellAD | float, b: CellAD | float) -> CellAD:
    if isinstance(a, CellAD):
        av = a.value
        other = b.value if isinstance(b, CellAD) else float(b)
        return where(av <= other, a, b)
    return where(float(a) <= (b.value if isinstance(b, CellAD) else float(b)), a, b)


def clip(a: CellAD, lo: float, hi: float) -> CellAD:
    return minimum(maximum(a, lo), hi)


def interp_with_slope(x: NDArray[np.float64], xp: NDArray[np.float64], fp: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Piecewise-linear interpolate and return segment slope df/dx at each x."""
    xp = np.asarray(xp, dtype=float).ravel()
    fp = np.asarray(fp, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    val = np.interp(x, xp, fp)
    # slope on segments; extend endpoints with edge slope
    dx = np.diff(xp)
    df = np.diff(fp)
    slope_seg = np.divide(df, dx, out=np.zeros_like(df), where=np.abs(dx) > 1.0e-30)
    # index of right segment for each x (clip)
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, slope_seg.size - 1)
    slope = slope_seg[idx]
    # outside range: use edge slopes
    slope = np.where(x <= xp[0], slope_seg[0], slope)
    slope = np.where(x >= xp[-1], slope_seg[-1], slope)
    return val, slope


def table_eval(x_ad: CellAD, xp: NDArray[np.float64], fp: NDArray[np.float64]) -> CellAD:
    """Evaluate tabulated f(x) with AD through x."""
    val, slope = interp_with_slope(x_ad.value, xp, fp)
    return CellAD(val, x_ad.derivs * slope[:, None])
