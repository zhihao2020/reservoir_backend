"""Flash backends. Reference PR is truth; fast/tabulated swap without touching DPDP."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.flash import FlashResult, flash_tp
from reservoir_backend.eos.flash_batch import FlashArrays, flash_batch, flash_tp_fast
from reservoir_backend.eos.pr import PengRobinson


class FlashBackend(Protocol):
    name: str

    def flash_tp(
        self,
        eos: PengRobinson,
        pressure: float,
        temperature: float,
        z: NDArray[np.float64],
        **kwargs,
    ) -> FlashResult: ...

    def evaluate_batch(
        self,
        eos: PengRobinson,
        pressure: NDArray[np.float64],
        temperature: float,
        z: NDArray[np.float64],
        **kwargs,
    ) -> FlashArrays: ...


class ReferencePRBackend:
    """Current scalar ``flash_tp``; used as the regression truth."""

    name = "reference"

    def flash_tp(self, eos: PengRobinson, pressure: float, temperature: float, z, **kwargs) -> FlashResult:
        return flash_tp(eos, pressure, temperature, z, **kwargs)

    def evaluate_batch(
        self,
        eos: PengRobinson,
        pressure: NDArray[np.float64],
        temperature: float,
        z: NDArray[np.float64],
        **kwargs,
    ) -> FlashArrays:
        p = np.asarray(pressure, dtype=float).ravel()
        zz = np.asarray(z, dtype=float)
        if zz.ndim == 1:
            zz = np.broadcast_to(zz, (p.size, zz.size)).copy()
        rows = [flash_tp(eos, float(p[i]), float(temperature), zz[i], **kwargs) for i in range(p.size)]
        nc = zz.shape[1]
        arr = FlashArrays(
            vapor_frac=np.array([r.vapor_frac for r in rows]),
            x=np.stack([r.x for r in rows]),
            y=np.stack([r.y for r in rows]),
            z_liq=np.array([r.z_liq for r in rows]),
            z_vap=np.array([r.z_vap for r in rows]),
            v_liq=np.array([r.v_liq for r in rows]),
            v_vap=np.array([r.v_vap for r in rows]),
            two_phase=np.array([r.two_phase for r in rows], dtype=bool),
            k=np.stack([r.k if r.k is not None else np.ones(nc) for r in rows]),
            converged=np.array([r.converged for r in rows], dtype=bool),
            iterations=np.array([r.iterations for r in rows], dtype=np.int32),
            fugacity_error=np.array([r.fugacity_error for r in rows]),
            stability_checked=np.array([r.stability_checked for r in rows], dtype=bool),
            stability_margin=np.array([r.stability_margin for r in rows]),
            fallback_used=np.array([r.fallback_used for r in rows], dtype=bool),
        )
        return arr


class FastPRBackend:
    """Vectorized isothermal PR. Same equilibrium as reference within rtol 1e-8."""

    name = "fast"

    def flash_tp(self, eos: PengRobinson, pressure: float, temperature: float, z, **kwargs) -> FlashResult:
        return flash_tp_fast(eos, pressure, temperature, z, **kwargs)

    def evaluate_batch(
        self,
        eos: PengRobinson,
        pressure: NDArray[np.float64],
        temperature: float,
        z: NDArray[np.float64],
        **kwargs,
    ) -> FlashArrays:
        return flash_batch(eos, pressure, temperature, z, **kwargs)


class TabulatedPRBackend:
    """(p, z_C1) table for isothermal binary V1. Falls back to FastPR near the envelope."""

    name = "tabulated"

    def __init__(self, table=None):
        self._table = table

    def flash_tp(self, eos: PengRobinson, pressure: float, temperature: float, z, **kwargs) -> FlashResult:
        from reservoir_backend.eos.tabulated_pr import flash_tabulated_one

        return flash_tabulated_one(self, eos, pressure, temperature, z, **kwargs)

    def evaluate_batch(
        self,
        eos: PengRobinson,
        pressure: NDArray[np.float64],
        temperature: float,
        z: NDArray[np.float64],
        **kwargs,
    ) -> FlashArrays:
        from reservoir_backend.eos.tabulated_pr import flash_tabulated_batch

        return flash_tabulated_batch(self, eos, pressure, temperature, z, **kwargs)


def flash_cold_warm_pair(
    eos: PengRobinson,
    pressure: float,
    temperature: float,
    z: NDArray[np.float64],
) -> tuple[FlashResult, FlashResult]:
    """Wilson start vs previous-K start. Same equilibrium required."""
    cold = flash_tp(eos, pressure, temperature, z)
    warm = flash_tp(eos, pressure, temperature, z, k_guess=None if cold.k is None else cold.k)
    if cold.k is not None and not _same_flash(cold, warm):
        warm = flash_tp(eos, pressure, temperature, z)
        warm.fallback_used = True
    return cold, warm


def _same_flash(a: FlashResult, b: FlashResult, rtol: float = 1.0e-8) -> bool:
    if bool(a.two_phase) != bool(b.two_phase):
        return False
    if abs(float(a.vapor_frac) - float(b.vapor_frac)) > max(rtol, rtol * abs(float(a.vapor_frac))):
        return False
    if abs(float(a.v_mix) - float(b.v_mix)) > max(rtol * max(abs(float(a.v_mix)), 1.0e-12), 1.0e-14):
        return False
    if float(np.max(np.abs(a.x - b.x))) > rtol:
        return False
    if float(np.max(np.abs(a.y - b.y))) > rtol:
        return False
    return True


def validate_backend(
    backend: FlashBackend,
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    *,
    rtol: float = 1.0e-8,
) -> bool:
    ref = ReferencePRBackend()
    a = ref.evaluate_batch(eos, pressure, temperature, z)
    b = backend.evaluate_batch(eos, pressure, temperature, z)
    if np.any(a.two_phase != b.two_phase):
        return False
    if float(np.max(np.abs(a.vapor_frac - b.vapor_frac))) > rtol:
        return False
    if float(np.max(np.abs(a.v_mix - b.v_mix) / np.maximum(np.abs(a.v_mix), 1.0e-12))) > rtol:
        return False
    if float(np.max(np.abs(a.x - b.x))) > rtol:
        return False
    if float(np.max(np.abs(a.y - b.y))) > rtol:
        return False
    return True


_BACKEND: FlashBackend | None = None


def set_flash_backend(backend: FlashBackend | None) -> None:
    global _BACKEND
    _BACKEND = backend


def get_flash_backend() -> FlashBackend:
    if _BACKEND is not None:
        return _BACKEND
    name = os.environ.get("RESERVOIR_FLASH", "fast").strip().lower()
    if name in {"reference", "ref", "scalar"}:
        return ReferencePRBackend()
    if name in {"tabulated", "obl", "table"}:
        return TabulatedPRBackend()
    return FastPRBackend()
