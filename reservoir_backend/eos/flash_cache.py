"""Per-continuum flash cache. Guess / phase hint only, never a silent result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FlashCache:
    k: NDArray[np.float64] | None = None
    two_phase: NDArray[np.bool_] | None = None
    vapor_frac: NDArray[np.float64] | None = None
    p_last: NDArray[np.float64] | None = None
    z_last: NDArray[np.float64] | None = None
    x: NDArray[np.float64] | None = None
    y: NDArray[np.float64] | None = None
    converged: NDArray[np.bool_] | None = None

    @classmethod
    def from_props(cls, props) -> FlashCache:
        return cls(
            k=None if props.k_flash is None else np.asarray(props.k_flash, dtype=float).copy(),
            two_phase=None if props.two_phase is None else np.asarray(props.two_phase, dtype=bool).copy(),
            vapor_frac=None if props.vapor_frac is None else np.asarray(props.vapor_frac, dtype=float).copy(),
            p_last=None if props.p_flash is None else np.asarray(props.p_flash, dtype=float).copy(),
            z_last=None if props.z_flash is None else np.asarray(props.z_flash, dtype=float).copy(),
            x=None if props.x is None else np.asarray(props.x, dtype=float).copy(),
            y=None if props.y is None else np.asarray(props.y, dtype=float).copy(),
        )

    def copy(self) -> FlashCache:
        def _c(a):
            return None if a is None else np.copy(a)

        return FlashCache(
            k=_c(self.k),
            two_phase=_c(self.two_phase),
            vapor_frac=_c(self.vapor_frac),
            p_last=_c(self.p_last),
            z_last=_c(self.z_last),
            x=_c(self.x),
            y=_c(self.y),
            converged=_c(self.converged),
        )


@dataclass
class DualFlashCache:
    fracture: FlashCache | None = None
    matrix: FlashCache | None = None

    def copy(self) -> DualFlashCache:
        return DualFlashCache(
            fracture=None if self.fracture is None else self.fracture.copy(),
            matrix=None if self.matrix is None else self.matrix.copy(),
        )


@dataclass
class DPDPCheckpoint:
    """Physical state plus flash guess. Cache is never a silent equilibrium."""

    state: object
    flash: DualFlashCache | None = None

    def copy(self) -> DPDPCheckpoint:
        st = self.state.copy() if self.state is not None and hasattr(self.state, "copy") else self.state
        return DPDPCheckpoint(state=st, flash=None if self.flash is None else self.flash.copy())
