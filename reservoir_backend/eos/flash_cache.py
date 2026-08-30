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
