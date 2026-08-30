"""Experimental PVT → tuned PR → lumped pseudo-components.

V1 still uses the published C1–nC10 EXAMPLE card. This module is the
handoff when laboratory PVT exists; it does not invent Jiyang criticals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.lumping import lump_peng_robinson
from reservoir_backend.eos.pr import PengRobinson


@dataclass
class PVTFitReport:
    n_raw: int
    n_lumped: int
    names: tuple[str, ...]
    notes: list[str]


def lump_experimental_eos(
    eos: PengRobinson,
    z: NDArray[np.float64],
    groups: list[list[int]],
) -> tuple[PengRobinson, NDArray[np.float64], PVTFitReport]:
    """Keep 3–6 pseudo-components before a 30³ FIM. Grouping is an input."""
    lumped, z2 = lump_peng_robinson(eos, z, groups)
    report = PVTFitReport(
        n_raw=int(eos.nc),
        n_lumped=int(lumped.nc),
        names=lumped.names,
        notes=["lumped from measured/characterized EOS; not a default five-cut split"],
    )
    if lumped.nc < 3 or lumped.nc > 6:
        report.notes.append(f"n_lumped={lumped.nc} is outside the 3–6 target for 30³ FIM")
    return lumped, z2, report
