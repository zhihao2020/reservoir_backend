"""Pseudo-component lumping for a later real-shale-oil PVT card.

V1 stays C1–nC10. When a many-component laboratory fluid is fitted, lump
first so reservoir unknowns stay ``2 N_cell (N_c+1)``. Grouping is an
input, not a fixed C1 / C2-C3 / … split.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.pr import PengRobinson


def lump_peng_robinson(
    eos: PengRobinson,
    z: NDArray[np.float64],
    groups: list[list[int]],
) -> tuple[PengRobinson, NDArray[np.float64]]:
    """Mole-weighted criticals inside each group. ``kij`` uses the first member."""
    z = np.asarray(z, dtype=float).ravel()
    if z.size != eos.nc:
        raise ValueError("z size must match eos.nc")
    if not groups or any(len(g) < 1 for g in groups):
        raise ValueError("each lumping group needs at least one component index")
    names: list[str] = []
    tc = []
    pc = []
    omega = []
    mw = []
    z_out = []
    heads: list[int] = []
    for g in groups:
        idx = np.asarray(g, dtype=int)
        w = np.maximum(z[idx], 0.0)
        s = float(np.sum(w))
        if s <= 0.0:
            w = np.ones(idx.size) / idx.size
            s = 1.0
        else:
            w = w / s
        tc.append(float(np.dot(w, eos.tc[idx])))
        pc.append(float(np.dot(w, eos.pc[idx])))
        omega.append(float(np.dot(w, eos.omega[idx])))
        mw.append(float(np.dot(w, eos.mw[idx])))
        z_out.append(float(np.sum(z[idx])))
        heads.append(int(idx[0]))
        if len(idx) == 1:
            names.append(eos.names[int(idx[0])])
        else:
            names.append(f"{eos.names[int(idx[0])]}-{eos.names[int(idx[-1])]}")
    n = len(groups)
    kij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kij[i, j] = float(eos.kij[heads[i], heads[j]])
    lumped = PengRobinson(
        tc=np.asarray(tc, dtype=float),
        pc=np.asarray(pc, dtype=float),
        omega=np.asarray(omega, dtype=float),
        mw=np.asarray(mw, dtype=float),
        kij=kij,
        names=tuple(names),
    )
    z_l = np.maximum(np.asarray(z_out, dtype=float), 0.0)
    z_l = z_l / max(float(np.sum(z_l)), 1.0e-30)
    return lumped, z_l
