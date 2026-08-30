"""Colored-FD Jacobian for compositional DPDP. CSR sparsity, not a dense matrix.

Block structure:

    J = [[J_ff, J_fm],
         [J_mf, J_mm]]

J_ff / J_mm follow the 7-point TPFA stencil. J_fm / J_mf are same-cell transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


@dataclass
class DPDPJacobianPattern:
    """Fixed CSR pattern for one grid and unknown packing (fracture then matrix)."""

    n_cells: int
    nu: int
    n_u: int
    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def empty_data(self) -> NDArray[np.float64]:
        return np.zeros(self.indices.size, dtype=float)

    def to_csr(self, data: NDArray[np.float64]) -> sparse.csc_matrix:
        """Packed by column (CSC). Name kept for call sites."""
        return sparse.csc_matrix(
            (np.asarray(data, dtype=float), self.indices, self.indptr),
            shape=(self.n_u, self.n_u),
        )


def build_sparsity_pattern(n_cells: int, nu: int, neighbors: list[list[int]]) -> DPDPJacobianPattern:
    """One column per unknown. Rows: same-continuum neighbours + same-cell other continuum."""
    n_cells = int(n_cells)
    nu = int(nu)
    half = n_cells * nu
    n_u = 2 * half
    rows_of: list[list[int]] = [[] for _ in range(n_u)]
    for c in range(n_cells):
        neigh = neighbors[c]
        for cont in (0, 1):
            offset = cont * half
            other = (1 - cont) * half
            for slot in range(nu):
                col = offset + c * nu + slot
                acc = rows_of[col]
                for cc in neigh:
                    base = offset + int(cc) * nu
                    acc.extend(range(base, base + nu))
                base_o = other + c * nu
                acc.extend(range(base_o, base_o + nu))
    counts = np.zeros(n_u + 1, dtype=np.int64)
    pieces: list[NDArray[np.int64]] = []
    for col in range(n_u):
        uniq = np.unique(np.asarray(rows_of[col], dtype=np.int64))
        pieces.append(uniq)
        counts[col + 1] = counts[col] + uniq.size
    indices = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
    return DPDPJacobianPattern(n_cells=n_cells, nu=nu, n_u=n_u, indptr=counts, indices=indices)


def residual_scales(n_cells: int, nc: int, n_ref: float, pv_ref: float) -> NDArray[np.float64]:
    """Row scale: mass / n_ref, volume / PV. Two continua packed fracture-then-matrix."""
    nu = nc + 1
    s = np.ones(2 * n_cells * nu, dtype=float)
    block = s.reshape(2, n_cells, nu)
    block[:, :, :nc] = 1.0 / max(float(n_ref), 1.0e-12)
    block[:, :, nc] = 1.0 / max(float(pv_ref), 1.0e-12)
    return s


def fill_column_slice(
    pattern: DPDPJacobianPattern,
    data: NDArray[np.float64],
    col: int,
    dres: NDArray[np.float64],
) -> None:
    sl = slice(int(pattern.indptr[col]), int(pattern.indptr[col + 1]))
    data[sl] = dres[pattern.indices[sl]]
