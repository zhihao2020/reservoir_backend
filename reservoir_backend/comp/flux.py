"""Thin compositional two-point face flux. Not the black-oil TPFA module.

Transmissibility ``T = k_harm A / d`` (k in m², A in m², d in m → T in m³).
Phase potential first cut: ``Φ_α = p + ρ_α g z`` with ``z`` upward
(``g = 0`` disables gravity). Upwind ``x`` / ``y`` and ``ξ_α`` with ``Φ_α``.

Molar rate of component ``i`` from left to right (mol/s):

    Q_i = Σ_α  T ξ_α^up λ_α^up (Φ_α,L − Φ_α,R) w_{α,i}^up

``λ_α = S_α / μ_α`` (linear kr, EXAMPLE first-cut viscosities). Rewrite of
the published two-point / phase-potential upwind idea; does not import
``discretization.tpfa``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash
from reservoir_backend.grid.cartesian import CartesianGrid

# EXAMPLE first-cut viscosities (Pa·s). Not site-calibrated.
EXAMPLE_MU_LIQUID = 1.0e-4
EXAMPLE_MU_VAPOR = 2.0e-5


@dataclass(frozen=True)
class InteriorFace:
    left: int
    right: int
    transmissibility: float  # m³
    # z_left − z_right (m); used only when g ≠ 0.


def _harmonic(a: float, b: float) -> float:
    if a > 0.0 and b > 0.0:
        return 2.0 * a * b / (a + b)
    return 0.0


def interior_faces(
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
) -> list[InteriorFace]:
    """Interior Cartesian faces with two-point harmonic ``T = k A / d``."""
    k = np.asarray(permeability, dtype=float).ravel()
    if k.size == 1:
        k = np.full(grid.n_cells, float(k[0]), dtype=float)
    if k.size != grid.n_cells:
        raise ValueError(f"permeability size {k.size} != n_cells {grid.n_cells}")
    faces: list[InteriorFace] = []
    for kk in range(grid.nz):
        for jj in range(grid.ny):
            for i in range(grid.nx - 1):
                left = grid.index(i, jj, kk)
                right = grid.index(i + 1, jj, kk)
                area = float(grid.dy[jj] * grid.dz[kk])
                dist = 0.5 * float(grid.dx[i] + grid.dx[i + 1])
                t_ij = _harmonic(float(k[left]), float(k[right])) * area / max(dist, 1.0e-30)
                faces.append(InteriorFace(left, right, t_ij))
    for kk in range(grid.nz):
        for j in range(grid.ny - 1):
            for i in range(grid.nx):
                left = grid.index(i, j, kk)
                right = grid.index(i, j + 1, kk)
                area = float(grid.dx[i] * grid.dz[kk])
                dist = 0.5 * float(grid.dy[j] + grid.dy[j + 1])
                t_ij = _harmonic(float(k[left]), float(k[right])) * area / max(dist, 1.0e-30)
                faces.append(InteriorFace(left, right, t_ij))
    for k_idx in range(grid.nz - 1):
        for jj in range(grid.ny):
            for i in range(grid.nx):
                left = grid.index(i, jj, k_idx)
                right = grid.index(i, jj, k_idx + 1)
                area = float(grid.dx[i] * grid.dy[jj])
                dist = 0.5 * float(grid.dz[k_idx] + grid.dz[k_idx + 1])
                t_ij = _harmonic(float(k[left]), float(k[right])) * area / max(dist, 1.0e-30)
                faces.append(InteriorFace(left, right, t_ij))
    return faces


def _upwind(cell_l: CellFlash, cell_r: CellFlash, dphi: float, phase: str) -> tuple[float, float, NDArray[np.float64]]:
    src = cell_l if dphi >= 0.0 else cell_r
    if phase == "liquid":
        return src.xi_liquid, src.S_liquid, src.x
    return src.xi_vapor, src.S_vapor, src.y


def phase_molar_flux(
    faces: list[InteriorFace],
    cells: list[CellFlash],
    pressure: NDArray[np.float64],
    z_center: NDArray[np.float64],
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
) -> NDArray[np.float64]:
    """Component molar rate L→R on each face, shape ``(n_faces, n_comp)``, mol/s."""
    p = np.asarray(pressure, dtype=float).ravel()
    zc = np.asarray(z_center, dtype=float).ravel()
    if not cells:
        return np.zeros((0, 0), dtype=float)
    nc = cells[0].z.size
    flux = np.zeros((len(faces), nc), dtype=float)
    g = float(gravity)
    mu_l = max(float(mu_liquid), 1.0e-30)
    mu_v = max(float(mu_vapor), 1.0e-30)
    for f_idx, face in enumerate(faces):
        left, right = face.left, face.right
        cell_l, cell_r = cells[left], cells[right]
        dp = float(p[left] - p[right])
        dz = float(zc[left] - zc[right])
        q = np.zeros(nc, dtype=float)
        for phase, mu, rho_l, rho_r in (
            ("liquid", mu_l, cell_l.rho_liquid, cell_r.rho_liquid),
            ("vapor", mu_v, cell_l.rho_vapor, cell_r.rho_vapor),
        ):
            rho = 0.5 * (float(rho_l) + float(rho_r))
            dphi = dp + rho * g * dz
            xi, sat, w = _upwind(cell_l, cell_r, dphi, phase)
            lam = max(sat, 0.0) / mu
            q += face.transmissibility * xi * lam * dphi * w
        flux[f_idx] = q
    return flux
