"""Gravity upwinding and dead-oil phase classification regression tests.

- The gas operator must take the face mobility upstream of the *gas potential*
  ``p + rho_g*z``: a sinking gas (``rho_g > 0``) above a gas-free cell drains into
  it, and a gas-free cell never loses gas. The old ``A - rho_g*A_grav`` form took
  the lower cell's mobility, i.e. the downstream one for a sinking phase.
- ``flash_direct_volumes`` must classify phases on the same dead-oil base it
  flashes on.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.cartesian import CartesianGrid
from src.core.pr_eos import _CO2_IDX, _Z_OIL_DEAD, _ab, _flash_vec, flash_direct_volumes
from src.programs.forward import _mobility_divergence_matrix, _phase_divergence_matrix

RHO_G = 759.0  # Pa/m, shale_oil case: CO2-rich phase denser than oil


def _column():
    grid = CartesianGrid(nx=1, ny=1, nz=2, dx=0.1, dy=0.1, dz=0.1)  # cell 0 bottom, cell 1 top
    return grid, np.full(2, 1.0e-15), np.full(2, 20.0e6)


def test_sinking_gas_drains_from_top_cell():
    grid, k, p = _column()
    A_g = _phase_divergence_matrix(grid, k, p, RHO_G)
    out = A_g @ np.array([0.0, 1.0])  # gas mobility only in the top cell
    assert out[1] > 0.0  # top loses gas
    assert out[0] < 0.0  # bottom receives it
    assert np.isclose(out.sum(), 0.0, atol=1e-30)


def test_gas_free_cell_never_loses_gas():
    grid, k, p = _column()
    A_g = _phase_divergence_matrix(grid, k, p, RHO_G)
    out = A_g @ np.array([1.0, 0.0])  # gas only at the bottom: nothing below to sink into
    assert np.allclose(out, 0.0, atol=1e-30)


def test_buoyant_gas_rises():
    grid, k, p = _column()
    A_g = _phase_divergence_matrix(grid, k, p, -RHO_G)
    out = A_g @ np.array([1.0, 0.0])
    assert out[0] > 0.0 and out[1] < 0.0


def test_zero_gravity_reduces_to_pressure_operator():
    grid, k, _ = _column()
    p = np.array([20.1e6, 20.0e6])
    A = _mobility_divergence_matrix(grid, k, p)
    A_g = _phase_divergence_matrix(grid, k, p, 0.0)
    assert np.allclose(A.toarray(), A_g.toarray())


def test_strong_pressure_gradient_overrides_gravity_upwind():
    # Upward pressure drive much larger than the gravity head: flow goes up, so the
    # bottom cell is upstream even though the gas is denser.
    grid, k, _ = _column()
    p = np.array([20.0e6 + 1.0e4, 20.0e6])  # 10 kPa over 0.1 m >> 759 Pa/m * 0.1 m
    A_g = _phase_divergence_matrix(grid, k, p, RHO_G)
    out = A_g @ np.array([1.0, 0.0])
    assert out[0] > 0.0 and out[1] < 0.0


@pytest.mark.parametrize("zc", np.linspace(0.30, 0.40, 11))
def test_dead_oil_classification_matches_exact_flash(zc):
    p = 20.0e6
    V, *_ = flash_direct_volumes(np.array([p]), np.array([zc]))
    z = (1.0 - zc) * _Z_OIL_DEAD
    z[_CO2_IDX] += zc
    aij, b = _ab(393.0)
    V_exact, _, _ = _flash_vec(z[None, :], np.array([p]), 393.0, aij, b)
    assert abs(float(V[0]) - float(V_exact[0])) < 1.0e-4
