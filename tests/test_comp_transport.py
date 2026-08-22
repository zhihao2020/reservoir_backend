"""Standalone compositional accumulation + TPFA tests. Not a FIM residual."""

import numpy as np

from reservoir_backend.comp import accumulate_system, component_moles, explicit_step, flash_cell
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def _two_cell_x() -> CartesianGrid:
    return CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)


def _example_binary():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert "EXAMPLE" in mix.marker
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    return mix


def test_example_marker_on_comp_flash() -> None:
    mix = _example_binary()
    cell = flash_cell(np.array([0.40, 0.60]), 250.0, 5.0e6, mix)
    assert "EXAMPLE" in cell.marker
    assert "NOT a Jiyang GEM card" in cell.marker


def test_accumulation_matches_closed_form_and_inventory() -> None:
    """n_i = Vp (ξ_L S_L x_i + ξ_V S_V y_i) = Vp z_i / v_mix."""
    mix = _example_binary()
    z = np.array([0.40, 0.60])
    cell = flash_cell(z, 250.0, 5.0e6, mix)
    vp = 0.20
    n = component_moles(cell, vp)
    v_mix = cell.nu * cell.v_vapor + (1.0 - cell.nu) * cell.v_liquid
    assert np.allclose(n, vp * cell.z / v_mix, atol=1e-12)
    assert np.allclose(n / n.sum(), cell.z, atol=1e-12)
    assert n.sum() > 0.0


def test_two_cell_closed_equal_pressure_conserves_moles() -> None:
    """Equal p → no flux; total n_i conserved after several explicit steps."""
    mix = _example_binary()
    grid = _two_cell_x()
    z = np.array([[0.40, 0.60], [0.55, 0.45]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    for _ in range(4):
        fields = explicit_step(fields, 250.0, p, mix, grid, permeability=1.0e-12, dt=1.0)
    assert np.allclose(fields.n.sum(axis=0), n0.sum(axis=0), atol=1e-10)
    assert np.allclose(fields.n, n0, atol=1e-10)


def test_pressure_driven_light_end_moves_and_totals_hold() -> None:
    """Higher-p cell richer in C1 loses C1; Σ n_i still conserved."""
    mix = _example_binary()
    grid = _two_cell_x()
    z = np.array([[0.70, 0.30], [0.25, 0.75]])
    p = np.array([6.0e6, 4.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    i_c1 = mix.names.index("C1")
    for _ in range(5):
        fields = explicit_step(fields, 250.0, p, mix, grid, permeability=1.0e-12, dt=0.05)
    assert fields.n[0, i_c1] < n0[0, i_c1]
    assert fields.n[1, i_c1] > n0[1, i_c1]
    assert np.allclose(fields.n.sum(axis=0), n0.sum(axis=0), atol=1e-9)
    assert np.all(fields.n >= -1e-14)


def test_gravity_path_conserves_moles() -> None:
    """Optional Φ = p + ρ g z: vertical pair, equal p, totals still close."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((1.0, 1.0, 2.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    fields = explicit_step(
        fields, 250.0, p, mix, grid, permeability=1.0e-12, dt=0.05, gravity=9.81
    )
    assert np.allclose(fields.n.sum(axis=0), n0.sum(axis=0), atol=1e-9)
    assert not np.allclose(fields.n, n0, atol=1e-16)
