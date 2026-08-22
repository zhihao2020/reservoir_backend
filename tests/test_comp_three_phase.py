"""Immiscible EXAMPLE aqueous + PR oil/gas. Not FIM, not GEM brine."""

import numpy as np

from reservoir_backend.comp import (
    EXAMPLE_AQUEOUS_ASSUMPTIONS,
    EXAMPLE_AQUEOUS_MARKER,
    accumulate_three_phase,
    explicit_step_three_phase,
    flash_cell,
    three_phase_saturations,
    water_moles,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def _example_binary():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    return mix


def test_aqueous_marker_is_example_not_gem() -> None:
    assert "EXAMPLE" in EXAMPLE_AQUEOUS_MARKER
    assert "NOT a Jiyang GEM card" in EXAMPLE_AQUEOUS_MARKER
    assert "immiscible" in EXAMPLE_AQUEOUS_ASSUMPTIONS
    assert "no Pc" in EXAMPLE_AQUEOUS_MARKER or "capillary-free" in EXAMPLE_AQUEOUS_ASSUMPTIONS


def test_three_phase_saturations_sum_to_one() -> None:
    mix = _example_binary()
    cell = flash_cell(np.array([0.40, 0.60]), 250.0, 5.0e6, mix)
    so, sg, sw = three_phase_saturations(cell, 0.25)
    assert abs(so + sg + sw - 1.0) < 1e-12
    assert 0.0 <= sw <= 1.0
    assert 0.0 <= so <= 1.0
    assert 0.0 <= sg <= 1.0
    so0, sg0, sw0 = three_phase_saturations(cell, 0.0)
    assert sw0 == 0.0
    assert abs(so0 - cell.S_liquid) < 1e-12
    assert abs(sg0 - cell.S_vapor) < 1e-12
    assert abs(so0 + sg0 - 1.0) < 1e-12


def test_sw_in_unit_interval_on_accumulate() -> None:
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.55, 0.45]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    sw0 = np.array([0.20, 0.35])
    state = accumulate_three_phase(z, 250.0, p, mix, vp, sw0)
    assert np.all(state.s_water >= 0.0) and np.all(state.s_water <= 1.0)
    assert np.allclose(state.s_oil + state.s_gas + state.s_water, 1.0, atol=1e-12)
    assert np.allclose(state.n_water, [water_moles(0.20, float(vp[0])), water_moles(0.35, float(vp[1]))])
    assert state.marker == EXAMPLE_AQUEOUS_MARKER
    assert "EXAMPLE" in state.hc.cells[0].marker


def test_three_phase_closed_equal_p_conserves_water_and_hc() -> None:
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.55, 0.45]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    state = accumulate_three_phase(z, 250.0, p, mix, vp, np.array([0.20, 0.35]))
    n_hc0 = state.hc.n.copy()
    n_w0 = state.n_water.copy()
    for _ in range(4):
        state = explicit_step_three_phase(
            state, 250.0, p, mix, grid, 1.0e-12, vp, dt=1.0
        )
    assert np.allclose(state.hc.n.sum(axis=0), n_hc0.sum(axis=0), atol=1e-10)
    assert np.allclose(state.n_water.sum(), n_w0.sum(), atol=1e-10)
    assert np.allclose(state.hc.n, n_hc0, atol=1e-10)
    assert np.allclose(state.n_water, n_w0, atol=1e-10)
    assert np.all((state.s_water >= 0.0) & (state.s_water <= 1.0))
    assert np.allclose(state.s_oil + state.s_gas + state.s_water, 1.0, atol=1e-12)


def test_three_phase_pressure_driven_water_and_hc_totals_hold() -> None:
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.70, 0.30], [0.25, 0.75]])
    p = np.array([6.0e6, 4.0e6])
    vp = 0.2 * grid.cell_volumes()
    state = accumulate_three_phase(z, 250.0, p, mix, vp, np.array([0.40, 0.15]))
    n_hc0 = state.hc.n.copy()
    n_w0 = state.n_water.copy()
    i_c1 = mix.names.index("C1")
    for _ in range(5):
        state = explicit_step_three_phase(
            state, 250.0, p, mix, grid, 1.0e-12, vp, dt=0.05
        )
    assert state.n_water[0] < n_w0[0]
    assert state.n_water[1] > n_w0[1]
    assert state.hc.n[0, i_c1] < n_hc0[0, i_c1]
    assert np.allclose(state.hc.n.sum(axis=0), n_hc0.sum(axis=0), atol=1e-9)
    assert np.allclose(state.n_water.sum(), n_w0.sum(), atol=1e-9)
    assert np.all(state.hc.n >= -1e-14)
    assert np.all(state.n_water >= -1e-14)
    assert np.all((state.s_water >= 0.0) & (state.s_water <= 1.0))
    assert np.allclose(state.s_oil + state.s_gas + state.s_water, 1.0, atol=1e-12)
