"""Immiscible EXAMPLE aqueous + PR oil/gas. Not FIM, not GEM brine."""

import numpy as np

from reservoir_backend.comp import (
    EXAMPLE_AQUEOUS_ASSUMPTIONS,
    EXAMPLE_AQUEOUS_MARKER,
    THREE_PHASE_VOLUME_CONSTRAINT,
    accumulate_three_phase,
    example_rate_injector,
    explicit_step_three_phase,
    flash_cell,
    implicit_newton_step_three_phase,
    three_phase_saturations,
    water_moles,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
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
    assert "Newton" in EXAMPLE_AQUEOUS_ASSUMPTIONS
    assert "R_p" in EXAMPLE_AQUEOUS_ASSUMPTIONS
    assert "no Pc" in EXAMPLE_AQUEOUS_MARKER or "capillary-free" in EXAMPLE_AQUEOUS_ASSUMPTIONS


def test_three_phase_volume_constraint_is_documented() -> None:
    assert "n_hc_tot * v_mix" in THREE_PHASE_VOLUME_CONSTRAINT
    assert "n_w * v_w" in THREE_PHASE_VOLUME_CONSTRAINT
    assert "V_pore" in THREE_PHASE_VOLUME_CONSTRAINT
    assert "Newton unknown" in THREE_PHASE_VOLUME_CONSTRAINT


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


def test_three_phase_newton_equal_p_conserves_water_and_hc() -> None:
    """Coupled (n_i, n_w, p): equal p holds totals; Sw exists; So+Sg+Sw=1."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    state = accumulate_three_phase(z, 250.0, p, mix, vp, np.array([0.20, 0.20]))
    n_hc0 = state.hc.n.copy()
    n_w0 = state.n_water.copy()
    report = implicit_newton_step_three_phase(state, 250.0, p, mix, grid, 1.0e-12, vp, dt=1.0)
    assert report.newton_converged
    assert report.has_pressure_unknown
    assert report.n_unknowns == 2 * (mix.n_components + 2)
    assert report.pressure is not None
    assert report.pressure.shape == (grid.n_cells,)
    out = report.state
    assert np.allclose(out.hc.n.sum(axis=0), n_hc0.sum(axis=0), atol=1e-9)
    assert np.allclose(out.n_water.sum(), n_w0.sum(), atol=1e-9)
    assert np.all((out.s_water >= 0.0) & (out.s_water <= 1.0))
    assert np.allclose(out.s_oil + out.s_gas + out.s_water, 1.0, atol=1e-12)


def test_three_phase_newton_includes_p_and_drops_residual() -> None:
    """Inject step: p is in the unknown vector; ||R|| drops by decades."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p0 = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    state = accumulate_three_phase(z, 250.0, p0, mix, vp, np.array([0.25, 0.25]))
    inj = example_rate_injector(grid, 0, 1.0e-18, mix, rate=1.0e-4)
    report = implicit_newton_step_three_phase(
        state, 250.0, p0, mix, grid, 1.0e-18, vp, 0.25 * SECONDS_PER_DAY, injectors=(inj,)
    )
    assert report.has_pressure_unknown
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 2)
    assert report.pressure is not None
    assert report.pressure.shape == (grid.n_cells,)
    assert report.newton_converged
    assert report.n_newton >= 1
    assert len(report.residual_hist) >= 2
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0
    assert np.max(np.abs(report.pressure - p0)) > 1.0
    out = report.state
    assert np.allclose(out.s_oil + out.s_gas + out.s_water, 1.0, atol=1e-12)


def test_three_phase_newton_pressure_driven_residual_drops() -> None:
    """Water is in the Newton residual; ||R|| drops; mass and So+Sg+Sw=1 hold."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.70, 0.30], [0.25, 0.75]])
    p = np.array([6.0e6, 4.0e6])
    vp = 0.2 * grid.cell_volumes()
    state = accumulate_three_phase(z, 250.0, p, mix, vp, np.array([0.40, 0.15]))
    n_hc0 = state.hc.n.copy()
    n_w0 = state.n_water.copy()
    report = implicit_newton_step_three_phase(state, 250.0, p, mix, grid, 1.0e-12, vp, dt=0.05)
    assert report.newton_converged
    assert report.has_pressure_unknown
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 2)
    assert report.residual_hist[-1] < report.residual_hist[0]
    out = report.state
    assert out.n_water[0] < n_w0[0]
    assert out.n_water[1] > n_w0[1]
    assert np.allclose(out.hc.n.sum(axis=0), n_hc0.sum(axis=0), atol=1e-9)
    assert np.allclose(out.n_water.sum(), n_w0.sum(), atol=1e-9)
    assert np.all((out.s_water >= 0.0) & (out.s_water <= 1.0))
    assert np.allclose(out.s_oil + out.s_gas + out.s_water, 1.0, atol=1e-12)
