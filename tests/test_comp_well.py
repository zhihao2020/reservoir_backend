"""Standalone Peaceman rate injector. Not FIM, not GEM, not industrial-grade."""

import numpy as np

from reservoir_backend.comp import (
    accumulate_system,
    example_rate_injector,
    explicit_step,
    peaceman_wi,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def _example_binary():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert "EXAMPLE" in mix.marker
    return mix


def test_peaceman_wi_positive_finite() -> None:
    """WI = 2π k h / ln(r_e/r_w) with r_e = 0.14 sqrt(dx²+dy²)."""
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 1.0)
    k = 1.0e-12
    wi, r_e, r_w = peaceman_wi(grid, 0, k)
    assert np.isfinite(wi) and wi > 0.0
    assert r_e > r_w > 0.0
    assert np.isclose(r_e, 0.14 * np.sqrt(2.0))
    wi2, _, _ = peaceman_wi(grid, 0, 2.0 * k)
    assert np.isclose(wi2, 2.0 * wi)


def test_inject_pure_co2_inventory_balance() -> None:
    """After N steps, Δn_CO2 = injected moles; C1 unchanged (pure CO2 stream)."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60]])
    p = np.array([5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    rate = 0.02
    dt = 1.0
    n_steps = 5
    inj = example_rate_injector(grid, 0, 1.0e-12, mix, rate=rate, stream="CO2")
    assert inj.well_index > 0.0 and np.isfinite(inj.well_index)
    for _ in range(n_steps):
        fields = explicit_step(
            fields, 250.0, p, mix, grid, permeability=1.0e-12, dt=dt, injectors=(inj,)
        )
    i_c1 = mix.names.index("C1")
    i_co2 = mix.names.index("CO2")
    injected = rate * dt * n_steps
    assert np.isclose(fields.n[:, i_co2].sum() - n0[:, i_co2].sum(), injected, atol=1e-12)
    assert np.isclose(fields.n[:, i_c1].sum(), n0[:, i_c1].sum(), atol=1e-12)
    assert np.allclose(fields.n.sum(axis=0) - n0.sum(axis=0), np.array([0.0, injected]), atol=1e-12)


def test_two_cell_inject_co2_totals_match_source() -> None:
    """Equal-p two-cell: well source is the only inventory change."""
    mix = _example_binary()
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    rate, dt, n_steps = 0.01, 0.5, 4
    inj = example_rate_injector(grid, 0, 1.0e-12, mix, rate=rate, stream="CO2")
    for _ in range(n_steps):
        fields = explicit_step(
            fields, 250.0, p, mix, grid, permeability=1.0e-12, dt=dt, injectors=(inj,)
        )
    injected = rate * dt * n_steps
    i_c1 = mix.names.index("C1")
    i_co2 = mix.names.index("CO2")
    assert np.isclose(fields.n[:, i_co2].sum() - n0[:, i_co2].sum(), injected, atol=1e-12)
    assert np.isclose(fields.n[:, i_c1].sum(), n0[:, i_c1].sum(), atol=1e-12)


def test_injector_carries_example_marker() -> None:
    mix = _example_binary()
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 1.0)
    inj = example_rate_injector(grid, 0, 1.0e-12, mix, rate=0.01, stream="CO2")
    assert "EXAMPLE" in inj.marker
    assert inj.marker == EXAMPLE_LIBRARY_MARKER
    assert "NOT a Jiyang GEM card" in inj.marker
    assert inj.z_inj[mix.names.index("CO2")] == 1.0
    assert inj.z_inj[mix.names.index("C1")] == 0.0
