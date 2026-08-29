"""FIM ≈ sequential consistency on a small dead-oil three-phase case."""

from __future__ import annotations

import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi import _lambda
from reservoir_backend.solver.impes import _cell_mobility, simulate


def test_dead_oil_lambda_matches_cell_mobility_when_pvt_mu_differs() -> None:
    """Regression: FIM must use Corey μ for dead oil, not PVT default μ_o=5e-3."""
    rp = CoreyThreePhase(mu_w=1.1e-3, mu_o=0.64e-3, mu_g=2.08e-5)
    pvt = BlackOilPVT.slightly_compressible(1.0e-9, pref=1.0e5)  # default mu_o=5e-3
    assert not pvt.has_live_oil()
    n = 4
    sw = np.full(n, 0.2)
    sg = np.zeros(n)
    p = np.full(n, 1.0e5)
    lw_f, lo_f, lg_f = _lambda(rp, pvt, sw, sg, p)
    lw_s, lo_s, lg_s, _ = _cell_mobility(
        CoreyTwoPhase(mu_w=rp.mu_w, mu_o=rp.mu_o),
        rp,
        sw,
        sg,
        pvt,
        p,
        single_phase=False,
        mu_single=1.0,
    )
    assert np.allclose(lw_f, lw_s)
    assert np.allclose(lo_f, lo_s)
    assert np.allclose(lg_f, lg_s)
    assert float(lo_f[0]) > 1000.0  # 1/0.64e-3, not 1/5e-3


def _dead_oil_case(t_end: float = 8.0):
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.2)
    rp = CoreyThreePhase()
    pvt = BlackOilPVT.slightly_compressible(1.0e-9, pref=1.1e5, mu_w=rp.mu_w, mu_o=rp.mu_o)
    assert not pvt.has_live_oil()
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=0.8)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04))
    q = 5.0e-9
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([q, q])),
        ControlSeries("INJ", "composition", times, np.array([0.8, 0.8])),
        ControlSeries("INJ", "gas_composition", times, np.array([0.05, 0.05])),
        ControlSeries("PROD", "pressure", times, np.array([1.0e5, 1.0e5])),
    ]
    state0 = State(
        pressure=np.full(grid.n_cells, 1.1e5),
        sw=np.full(grid.n_cells, 0.22),
        sg=np.full(grid.n_cells, 0.05),
    )
    return grid, rock, rp, pvt, [inj, prod], controls, state0


def _simulate(fully_implicit: bool):
    t_end = 8.0
    grid, rock, rp, pvt, ports, controls, state0 = _dead_oil_case(t_end)
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=rp.mu_w, mu_o=rp.mu_o),
        ports,
        controls,
        state0,
        t_end,
        three_phase=rp,
        pvt=pvt,
        capillary=NoCapillary(),
        gravity=0.0,
        implicit=True,
        fully_implicit=bool(fully_implicit),
        dt_init=1.0,
        dt_min=0.05,
        dt_max=2.0,
        max_cfl=0.5,
        max_ds=0.2,
        max_steps=5000,
    )
    st = traj.states[-1]
    sg = np.zeros(grid.n_cells) if st.sg is None else np.asarray(st.sg, dtype=float)
    return st.pressure, st.sw, sg


def test_fim_matches_sequential_dead_oil_small() -> None:
    """Same discrete equations → short-time FIM and sequential must stay close."""
    p_s, sw_s, sg_s = _simulate(False)
    p_f, sw_f, sg_f = _simulate(True)
    pref = max(float(np.mean(np.abs(p_s))), 1.0e5)
    dp_rel = float(np.max(np.abs(p_f - p_s)) / pref)
    dsw = float(np.max(np.abs(sw_f - sw_s)))
    dsg = float(np.max(np.abs(sg_f - sg_s)))
    assert dp_rel < 0.02, f"pressure rel max {dp_rel}"
    assert dsw < 0.05, f"sw max {dsw}"
    assert dsg < 0.05, f"sg max {dsg}"
