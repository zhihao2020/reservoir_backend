import numpy as np

from reservoir_backend.domain.types import State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import BrooksCorey, NoCapillary, VanGenuchten
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.impes import simulate


def test_brooks_corey_monotone_decreasing() -> None:
    pc = BrooksCorey(entry_pressure=2.0e3, lambda_pc=2.0, swi=0.2, sor=0.2)
    sw = np.linspace(0.21, 0.79, 20)
    values = pc.pc(sw)
    assert np.all(np.diff(values) < 0.0)
    assert values[0] > values[-1]


def test_no_capillary_zero() -> None:
    assert np.all(NoCapillary().pc(np.array([0.2, 0.5, 0.8])) == 0.0)


def test_van_genuchten_positive() -> None:
    vg = VanGenuchten()
    sw = np.linspace(0.25, 0.7, 8)
    assert np.all(vg.pc(sw) >= 0.0)


def test_capillary_imbibes_into_dry_side() -> None:
    """Pc in the phase potential must drive water into the dry cells."""
    grid = CartesianGrid.uniform((0.32, 0.04, 0.04), 0.04)
    rock = Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20)
    sw0 = np.array([0.75 if grid.ijk(c)[0] < 4 else 0.25 for c in range(grid.n_cells)])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [],
        [],
        State(pressure=np.full(grid.n_cells, 1.5e5), sw=sw0),
        t_end=8.0,
        capillary=BrooksCorey(entry_pressure=5.0e3, lambda_pc=2.0),
        pvt=BlackOilPVT.slightly_compressible(2.0e-9, pref=1.5e5, mu_w=1e-3, mu_o=1e-3),
        dt_init=0.5,
        dt_max=1.0,
        max_cfl=0.40,
        max_ds=0.12,
    )
    last = traj.states[-1].sw
    assert float(np.mean(last[:4])) < float(np.mean(sw0[:4])) - 0.02
    assert float(np.mean(last[4:])) > float(np.mean(sw0[4:])) + 0.02
    impl = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [],
        [],
        State(pressure=np.full(grid.n_cells, 1.5e5), sw=sw0),
        t_end=8.0,
        capillary=BrooksCorey(entry_pressure=5.0e3, lambda_pc=2.0),
        pvt=BlackOilPVT.slightly_compressible(2.0e-9, pref=1.5e5, mu_w=1e-3, mu_o=1e-3),
        implicit=True,
        dt_init=1.0,
        dt_max=2.0,
        max_cfl=0.40,
        max_ds=0.12,
    )
    last_i = impl.states[-1].sw
    assert float(np.mean(last_i[:4])) < float(np.mean(sw0[:4])) - 0.02
    assert float(np.mean(last_i[4:])) > float(np.mean(sw0[4:])) + 0.02


def test_gravity_segregates_water_down() -> None:
    """Phase fluxes, not fw*vT, make water fall when total velocity is ~0."""
    nz = 8
    grid = CartesianGrid(
        nx=1,
        ny=1,
        nz=nz,
        dx=np.array([0.05]),
        dy=np.array([0.05]),
        dz=np.full(nz, 0.04),
    )
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.20)
    pvt = BlackOilPVT(
        cr=2.0e-9,
        pref_r=1.5e5,
        pref_w=1.5e5,
        pref_o=1.5e5,
        mu_w=1.0e-3,
        mu_o=1.0e-3,
        rho_w_sc=1000.0,
        rho_o_sc=200.0,
    )
    z = grid.cell_centers()[:, 2]
    rho = 0.5 * pvt.rho_w_sc + 0.5 * pvt.rho_o_sc
    p0 = 1.5e5 - rho * 9.81 * (z - float(np.mean(z)))
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [],
        [],
        State(pressure=p0, sw=np.full(grid.n_cells, 0.50)),
        t_end=40.0,
        capillary=NoCapillary(),
        pvt=pvt,
        gravity=9.81,
        dt_init=0.5,
        dt_max=1.0,
        max_cfl=0.40,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    assert float(sw[0]) > float(sw[-1]) + 0.02


def test_default_init_matches_pres_con() -> None:
    """Virtual-experiment *PRES *CON: uniform p. Hydrostatic is opt-in."""
    from reservoir_backend.domain.types import Experiment
    from reservoir_backend.inverse.parameterization import RegionParameterization
    from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec

    grid = CartesianGrid(
        nx=1, ny=1, nz=4, dx=np.array([1.0]), dy=np.array([1.0]), dz=np.full(4, 1.0)
    )
    param = RegionParameterization(np.zeros(grid.n_cells, dtype=np.int64), phi=0.20)
    exp = Experiment(size_m=grid.size_m(), sensors=[], controls=[], observations=[])
    phys = PhysicsSpec(gravity=9.81, p_init=2.0e6, hydrostatic_init=False)
    twin = DigitalTwin(grid, exp, [], phys, param, inverse=InverseSpec(n_ensemble=2, n_assimilations=1))
    assert np.allclose(twin.initial_state().pressure, 2.0e6)
    phys_h = PhysicsSpec(gravity=9.81, p_init=2.0e6, hydrostatic_init=True)
    twin_h = DigitalTwin(grid, exp, [], phys_h, param, inverse=InverseSpec(n_ensemble=2, n_assimilations=1))
    ph = twin_h.initial_state().pressure
    assert float(ph[0]) > float(ph[-1])
