"""Wellbore head and directional connections against an actual GEM .out excerpt."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.comp.properties import flash_state
from reservoir_backend.comp.wells import perforation_pressures, well_molar_sources
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.io.case import load_case
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.cmg_benchmark import parse_gem_well_connections


def test_physical_3d_injector_allows_crossflow_and_cpor():
    twin = load_case('examples/lab_v1/cmg_gem/physical_3d/case.yaml')
    inj = next(p for p in twin.ports if p.name == 'INJ')
    assert inj.allow_crossflow is True
    prods = [p for p in twin.ports if p.role == 'producer']
    assert all(p.allow_crossflow is False for p in prods)
    rock = twin.rock_from_theta(np.zeros(twin.parameterization.n_params))
    assert rock.cpor == pytest.approx(1.2e-9)
    assert rock.prpor == pytest.approx(5.0e7)
    gm = twin.physics.geomech
    assert gm.enabled
    assert gm.nocouperm
    assert gm.boundary == "unconstrained"
    assert gm.biot == pytest.approx(1.0)
    assert gm.E == pytest.approx(20.0e9)
    assert gm.nu == pytest.approx(0.22)
    assert rock.biot == pytest.approx(1.0)
    assert rock.k_dry == pytest.approx(0.0)
    assert rock.storage_1_per_pa() == pytest.approx(1.2e-9)


def test_gem_connection_parser_uses_delta_column_not_rounded_bhp():
    raw = Path('tests/fixtures/gem_physical_3d_wells_864.txt').read_text()
    rows = parse_gem_well_connections(raw, 864.)
    assert len(rows) == 51
    expected = {('INJ', 11): 126.78, ('PROD2', 15): 2109.9, ('PROD3', 15): 2110.0,
                ('PROD4', 11): -10.544}
    for (well, k), dp in expected.items():
        row = next(r for r in rows if r['well'] == well and r['ijk'][2] == k)
        assert row['bhp_minus_block_pa'] == pytest.approx(dp)
    with pytest.raises(ValueError, match='no GEM'):
        parse_gem_well_connections(raw, 8.64)


def test_well_head_against_gem_snippet_at_gem_block_pressure():
    # Isolate the well equation from reservoir F: feed the independently
    # exported GEM block map here only (never to F or inversion). Its printed
    # pressure resolution is 100 Pa, hence a 50 Pa half-bin allowance.
    twin=load_case('examples/lab_v1/cmg_gem/physical_3d/case.yaml')
    p=np.load('examples/lab_v1/cmg_gem/physical_3d/export/hidden/pressure.npy')[1]
    props=flash_state(twin.physics.fluid,p,np.broadcast_to(twin.physics.fluid.z_init,(len(p),7)))
    rows=parse_gem_well_connections(Path('tests/fixtures/gem_physical_3d_wells_864.txt').read_text(),864.)
    for port in twin.ports:
        row=[r for r in rows if r['well']==port.name][-1]
        i,j,k=row['ijk']
        cell=twin.grid.index(i-1,j-1,twin.grid.nz-k)
        slot=list(port.cell_ids).index(cell)
        pw=perforation_pressures(twin.grid,port,twin.physics.fluid,props,5e7)
        assert pw[slot]-p[cell] == pytest.approx(row['bhp_minus_block_pa'],abs=50.)


@pytest.mark.parametrize('role', ['injector', 'producer'])
def test_head_datum_sign_density_and_no_reverse_flow(role):
    twin = load_case('examples/lab_v1/cmg_gem/physical_3d/case.yaml')
    spec = twin.physics.fluid
    grid = CartesianGrid.uniform((.02,.02,.30), .02)
    rock = Rock.uniform(15, k=1.776e-17, phi=.0367)
    port = FlowPort('W', role, 'pressure', np.arange(15), use_productivity=True,
                    rw_m=.003, geofac=.34)
    p = np.full(15, 5e7)
    props = flash_state(spec, p, np.broadcast_to(spec.z_init, (15,7)).copy())
    pw = perforation_pressures(grid, port, spec, props, 5e7)
    assert pw[-1] == 5e7  # top completion is BHP datum
    assert np.all(np.diff(pw) < 0)  # our z increases upward
    # GEM full-column producer prints ~2.11 kPa BHP-Pblock; computed head
    # is ~2.30 kPa, with independent PR wellbore density (no fitted density).
    assert 1800 < pw[0] - pw[-1] < 2600
    mid = replace(port, bhp_reference_z_m=.15)
    pw_mid = perforation_pressures(grid, mid, spec, props, 5e7)
    assert pw_mid[0] > 5e7 > pw_mid[-1]
    flat = perforation_pressures(grid, replace(port, use_productivity=False), spec, props, 5e7)
    np.testing.assert_array_equal(flat, p)
    controls = {('W','pressure'): ControlSeries('W','pressure',np.array([0.,864.]),np.array([5e7,5e7]))}
    q, _, bhp = well_molar_sources(grid, rock, [port], controls, p, props, spec, 864.)
    assert bhp['W'] == 5e7
    if role == 'producer':
        np.testing.assert_array_equal(q, 0.)  # same as positive GEM producer Delta-p rows
    else:
        assert q.sum() > 0
        np.testing.assert_allclose(q.sum(axis=0)/q.sum(),spec.z_inj)


def test_hydrostatic_bhp_jacobian_matches_column_fd():
    from reservoir_backend.solver.fi_comp import _well_jacobian
    twin = load_case('examples/lab_v1/cmg_gem/physical_3d/case.yaml')
    spec = twin.physics.fluid
    grid = CartesianGrid.uniform((.02,.02,.06),.02)
    rock = Rock.uniform(3,k=1.776e-17,phi=.0367)
    port = FlowPort('P','producer','pressure',np.arange(3),use_productivity=True,rw_m=.003,geofac=.34)
    p = np.full(3,5.001e7)
    n = np.broadcast_to(spec.z_init,(3,7)).copy() * .002
    props = flash_state(spec,p,n)
    ctrl = {('P','pressure'): ControlSeries('P','pressure',np.array([0.,1.]),np.array([5e7,5e7]))}
    jac = _well_jacobian(grid,rock,spec,[port],ctrl,n,p,props,1.,1.,.002,5e7).toarray()
    for c, slot in [(0,0),(0,7),(2,4),(2,7)]:
        eps=1e-8 if slot<7 else .5
        n1,p1=n.copy(),p.copy()
        if slot<7: n1[c,slot]+=eps
        else: p1[c]+=eps
        q0=well_molar_sources(grid,rock,[port],ctrl,p,props,spec,1.)[0]
        q1=well_molar_sources(grid,rock,[port],ctrl,p1,flash_state(spec,p1,n1),spec,1.)[0]
        ref=np.zeros((3,8));ref[:,:7]=-(q1-q0)/eps
        np.testing.assert_allclose(jac[:,c*8+slot],ref.ravel(),rtol=2e-4,atol=1e-15)
