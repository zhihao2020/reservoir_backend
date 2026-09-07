import numpy as np
import pytest

from reservoir_backend.comp.lbc import lbc_viscosity
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.io.case import load_case
from reservoir_backend.io.eos_load import load_eos_card


def test_physical_3d_card_loads_vshift_vcrit_and_phaseid_crit() -> None:
    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    fluid = twin.physics.fluid
    assert fluid is not None
    assert fluid.eos.vcrit is not None
    np.testing.assert_allclose(fluid.eos.vshift[0], -0.0943)
    assert fluid.phaseid == "crit"


def test_phaseid_crit_labels_pure_co2_as_gas() -> None:
    from reservoir_backend.comp.properties import flash_state
    from reservoir_backend.comp.fluid import CompSpec

    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    spec = twin.physics.fluid
    assert spec is not None
    p = np.full(2, 5.0e7)
    z_oil = spec.z_init
    z_co2 = spec.z_inj
    n = np.vstack([z_oil, z_co2])
    props = flash_state(spec, p, n)
    assert props.sv[0] == pytest.approx(0.0)
    assert props.sv[1] == pytest.approx(1.0)


def test_lbc_init_oil_viscosity_is_physical() -> None:
    eos = load_eos_card("examples/lab_v1/cmg_gem/physical_3d/pvt_co2.yaml")
    z = np.array([0.0080, 0.0630, 0.0387, 0.3146, 0.4122, 0.1285, 0.0350])
    fl = flash_tp(eos, 5.0e7, 393.15, z)
    mu = lbc_viscosity(eos, 393.15, fl.x, np.array([1.0 / fl.v_liq]))
    assert mu.shape == (1,)
    assert 1.0e-5 < float(mu[0]) < 0.05


def test_flash_state_reflash_subset_does_not_broadcast_all_cells() -> None:
    from reservoir_backend.comp.properties import flash_state

    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    spec = twin.physics.fluid
    n = twin.grid.n_cells
    p = np.full(n, float(twin.physics.p_init))
    moles = np.broadcast_to(spec.z_init, (n, spec.nc)).copy()
    props = flash_state(spec, p, moles)
    cells = np.array([0, 17, 100], dtype=np.int64)
    flash_state(spec, p, moles, cells=cells, out=props)
    assert np.isfinite(props.lam_l[cells]).all()
    assert props.lam_l.shape == (n,)


def test_lbc_refuses_missing_vcrit() -> None:
    from reservoir_backend.eos.example import example_c1_nc10

    eos = example_c1_nc10()
    with pytest.raises(ValueError, match="vcrit"):
        lbc_viscosity(eos, 350.0, np.array([0.55, 0.45]), np.array([4000.0]))
