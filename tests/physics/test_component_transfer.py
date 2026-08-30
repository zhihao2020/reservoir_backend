import numpy as np
import pytest

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.physics.transfer import ComponentTransfer, WarrenRootTransfer


def _one_cell_props(p: float, z=None):
    spec = fluid_from_name("example", temperature_k=350.0)
    z = spec.z_init if z is None else np.asarray(z, dtype=float)
    moles = moles_from_z(spec, np.array([p]), z, np.array([1.0e-4]))
    props = flash_state(spec, np.array([p]), moles)
    return spec, moles, props


def test_transfer_zero_at_equilibrium() -> None:
    _, _, props = _one_cell_props(1.2e7)
    tr = ComponentTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15)
    rates = tr.compute(np.array([1.2e7]), np.array([1.2e7]), np.array([1.0e-4]), props, props)
    assert rates.molar_rate == pytest.approx(0.0, abs=1e-18)
    assert rates.phase_liquid_rate == pytest.approx(0.0, abs=1e-18)


def test_transfer_sign_reverses_with_pressure() -> None:
    _, _, props = _one_cell_props(1.2e7)
    tr = ComponentTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15)
    fwd = tr.compute(np.array([1.3e7]), np.array([1.1e7]), np.array([1.0e-4]), props, props)
    rev = tr.compute(np.array([1.1e7]), np.array([1.3e7]), np.array([1.0e-4]), props, props)
    assert float(np.sum(fwd.molar_rate)) > 0.0
    assert rev.molar_rate == pytest.approx(-fwd.molar_rate)


def test_transfer_uses_upstream_mobility() -> None:
    spec = fluid_from_name("example", temperature_k=350.0)
    p_hi, p_lo = 1.4e7, 1.0e7
    moles_hi = moles_from_z(spec, np.array([p_hi]), spec.z_init, np.array([1.0e-4]))
    moles_lo = moles_from_z(spec, np.array([p_lo]), spec.z_init, np.array([1.0e-4]))
    props_hi = flash_state(spec, np.array([p_hi]), moles_hi)
    props_lo = flash_state(spec, np.array([p_lo]), moles_lo)
    tr = ComponentTransfer(shape_factor=1.0, k_matrix_m2=1.0e-14)
    rates = tr.compute(np.array([p_hi]), np.array([p_lo]), np.array([1.0e-3]), props_hi, props_lo)
    dphi = p_hi - p_lo
    q_l_expected = 1.0e-3 * 1.0e-14 * float(props_hi.lam_l[0]) * dphi
    assert rates.phase_liquid_rate[0] == pytest.approx(q_l_expected, rel=1e-12)


def test_transfer_component_sum() -> None:
    _, _, props = _one_cell_props(1.2e7)
    tr = ComponentTransfer(shape_factor=2.0, k_matrix_m2=1.0e-15)
    rates = tr.compute(np.array([1.3e7]), np.array([1.1e7]), np.array([2.0e-4]), props, props)
    molar = rates.molar_rate[0]
    recon = (
        float(props.xi_l[0]) * props.x[0] * rates.phase_liquid_rate[0]
        + float(props.xi_v[0]) * props.y[0] * rates.phase_vapor_rate[0]
    )
    assert molar == pytest.approx(recon)


def test_transfer_matrix_fracture_antisymmetry() -> None:
    _, _, props = _one_cell_props(1.2e7)
    tr = ComponentTransfer(shape_factor=3.0, k_matrix_m2=2.0e-15)
    n_mf = tr.compute(np.array([1.25e7]), np.array([1.15e7]), np.array([1.0e-4]), props, props).molar_rate
    n_fm = tr.compute(np.array([1.15e7]), np.array([1.25e7]), np.array([1.0e-4]), props, props).molar_rate
    assert n_mf + n_fm == pytest.approx(0.0)


def test_transfer_units_regression() -> None:
    """q = σ k_m V λ Δp → m³/s when λ=1 (Pa⁻¹ s⁻¹ cancelled by Pa)."""
    wr = WarrenRootTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15)
    q = wr.compute_transfer(2.0e5, 1.0e5, 0.001)
    assert q == pytest.approx(4.0 * 1.0e-15 * 0.001 * 1.0e5)
    assert q > 0.0
    q_rev = wr.compute_transfer(1.0e5, 2.0e5, 0.001)
    assert q_rev == pytest.approx(-float(q))
    wr2 = WarrenRootTransfer(shape_factor=1.0, k_matrix_m2=1.0e-14)
    qv = wr2.compute_transfer(np.array([2.0, 3.0]), np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    assert qv.shape == (2,)
    assert qv[1] == pytest.approx(2.0 * qv[0])
