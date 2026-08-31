"""Build the Gate 0 flash regression corpus from a p–z grid plus a tiny DPDP step."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.eos.flash_backend import get_flash_backend


def corpus_path() -> Path:
    return Path(__file__).resolve().parents[1].parent / "tests" / "fixtures" / "flash_states.npz"


def realfluid_corpus_path() -> Path:
    return Path(__file__).resolve().parents[1].parent / "tests" / "fixtures" / "flash_states_realfluid.npz"


def _grid_states(eos, temperature: float, n_p: int = 12, n_z: int = 12):
    p = np.geomspace(2.0e6, 3.0e7, n_p)
    z1 = np.linspace(0.08, 0.92, n_z)
    pp, zz = np.meshgrid(p, z1, indexing="ij")
    z = np.stack((zz.ravel(), 1.0 - zz.ravel()), axis=1)
    return pp.ravel(), z


def _dpdp_states():
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import moles_from_z
    from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
    from reservoir_backend.grid.cartesian import CartesianGrid
    from reservoir_backend.physics.dual_rock import DualRock
    from reservoir_backend.physics.transfer import ComponentTransfer
    from reservoir_backend.solver.fi_comp_dual import solve_dual_comp_step

    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    vol = grid.cell_volumes()
    p_f = np.array([1.05e7])
    p_m = np.array([1.25e7])
    n_f = moles_from_z(spec, p_f, spec.z_init, dual.fracture.porosity * vol)
    n_m = moles_from_z(spec, p_m, spec.z_init, dual.matrix.porosity * vol)
    state = DualCompositionalState(
        fracture=CompositionalContinuumState(p_f, n_f),
        matrix=CompositionalContinuumState(p_m, n_m),
        time_s=0.0,
    )
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    nxt = solve_dual_comp_step(grid, dual, spec, state, 2.0, transfer)
    st = nxt.state
    p = np.concatenate([st.fracture.pressure, st.matrix.pressure])
    n = np.concatenate([st.fracture.moles, st.matrix.moles], axis=0)
    tot = np.maximum(n.sum(axis=1, keepdims=True), 1.0e-18)
    z = n / tot
    return p, z[:, : spec.n_hc]


def build_flash_corpus(path: Path | None = None, *, temperature: float = 350.0) -> Path:
    eos = example_c1_nc10()
    p_g, z_g = _grid_states(eos, temperature)
    p_d, z_d = _dpdp_states()
    p = np.concatenate([p_g, p_d])
    z = np.concatenate([z_g, z_d], axis=0)
    backend = get_flash_backend()
    arr = backend.evaluate_batch(eos, p, temperature, z)
    ref = [flash_tp(eos, float(p[i]), temperature, z[i]) for i in range(p.size)]
    xi_l = 1.0 / np.maximum(arr.v_liq, 1.0e-12)
    xi_v = 1.0 / np.maximum(arr.v_vap, 1.0e-12)
    v_mix = arr.v_mix
    sl = (1.0 - arr.vapor_frac) * arr.v_liq / np.maximum(v_mix, 1.0e-30)
    sv = 1.0 - sl
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import _corey_og

    spec = fluid_from_name("example", temperature_k=float(temperature))
    kro, krg = _corey_og(sv, spec)
    dest = path or corpus_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        pressure=p,
        composition=z,
        temperature=np.array([temperature]),
        two_phase=arr.two_phase,
        k=arr.k,
        x=arr.x,
        y=arr.y,
        vapor_frac=arr.vapor_frac,
        v_mix=v_mix,
        v_liq=arr.v_liq,
        v_vap=arr.v_vap,
        xi_l=xi_l,
        xi_v=xi_v,
        sl=sl,
        sv=sv,
        lam_l=kro / spec.mu_liquid,
        lam_v=krg / spec.mu_vapor,
        ref_vapor_frac=np.array([r.vapor_frac for r in ref]),
        ref_two_phase=np.array([r.two_phase for r in ref], dtype=bool),
    )
    return dest


def build_realfluid_corpus(path: Path | None = None, *, card: Path | None = None, temperature: float = 350.0) -> Path:
    """Flash regression on the lab_v1 EOS card (C1–nC10 or lumped 3–6)."""
    from reservoir_backend.io.eos_load import load_eos_card

    default_card = Path(__file__).resolve().parents[2] / "examples" / "lab_v1" / "pvt.yaml"
    eos = load_eos_card(card or default_card)
    p = np.concatenate(
        [
            np.geomspace(2.0e6, 3.0e7, 8),
            np.linspace(1.0e7, 1.4e7, 6),
        ]
    )
    if eos.nc == 2:
        z1 = np.linspace(0.10, 0.90, 8)
        pp, zz = np.meshgrid(p, z1, indexing="ij")
        z = np.stack((zz.ravel(), 1.0 - zz.ravel()), axis=1)
        p = pp.ravel()
    else:
        rng = np.random.default_rng(0)
        z = rng.random((p.size * 4, eos.nc))
        z = z / z.sum(axis=1, keepdims=True)
        p = np.repeat(p, 4)
    backend = get_flash_backend()
    arr = backend.evaluate_batch(eos, p, temperature, z)
    ref = [flash_tp(eos, float(p[i]), temperature, z[i]) for i in range(p.size)]
    dest = path or realfluid_corpus_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        pressure=p,
        composition=z,
        temperature=np.array([temperature]),
        two_phase=arr.two_phase,
        k=arr.k,
        vapor_frac=arr.vapor_frac,
        ref_vapor_frac=np.array([r.vapor_frac for r in ref]),
        ref_two_phase=np.array([r.two_phase for r in ref], dtype=bool),
        names=np.array(eos.names),
    )
    return dest
