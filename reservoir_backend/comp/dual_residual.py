"""DPDP compositional residual. Two TPFA operators plus antisymmetric transfer.

    R_f = Δn_f + Δt ∇·F_f − Δt Q_f − Δt N_mf
    R_m = Δn_m + Δt ∇·F_m − Δt Q_m + Δt N_mf
    plus volume residuals on each continuum.

N_mf > 0 is matrix → fracture.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.properties import PhaseProps
from reservoir_backend.comp.residual import coupled_residual, pack_unknowns, unpack_unknowns
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer, TransferRates


def pack_dual(state: DualCompositionalState) -> NDArray[np.float64]:
    return np.concatenate(
        [
            pack_unknowns(state.fracture.moles, state.fracture.pressure),
            pack_unknowns(state.matrix.moles, state.matrix.pressure),
        ]
    )


def unpack_dual(u: NDArray[np.float64], n_cells: int, nc: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    half = n_cells * (nc + 1)
    nf, pf = unpack_unknowns(u[:half], n_cells, nc)
    nm, pm = unpack_unknowns(u[half:], n_cells, nc)
    return nf, pf, nm, pm


def transmissibilities(grid: CartesianGrid, rock_pair: DualRock):
    t_f = geometric_transmissibility(grid, rock_pair.fracture.permeability, kz=rock_pair.fracture.vertical_permeability())
    t_m = geometric_transmissibility(grid, rock_pair.matrix.permeability, kz=rock_pair.matrix.vertical_permeability())
    return t_f, t_m


def dual_residual(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    state: DualCompositionalState,
    state_old: DualCompositionalState,
    dt: float,
    transfer: ComponentTransfer,
    *,
    q_src_fracture: NDArray[np.float64] | None = None,
    q_src_matrix: NDArray[np.float64] | None = None,
    t_fracture: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = None,
    t_matrix: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None = None,
    props_fracture: PhaseProps | None = None,
    props_matrix: PhaseProps | None = None,
    reflash_fracture: NDArray[np.int64] | None = None,
    reflash_matrix: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], PhaseProps, PhaseProps, TransferRates]:
    n_cells = grid.n_cells
    nc = spec.nc
    zf = np.zeros((n_cells, nc)) if q_src_fracture is None else q_src_fracture
    zm = np.zeros((n_cells, nc)) if q_src_matrix is None else q_src_matrix
    if t_fracture is None or t_matrix is None:
        t_fracture, t_matrix = transmissibilities(grid, dual_rock)
    res_f, props_f = coupled_residual(
        grid,
        dual_rock.fracture,
        spec,
        state.fracture.moles,
        state.fracture.pressure,
        state_old.fracture.moles,
        dt,
        zf,
        t_fracture,
        props=props_fracture,
        reflash=reflash_fracture,
    )
    res_m, props_m = coupled_residual(
        grid,
        dual_rock.matrix,
        spec,
        state.matrix.moles,
        state.matrix.pressure,
        state_old.matrix.moles,
        dt,
        zm,
        t_matrix,
        props=props_matrix,
        reflash=reflash_matrix,
    )
    rates = transfer.compute(
        state.matrix.pressure,
        state.fracture.pressure,
        grid.cell_volumes(),
        props_m,
        props_f,
    )
    rf = res_f.reshape(n_cells, nc + 1)
    rm = res_m.reshape(n_cells, nc + 1)
    rf[:, :nc] -= float(dt) * rates.molar_rate
    rm[:, :nc] += float(dt) * rates.molar_rate
    return np.concatenate([rf.ravel(), rm.ravel()]), props_f, props_m, rates
