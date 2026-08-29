"""Coupled residuals: component moles + volume. Flash is an inner property."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.flux import molar_divergence
from reservoir_backend.comp.properties import PhaseProps, flash_state
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock


def pack_unknowns(moles: NDArray[np.float64], pressure: NDArray[np.float64]) -> NDArray[np.float64]:
    n_cells, nc = moles.shape
    u = np.zeros(n_cells * (nc + 1))
    u.reshape(n_cells, nc + 1)[:, :nc] = moles
    u.reshape(n_cells, nc + 1)[:, nc] = pressure
    return u


def unpack_unknowns(u: NDArray[np.float64], n_cells: int, nc: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    w = np.asarray(u, dtype=float).reshape(n_cells, nc + 1)
    return w[:, :nc].copy(), w[:, nc].copy()


def volume_residual(
    moles: NDArray[np.float64],
    props: PhaseProps,
    pore_volume: NDArray[np.float64],
    n_hc: int,
) -> NDArray[np.float64]:
    hc = np.sum(moles[:, :n_hc], axis=1)
    vol = hc * props.v_mix
    if props.has_water and moles.shape[1] > n_hc:
        vol = vol + moles[:, n_hc] * props.vw
    return vol - np.asarray(pore_volume, dtype=float).ravel()


def coupled_residual(
    grid: CartesianGrid,
    rock: Rock,
    spec: CompSpec,
    moles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    moles_old: NDArray[np.float64],
    dt: float,
    q_src: NDArray[np.float64],
    t_geom: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    *,
    props: PhaseProps | None = None,
    reflash: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], PhaseProps]:
    """Packed residual (n_cells * (nc+1),). Mass then volume per cell."""
    if props is None:
        props = flash_state(spec, pressure, moles)
    elif reflash is not None:
        flash_state(spec, pressure, moles, cells=reflash, out=props)
    pv = np.asarray(rock.porosity, dtype=float).ravel() * grid.cell_volumes()
    div = molar_divergence(grid, t_geom[0], t_geom[1], t_geom[2], pressure, props)
    mass = (moles - moles_old) + float(dt) * (div - q_src)
    vol = volume_residual(moles, props, pv, spec.n_hc)
    n_cells, nc = moles.shape
    res = np.zeros(n_cells * (nc + 1))
    block = res.reshape(n_cells, nc + 1)
    block[:, :nc] = mass
    block[:, nc] = vol
    return res, props
