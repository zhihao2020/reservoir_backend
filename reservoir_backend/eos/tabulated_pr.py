"""Isothermal binary (p, z_C1) thermo table. Fallback exact flash on the envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.flash import FlashResult
from reservoir_backend.eos.flash_batch import FlashArrays, flash_arrays_to_result, flash_batch
from reservoir_backend.eos.pr import PengRobinson


@dataclass
class ThermoTable:
    temperature_k: float
    p: NDArray[np.float64]
    z1: NDArray[np.float64]
    vapor_frac: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    v_liq: NDArray[np.float64]
    v_vap: NDArray[np.float64]
    two_phase: NDArray[np.bool_]
    k: NDArray[np.float64]
    z_liq: NDArray[np.float64]
    z_vap: NDArray[np.float64]

    @classmethod
    def build(
        cls,
        eos: PengRobinson,
        temperature: float,
        *,
        p_lo: float = 1.0e6,
        p_hi: float = 4.0e7,
        n_p: int = 48,
        n_z: int = 48,
    ) -> ThermoTable:
        if int(eos.nc) != 2:
            raise ValueError("V1 table is binary (Nc=2) only")
        p = np.geomspace(float(p_lo), float(p_hi), int(n_p))
        z1 = np.linspace(0.02, 0.98, int(n_z))
        pp, zz = np.meshgrid(p, z1, indexing="ij")
        z = np.stack((zz.ravel(), 1.0 - zz.ravel()), axis=1)
        arr = flash_batch(eos, pp.ravel(), float(temperature), z)
        shape = (p.size, z1.size)
        nc = 2
        return cls(
            temperature_k=float(temperature),
            p=p,
            z1=z1,
            vapor_frac=arr.vapor_frac.reshape(shape),
            x=arr.x.reshape(shape + (nc,)),
            y=arr.y.reshape(shape + (nc,)),
            v_liq=arr.v_liq.reshape(shape),
            v_vap=arr.v_vap.reshape(shape),
            two_phase=arr.two_phase.reshape(shape),
            k=arr.k.reshape(shape + (nc,)),
            z_liq=arr.z_liq.reshape(shape),
            z_vap=arr.z_vap.reshape(shape),
        )


def _ensure_table(backend, eos: PengRobinson, temperature: float) -> ThermoTable:
    tab = getattr(backend, "_table", None)
    t = float(temperature)
    if tab is None or abs(float(tab.temperature_k) - t) > 1.0e-9:
        tab = ThermoTable.build(eos, t)
        backend._table = tab
    return tab


def _weights(grid: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    i = np.searchsorted(grid, x, side="right") - 1
    i = np.clip(i, 0, grid.size - 2)
    j = i + 1
    dx = grid[j] - grid[i]
    w = np.divide(x - grid[i], dx, out=np.zeros_like(x), where=np.abs(dx) > 1.0e-30)
    return i.astype(np.int64), j.astype(np.int64), np.clip(w, 0.0, 1.0)


def flash_tabulated_batch(
    backend,
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    **kwargs,
) -> FlashArrays:
    if int(eos.nc) != 2:
        return flash_batch(eos, pressure, temperature, z, **kwargs)
    tab = _ensure_table(backend, eos, temperature)
    p = np.asarray(pressure, dtype=float).ravel()
    zz = np.asarray(z, dtype=float)
    if zz.ndim == 1:
        zz = np.broadcast_to(zz, (p.size, zz.size)).copy()
    z1 = zz[:, 0] / np.maximum(zz.sum(axis=1), 1.0e-30)
    outside = (p < tab.p[0]) | (p > tab.p[-1]) | (z1 < tab.z1[0]) | (z1 > tab.z1[-1])
    ip, jp, wp = _weights(tab.p, np.clip(p, tab.p[0], tab.p[-1]))
    iz, jz, wz = _weights(tab.z1, np.clip(z1, tab.z1[0], tab.z1[-1]))
    corners = (
        tab.two_phase[ip, iz],
        tab.two_phase[ip, jz],
        tab.two_phase[jp, iz],
        tab.two_phase[jp, jz],
    )
    mixed = (corners[0] != corners[1]) | (corners[0] != corners[2]) | (corners[0] != corners[3])
    need_exact = outside | mixed
    w00 = (1.0 - wp) * (1.0 - wz)
    w01 = (1.0 - wp) * wz
    w10 = wp * (1.0 - wz)
    w11 = wp * wz

    def blend(field):
        return w00 * field[ip, iz] + w01 * field[ip, jz] + w10 * field[jp, iz] + w11 * field[jp, jz]

    def blend2(field):
        return (
            w00[:, None] * field[ip, iz]
            + w01[:, None] * field[ip, jz]
            + w10[:, None] * field[jp, iz]
            + w11[:, None] * field[jp, jz]
        )

    n = p.size
    nc = 2
    arr = FlashArrays(
        vapor_frac=blend(tab.vapor_frac),
        x=blend2(tab.x),
        y=blend2(tab.y),
        z_liq=blend(tab.z_liq),
        z_vap=blend(tab.z_vap),
        v_liq=blend(tab.v_liq),
        v_vap=blend(tab.v_vap),
        two_phase=corners[0].copy(),
        k=blend2(tab.k),
        converged=np.ones(n, dtype=bool),
        iterations=np.zeros(n, dtype=np.int32),
        fugacity_error=np.zeros(n),
        stability_checked=np.zeros(n, dtype=bool),
        stability_margin=np.zeros(n),
        fallback_used=need_exact.copy(),
    )
    if np.any(need_exact):
        idx = np.where(need_exact)[0]
        exact = flash_batch(eos, p[idx], float(temperature), zz[idx], **kwargs)
        arr.vapor_frac[idx] = exact.vapor_frac
        arr.x[idx] = exact.x
        arr.y[idx] = exact.y
        arr.z_liq[idx] = exact.z_liq
        arr.z_vap[idx] = exact.z_vap
        arr.v_liq[idx] = exact.v_liq
        arr.v_vap[idx] = exact.v_vap
        arr.two_phase[idx] = exact.two_phase
        arr.k[idx] = exact.k
        arr.converged[idx] = exact.converged
        arr.iterations[idx] = exact.iterations
        arr.fugacity_error[idx] = exact.fugacity_error
        arr.stability_checked[idx] = exact.stability_checked
        arr.stability_margin[idx] = exact.stability_margin
    return arr


def flash_tabulated_one(backend, eos: PengRobinson, pressure: float, temperature: float, z, **kwargs) -> FlashResult:
    arr = flash_tabulated_batch(backend, eos, np.array([float(pressure)]), temperature, np.asarray(z, dtype=float).reshape(1, -1), **kwargs)
    return flash_arrays_to_result(arr, 0)
