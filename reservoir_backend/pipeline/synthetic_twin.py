"""Synthetic buried-channel (mountain-like) twin for algorithm validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.state import (
    AxisAlignedBounds,
    BoundaryConditions,
    MeshBundle,
    SensorSample,
    WellPoint,
)
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d


@dataclass
class SyntheticTwin:
    """Known truth fields and well-sampled sensor series."""

    mesh: MeshBundle
    true_k: NDArray[np.float64]
    true_phi: NDArray[np.float64]
    true_channel_mask: NDArray[np.bool_]
    pressure_series: list[NDArray[np.float64]]
    sw_series: list[NDArray[np.float64]]
    samples: list[SensorSample]
    times: list[float]
    true_fault_mask: NDArray[np.bool_] | None = None


def build_channel_twin(
    *,
    nx: int = 12,
    ny: int = 10,
    nz: int = 4,
    lx: float = 120.0,
    ly: float = 100.0,
    lz: float = 40.0,
    k_background: float = 5.0e-14,
    k_channel: float = 5.0e-13,
    phi: float = 0.22,
    n_times: int = 4,
) -> SyntheticTwin:
    """Create a box with a raised high-k channel (mountain-like body) between wells.

    Forward pressure uses TPFA with true k. Saturation is a smooth front that
    advances along the channel over time (proxy for displacement footprint).
    """
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    wells = [
        WellPoint("INJ", 0.15 * lx, 0.5 * ly, 0.5 * lz),
        WellPoint("PROD", 0.85 * lx, 0.5 * ly, 0.5 * lz),
    ]
    mesh = build_mesh(bounds, dx, dy, dz, wells=wells)
    grid = mesh.grid

    # Mountain / channel: undulating ridge between wells (not a flat layer).
    # Centerline y and z both vary with along-path coordinate t.
    channel = np.zeros(grid.shape, dtype=bool)
    for n in range(mesh.n_cells):
        x, y, z = mesh.x[n], mesh.y[n], mesh.z[n]
        t = (x - 0.15 * lx) / (0.7 * lx)
        if not (0.0 <= t <= 1.0):
            continue
        # plan-view meander + structural crest (mountain) + short-wavelength ripple
        y_c = 0.5 * ly + 0.08 * ly * np.sin(2.0 * np.pi * t)
        z_c = (
            0.32 * lz
            + 0.28 * lz * np.exp(-((t - 0.5) ** 2) / 0.07)
            + 0.05 * lz * np.sin(4.0 * np.pi * t)
        )
        half_w = 0.11 * ly * (1.0 + 0.25 * np.sin(np.pi * t))
        half_h = 0.16 * lz * (1.0 + 0.35 * np.exp(-((t - 0.5) ** 2) / 0.1))
        if abs(y - y_c) < half_w and abs(z - z_c) < half_h:
            i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
            channel[k, j, i] = True

    true_k = np.full(grid.shape, k_background, dtype=float)
    true_k[channel] = k_channel
    true_phi = np.full(grid.shape, phi, dtype=float)

    # multi-time pressure: vary boundary slightly + well pressures
    times = [float(i) * 30.0 for i in range(n_times)]  # days-like labels
    pressure_series: list[NDArray[np.float64]] = []
    sw_series: list[NDArray[np.float64]] = []
    samples: list[SensorSample] = []

    for ti, t in enumerate(times):
        p_inj = 12.0e6 + 0.2e6 * ti
        p_prod = 10.0e6 - 0.1e6 * ti
        result = solve_steady_state_pressure_3d(
            grid,
            true_k,
            true_k,
            true_k,
            mu=1.0e-3,
            dirichlet_boundaries={"left": p_inj, "right": p_prod},
            reference_pressure=p_prod,
        )
        p = result.pressure.values.copy()
        # pin wells to prescribed sensor values (measurement model)
        for name, val in (("INJ", p_inj), ("PROD", p_prod)):
            c = mesh.well_cell_id[name]
            i, j, k = grid.ijk(c)
            p[k, j, i] = val
        pressure_series.append(p)

        # saturation front along channel: high sw near injector, grows with time
        sw = np.full(grid.shape, 0.25, dtype=float)
        progress = 0.2 + 0.7 * (ti + 1) / n_times
        for n in range(mesh.n_cells):
            i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
            if not channel[k, j, i]:
                continue
            x = mesh.x[n]
            s = (x - 0.15 * lx) / (0.7 * lx)
            if s <= progress:
                sw[k, j, i] = 0.75 - 0.2 * s
            else:
                sw[k, j, i] = 0.30
        # well saturations
        for name, sval in (("INJ", 0.80), ("PROD", 0.28 + 0.05 * ti)):
            c = mesh.well_cell_id[name]
            i, j, k = grid.ijk(c)
            sw[k, j, i] = min(0.9, sval)
        sw_series.append(sw)

        pi, pj, pk = grid.ijk(mesh.well_cell_id["PROD"])
        swp = float(sw[pk, pj, pi])
        # synthetic rates (m3/s): injection / production for transport sources
        q_inj = 2.0e-5 * (1.0 + 0.05 * ti)
        q_prod = -2.0e-5 * (1.0 + 0.05 * ti)
        samples.append(
            SensorSample(
                time=t,
                well_pressure={"INJ": p_inj, "PROD": p_prod},
                well_saturation={
                    "INJ": (0.80, 0.20, 0.0),
                    "PROD": (swp, 1.0 - swp, 0.0),
                },
                boundary=BoundaryConditions(pressure={"left": p_inj, "right": p_prod}),
                well_rate={"INJ": q_inj, "PROD": q_prod},
            )
        )

    return SyntheticTwin(
        mesh=mesh,
        true_k=true_k,
        true_phi=true_phi,
        true_channel_mask=channel,
        pressure_series=pressure_series,
        sw_series=sw_series,
        samples=samples,
        times=times,
    )


def build_faulted_channel_twin(
    *,
    nx: int = 12,
    ny: int = 10,
    nz: int = 4,
    lx: float = 120.0,
    ly: float = 100.0,
    lz: float = 40.0,
    k_background: float = 5.0e-14,
    k_channel: float = 5.0e-13,
    k_fault: float = 1.0e-18,
    phi: float = 0.22,
    n_times: int = 4,
    fault_i_frac: float = 0.5,
) -> SyntheticTwin:
    """Channel twin cut by a mid-domain sealing fault with a dogleg window.

    Fault is approximated as a low-k vertical plane of cells (TPFA has no
    face-multiplier API yet). The high-k channel is offset across the fault
    and only connects through a high-j window — matching the CMG fault case.
    """
    bounds = AxisAlignedBounds(0.0, lx, 0.0, ly, 0.0, lz)
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    fault_x = float(fault_i_frac) * lx
    wells = [
        WellPoint("INJ", 0.12 * lx, 0.45 * ly, 0.55 * lz),
        WellPoint("PROD", 0.88 * lx, 0.75 * ly, 0.55 * lz),
    ]
    mesh = build_mesh(bounds, dx, dy, dz, wells=wells)
    grid = mesh.grid

    fault = np.zeros(grid.shape, dtype=bool)
    channel = np.zeros(grid.shape, dtype=bool)
    for n in range(mesh.n_cells):
        x, y, z = mesh.x[n], mesh.y[n], mesh.z[n]
        i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
        # sealing plane near fault_x (one-cell thick)
        if abs(x - fault_x) <= 0.55 * dx:
            # leave a window at high y for connectivity
            if y < 0.65 * ly:
                fault[k, j, i] = True

        # dogleg channel: left y~0.45, right y~0.75, connect at window
        if x <= fault_x:
            y_c = 0.45 * ly
            z_c = 0.45 * lz + 0.12 * lz * np.sin(2.0 * np.pi * x / lx)
            on = abs(y - y_c) < 0.12 * ly and abs(z - z_c) < 0.18 * lz
        else:
            y_c = 0.75 * ly
            z_c = 0.50 * lz + 0.10 * lz * np.cos(2.0 * np.pi * x / lx)
            on = abs(y - y_c) < 0.12 * ly and abs(z - z_c) < 0.18 * lz
        # window bridge near fault
        if abs(x - fault_x) < 1.2 * dx and y > 0.62 * ly:
            on = abs(z - 0.5 * lz) < 0.20 * lz
        if on and not fault[k, j, i]:
            channel[k, j, i] = True

    true_k = np.full(grid.shape, k_background, dtype=float)
    true_k[channel] = k_channel
    true_k[fault] = k_fault
    true_phi = np.full(grid.shape, phi, dtype=float)

    times = [float(i) * 30.0 for i in range(n_times)]
    pressure_series: list[NDArray[np.float64]] = []
    sw_series: list[NDArray[np.float64]] = []
    samples: list[SensorSample] = []

    for ti, t in enumerate(times):
        p_inj = 12.0e6 + 0.15e6 * ti
        p_prod = 10.0e6 - 0.08e6 * ti
        result = solve_steady_state_pressure_3d(
            grid,
            true_k,
            true_k,
            true_k,
            mu=1.0e-3,
            dirichlet_boundaries={"left": p_inj, "right": p_prod},
            reference_pressure=p_prod,
        )
        p = result.pressure.values.copy()
        for name, val in (("INJ", p_inj), ("PROD", p_prod)):
            c = mesh.well_cell_id[name]
            ii, jj, kk = grid.ijk(c)
            p[kk, jj, ii] = val
        pressure_series.append(p)

        sw = np.full(grid.shape, 0.25, dtype=float)
        progress = 0.15 + 0.75 * (ti + 1) / n_times
        for n in range(mesh.n_cells):
            i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
            if not channel[k, j, i]:
                continue
            # path coordinate along dogleg (normalized x with y detour weight)
            s = (mesh.x[n] / lx) * 0.7 + (mesh.y[n] / ly) * 0.3
            if s <= progress:
                sw[k, j, i] = 0.72 - 0.15 * s
            else:
                sw[k, j, i] = 0.30
        pi, pj, pk = grid.ijk(mesh.well_cell_id["PROD"])
        swp = float(sw[pk, pj, pi])
        ii, jj, kk = grid.ijk(mesh.well_cell_id["INJ"])
        sw[kk, jj, ii] = 0.80
        sw_series.append(sw)
        q_inj = 2.5e-5
        q_prod = -2.5e-5
        samples.append(
            SensorSample(
                time=t,
                well_pressure={"INJ": p_inj, "PROD": p_prod},
                well_saturation={
                    "INJ": (0.80, 0.20, 0.0),
                    "PROD": (swp, 1.0 - swp, 0.0),
                },
                boundary=BoundaryConditions(pressure={"left": p_inj, "right": p_prod}),
                well_rate={"INJ": q_inj, "PROD": q_prod},
            )
        )

    return SyntheticTwin(
        mesh=mesh,
        true_k=true_k,
        true_phi=true_phi,
        true_channel_mask=channel,
        pressure_series=pressure_series,
        sw_series=sw_series,
        samples=samples,
        times=times,
        true_fault_mask=fault,
    )


def mask_overlap(pred: NDArray[np.bool_], truth: NDArray[np.bool_]) -> dict[str, float]:
    """Dice / precision / recall between boolean masks."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    inter = float(np.sum(pred & truth))
    psum = float(np.sum(pred))
    tsum = float(np.sum(truth))
    dice = (2.0 * inter) / (psum + tsum + 1.0e-30)
    prec = inter / (psum + 1.0e-30)
    rec = inter / (tsum + 1.0e-30)
    return {"dice": dice, "precision": prec, "recall": rec, "intersection": inter}
