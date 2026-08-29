"""Unit tests for CMOST-style fracture-strip parameterization."""

from __future__ import annotations

import numpy as np

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.frac import (
    FractureStripParameterization,
    WellTrack,
    decode_frac_theta,
    paint_fracture_strips,
)


def _grid() -> CartesianGrid:
    return CartesianGrid(
        nx=12,
        ny=10,
        nz=5,
        dx=np.full(12, 50.0),
        dy=np.full(10, 50.0),
        dz=np.full(5, 10.0),
    )


def test_expand_shape_and_fractions() -> None:
    grid = _grid()
    well = WellTrack("HW1", j=4, k=2, i0=2, i1=9)
    param = FractureStripParameterization(
        grid, (well,), free_geometry=True, prior_mean=np.zeros(6), prior_std=np.ones(6)
    )
    theta = np.array([np.log(1e-18), np.log(1e-12), np.log(1e-15), np.log(60.0), 4.0, 0.0])
    k = param.expand(theta)
    assert k.shape == (grid.n_cells,)
    _, frac, srv = paint_fracture_strips(
        grid,
        (well,),
        log_k_m=float(theta[0]),
        log_k_f=float(theta[1]),
        log_k_srv=float(theta[2]),
        x_f_m=60.0,
        n_frac=4,
        frac_phase=0.0,
    )
    assert int(np.sum(frac)) > 0
    assert int(np.sum(srv)) > int(np.sum(frac))
    assert float(np.max(k[frac])) > float(np.max(k[~frac & ~srv]))


def test_project_clips_n_frac_and_phase() -> None:
    grid = _grid()
    well = WellTrack("HW1", j=4, k=2, i0=2, i1=9)
    param = FractureStripParameterization(grid, (well,), n_frac_max=6, free_geometry=True)
    th = param.project(np.array([0.0, 0.0, 0.0, np.log(500.0), 9.5, 1.25]))
    assert th[4] == 6.0
    assert th[5] == 0.25


def test_default_is_four_params_frozen_geometry() -> None:
    grid = _grid()
    well = WellTrack("HW1", j=4, k=2, i0=2, i1=9)
    param = FractureStripParameterization(
        grid, (well,), prior_mean=np.zeros(4), prior_std=np.ones(4), fixed_n_frac=4.0
    )
    assert param.n_params == 4
    k = param.expand(np.array([np.log(1e-18), np.log(1e-12), np.log(1e-15), np.log(60.0)]))
    assert k.shape == (grid.n_cells,)
    eng = decode_frac_theta(param, np.array([np.log(1e-18), np.log(1e-12), np.log(1e-15), np.log(40.0)]))
    assert eng["n_frac"] == 4.0
    assert eng["free_geometry"] is False


def test_decode_frac_theta_keys() -> None:
    grid = _grid()
    well = WellTrack("HW1", j=4, k=2, i0=2, i1=9)
    param = FractureStripParameterization(
        grid, (well,), frac_aperture_m=50.0, free_geometry=True, prior_mean=np.zeros(6), prior_std=np.ones(6)
    )
    theta = np.array([np.log(1e-18), np.log(8e-12), np.log(4e-16), np.log(40.0), 5.0, 0.0])
    eng = decode_frac_theta(param, theta)
    assert eng["n_frac"] == 5.0
    assert eng["x_f_m"] == 40.0
    assert eng["F_cd_m3"] > 0.0
    assert eng["k_frac_over_matrix"] > 1.0e3
