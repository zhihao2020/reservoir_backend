"""Fixed-kind parameterization from a case YAML. Unknown kinds error.

"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
from reservoir_backend.inverse.parameterization import (
    ContrastParameterization,
    RegionParameterization,
)
from reservoir_backend.physics.conductivity import FractureConductivityModel

_AXIS = {"x": 0, "y": 1, "z": 2}


def region_ids(grid: CartesianGrid, inv: dict[str, Any], cfg_dir: Path) -> np.ndarray:
    raw_map = inv.get("region_map")
    if raw_map:
        mp = Path(str(raw_map))
        if not mp.is_file():
            mp = Path(cfg_dir) / mp
        if not mp.is_file():
            raise FileNotFoundError(f"inverse.region_map not found: {raw_map}")
        if mp.suffix.lower() == ".npy":
            rid = np.load(mp)
        else:
            rid = np.loadtxt(mp, dtype=np.int64, delimiter=",")
        rid = np.asarray(rid, dtype=np.int64).ravel()
        if rid.size != grid.n_cells:
            raise ValueError(f"region_map length {rid.size} != n_cells {grid.n_cells}")
        if np.any(rid < 0):
            raise ValueError("region ids must be >= 0")
        return rid

    axis_name = str(inv.get("region_axis", "z")).lower()
    if axis_name not in _AXIS:
        raise ValueError(f"inverse.region_axis must be x, y, or z, got {axis_name!r}")
    coord = grid.cell_centers()[:, _AXIS[axis_name]]
    nreg = int(inv.get("n_regions", 2))
    if nreg < 1:
        raise ValueError("inverse.n_regions must be >= 1")
    cuts = inv.get("region_cuts")
    if cuts is None:
        edges = np.quantile(coord, np.linspace(0.0, 1.0, nreg + 1)[1:-1]) if nreg > 1 else np.asarray([])
    else:
        edges = np.asarray(cuts, dtype=float).ravel()
        lo, hi = float(coord.min()), float(coord.max())
        span = max(hi - lo, 1.0e-30)
        if edges.size and float(np.max(edges)) <= 1.0 and float(np.min(edges)) >= 0.0:
            edges = lo + edges * span
    rid = np.zeros(grid.n_cells, dtype=np.int64)
    for i, c in enumerate(np.sort(edges), start=1):
        rid[coord >= float(c)] = i
    return rid


def parameterization_from_cfg(grid: CartesianGrid, cfg: dict[str, Any], cfg_dir: Path):
    inv = cfg.get("inverse") or {}
    kind = str(inv.get("parameterization", "region")).lower()
    phi = float((cfg.get("rock") or {}).get("porosity", 0.20))
    if kind in {"region", "contrast"}:
        rid = region_ids(grid, inv, Path(cfg_dir))
        if kind == "contrast":
            high = inv.get("high_region")
            if high is not None:
                rid = (np.asarray(rid) == int(high)).astype(np.int64)
            return ContrastParameterization(
                rid,
                phi=phi,
                log_contrast_mean=float(inv.get("log_contrast_mean", float(np.log(20.0)))),
                log_contrast_std=float(inv.get("log_contrast_std", 1.0)),
            )
        return RegionParameterization(rid, phi=phi)
    if kind in {"log_conductivity", "cf", "scalar_cf", "fracture_conductivity"}:
        inv_cf = dict(inv)
        if not inv.get("region_map") and inv.get("n_regions") is None:
            inv_cf["n_regions"] = 1
        rid = region_ids(grid, inv_cf, Path(cfg_dir))
        if int(rid.max()) <= 0:
            mask = np.ones(grid.n_cells, dtype=bool)
        else:
            high = int(inv.get("fracture_region", rid.max()))
            mask = np.asarray(rid, dtype=np.int64).ravel() == high
        km = float(inv.get("k_matrix_m2", (cfg.get("rock") or {}).get("k_matrix_m2", 1.0e-15)))
        cond = FractureConductivityModel(n_cells=grid.n_cells, fracture_mask=mask, k_matrix_m2=km)
        pm = inv.get("prior_mean", np.log(1.0e-13))
        ps = inv.get("prior_std", 1.0)
        if isinstance(pm, list):
            pm = float(np.asarray(pm, dtype=float).ravel()[0])
        if isinstance(ps, list):
            ps = float(np.asarray(ps, dtype=float).ravel()[0])
        return LogConductivityParameterization(
            n_zones=1,
            phi=phi,
            conductivity=cond,
            prior_mean=float(pm),
            prior_std=float(ps),
        )
    raise ValueError(
        f"unknown inverse.parameterization {kind!r}; use region, contrast, or log_conductivity"
    )
