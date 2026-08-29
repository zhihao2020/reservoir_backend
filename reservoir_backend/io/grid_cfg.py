"""Cartesian or corner-point grid from case YAML / *GRID file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.exceptions import GridError
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.grid.corner_point import CornerPointGrid
from reservoir_backend.io.cpg_load import cpg_grid_from_cfg, merge_grid_sidecar

_VAR_KEYS = frozenset({"nx", "ny", "nz", "dx", "dy", "dz", "DX", "DY", "DZ"})


def _pick(cfg: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in cfg and cfg[name] is not None:
            return cfg[name]
    return None


def _as_int(name: str, value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise GridError("grid." + name + " must be a positive integer") from exc
    if n <= 0:
        raise GridError("grid." + name + " must be a positive integer")
    return n


def _is_cpg_cfg(grid_cfg: dict[str, Any]) -> bool:
    if _pick(grid_cfg, ("coord", "COORD")) is not None:
        return True
    if _pick(grid_cfg, ("zcorn", "ZCORN")) is not None:
        return True
    gtype = str(grid_cfg.get("type") or "").lower()
    return gtype in {"corner_point", "corner-point", "cpg", "corner"}


def _axis_spacing(
    name: str,
    raw: Any,
    count: int | None,
    size_m: float | None,
    fallback_spacing: float | None,
) -> tuple[int, Any]:
    if raw is not None:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            hx = float(arr)
            if not np.isfinite(hx) or hx <= 0.0:
                raise GridError("grid." + name + " must be positive and finite")
            if count is None:
                if size_m is None:
                    raise GridError(
                        "grid." + name + " scalar needs nx/ny/nz or geometry.size_m"
                    )
                count = max(1, int(round(float(size_m) / hx)))
            return count, np.full(count, hx, dtype=float)
        if arr.ndim != 1:
            raise GridError(
                "grid." + name + " must be a scalar or 1-D array "
                "(per-axis Cartesian, not per-cell 3-D)"
            )
        if count is None:
            count = int(arr.size)
        elif int(arr.size) != count:
            raise GridError(
                "grid." + name + " length " + str(int(arr.size)) + " != " + str(count)
            )
        return count, np.ascontiguousarray(arr, dtype=float)
    if count is not None and size_m is not None:
        return count, np.full(count, float(size_m) / float(count), dtype=float)
    if count is not None and fallback_spacing is not None:
        hx = float(fallback_spacing)
        if not np.isfinite(hx) or hx <= 0.0:
            raise GridError("grid.spacing_m must be positive and finite")
        return count, np.full(count, hx, dtype=float)
    if size_m is not None and fallback_spacing is not None:
        hx = float(fallback_spacing)
        if not np.isfinite(hx) or hx <= 0.0:
            raise GridError("grid.spacing_m must be positive and finite")
        n = max(1, int(round(float(size_m) / hx)))
        return n, np.full(n, float(size_m) / float(n), dtype=float)
    raise GridError(
        "grid needs " + name + " (scalar or list), or count+size_m, or spacing_m"
    )


def _spacing_components(
    grid_cfg: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    spacing = grid_cfg.get("spacing_m")
    if spacing is None:
        return None, None, None
    if isinstance(spacing, (int, float)):
        h = float(spacing)
        return h, h, h
    vals = tuple(float(x) for x in spacing)
    if len(vals) != 3:
        raise GridError("grid.spacing_m must be a scalar or length-3 list")
    return vals[0], vals[1], vals[2]


def _size_components(size: Any) -> tuple[float | None, float | None, float | None]:
    if size is None:
        return None, None, None
    vals = tuple(float(x) for x in size)
    if len(vals) != 3:
        raise GridError("geometry.size_m must have length 3")
    return vals[0], vals[1], vals[2]


def _has_variable_keys(grid_cfg: dict[str, Any]) -> bool:
    return any(k in grid_cfg and grid_cfg[k] is not None for k in _VAR_KEYS)


def _check_size(
    size: tuple[float | None, float | None, float | None], grid: CartesianGrid
) -> None:
    computed = grid.size_m()
    names = ("x", "y", "z")
    for i, given in enumerate(size):
        if given is None:
            continue
        got = computed[i]
        tol = max(1.0e-12, 1.0e-9 * max(abs(given), abs(got)))
        if abs(got - given) > tol:
            raise GridError(
                "geometry.size_m["
                + names[i]
                + "] "
                + str(given)
                + " != sum(d"
                + names[i]
                + ") "
                + str(got)
            )


def _apply_cartesian_actnum(grid: CartesianGrid, grid_cfg: dict[str, Any]) -> CartesianGrid:
    raw = _pick(grid_cfg, ("actnum", "ACTNUM"))
    if raw is None:
        return grid
    act = np.ascontiguousarray(np.asarray(raw, dtype=float).ravel())
    if act.size != grid.n_cells:
        raise GridError(
            "grid.actnum length " + str(int(act.size)) + " != " + str(grid.n_cells)
        )
    return CartesianGrid(
        nx=grid.nx,
        ny=grid.ny,
        nz=grid.nz,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        origin=grid.origin,
        active=act != 0.0,
    )


def grid_from_cfg(cfg: dict[str, Any], *, cfg_dir: str | Path = ".") -> CartesianGrid | CornerPointGrid:
    """Build a Cartesian or corner-point grid from a case mapping.

    Uniform path (lab defaults): ``geometry.size_m`` + ``grid.spacing_m``.
    Variable Cartesian: ``grid.nx`` / ``ny`` / ``nz`` and/or ``dx`` / ``dy`` /
    ``dz`` (aliases ``DX`` / ``DY`` / ``DZ``) as a scalar or a 1-D list along
    that axis. Optional ``grid.file`` relative to the case file directory:
    YAML/JSON sidecar, or a CMG/Eclipse ``*GRID`` snippet (``.grdecl`` /
    ``.dat``) with CART/CORNER, NX/NY/NZ, DX/DY/DZ (or DI/DJ/DK), COORD,
    ZCORN, ACTNUM. ``size_m`` must match the axis sums when both are given.
    COORD+ZCORN (Eclipse order) builds ``CornerPointGrid``; DX/DY/DZ without
    COORD/ZCORN builds ``CartesianGrid``. See ``grid.corner_point``.
    """
    geom = cfg.get("geometry") or {}
    origin_raw = geom.get("origin_m", [0.0, 0.0, 0.0])
    origin = tuple(float(x) for x in origin_raw)
    if len(origin) != 3:
        raise GridError("geometry.origin_m must have length 3")
    size_raw = geom.get("size_m")
    raw_grid = dict(cfg.get("grid") or {})
    grid_cfg = merge_grid_sidecar(raw_grid, Path(cfg_dir))
    if _is_cpg_cfg(grid_cfg):
        return cpg_grid_from_cfg(grid_cfg, origin, Path(cfg_dir))

    gtype = str(grid_cfg.get("type") or "cartesian").lower()
    if gtype not in {"cartesian", "cart", "xyz"}:
        raise GridError(
            "grid.type "
            + repr(gtype)
            + " is not supported (cartesian or corner_point/cpg)"
        )

    if not _has_variable_keys(grid_cfg):
        size = tuple(float(x) for x in (size_raw if size_raw is not None else [0.3, 0.3, 0.3]))
        if len(size) != 3:
            raise GridError("geometry.size_m must have length 3")
        spacing = grid_cfg.get("spacing_m", 0.01)
        if isinstance(spacing, (int, float)):
            grid = CartesianGrid.uniform(size, float(spacing), origin=origin)
        else:
            grid = CartesianGrid.uniform(size, tuple(float(x) for x in spacing), origin=origin)
        return _apply_cartesian_actnum(grid, grid_cfg)

    sx, sy, sz = _size_components(size_raw)
    hx, hy, hz = _spacing_components(grid_cfg)
    nx = _pick(grid_cfg, ("nx", "NX"))
    ny = _pick(grid_cfg, ("ny", "NY"))
    nz = _pick(grid_cfg, ("nz", "NZ"))
    nx_i = None if nx is None else _as_int("nx", nx)
    ny_i = None if ny is None else _as_int("ny", ny)
    nz_i = None if nz is None else _as_int("nz", nz)

    nx_i, dx = _axis_spacing("dx", _pick(grid_cfg, ("dx", "DX")), nx_i, sx, hx)
    ny_i, dy = _axis_spacing("dy", _pick(grid_cfg, ("dy", "DY")), ny_i, sy, hy)
    nz_i, dz = _axis_spacing("dz", _pick(grid_cfg, ("dz", "DZ")), nz_i, sz, hz)

    grid = CartesianGrid(nx=nx_i, ny=ny_i, nz=nz_i, dx=dx, dy=dy, dz=dz, origin=origin)
    _check_size((sx, sy, sz), grid)
    return _apply_cartesian_actnum(grid, grid_cfg)
