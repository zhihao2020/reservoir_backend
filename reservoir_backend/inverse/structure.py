"""Discrete structure hypotheses. Hold-out picks the winner. Not geology from images."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import ContrastParameterization, RegionParameterization

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class StructureSpec:
    name: str
    kind: str
    n_regions: int
    region_axis: str = "z"
    flip: bool = False


CATALOG: dict[str, StructureSpec] = {
    "z1": StructureSpec("z1", "region", 1, "z"),
    "z2": StructureSpec("z2", "region", 2, "z"),
    "z3": StructureSpec("z3", "region", 3, "z"),
    "z2_contrast_top": StructureSpec("z2_contrast_top", "contrast", 2, "z", flip=False),
    "z2_contrast_bot": StructureSpec("z2_contrast_bot", "contrast", 2, "z", flip=True),
}

DEFAULT_NAMES = ("z1", "z2", "z3", "z2_contrast_top", "z2_contrast_bot")


def specs_from_names(names: list[str] | tuple[str, ...]) -> list[StructureSpec]:
    out = []
    for raw in names:
        name = str(raw).strip().lower()
        if name not in CATALOG:
            raise ValueError(f"unknown structure candidate {raw!r}; use {sorted(CATALOG)}")
        out.append(CATALOG[name])
    return out


def should_search_structure(
    *,
    has_region_map: bool,
    search_structure: bool | None,
    candidates: list[str] | None,
) -> bool:
    if has_region_map and search_structure is not True and not candidates:
        return False
    if candidates:
        return True
    return bool(search_structure)


def _axis_regions(grid: CartesianGrid, axis: str, n_regions: int) -> np.ndarray:
    if axis not in _AXIS:
        raise ValueError(f"region axis must be x, y, or z, got {axis!r}")
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    coord = grid.cell_centers()[:, _AXIS[axis]]
    rid = np.zeros(grid.n_cells, dtype=np.int64)
    if n_regions > 1:
        edges = np.quantile(coord, np.linspace(0.0, 1.0, n_regions + 1)[1:-1])
        for i, c in enumerate(np.sort(edges), start=1):
            rid[coord >= float(c)] = i
    return rid


def parameterization_for(grid: CartesianGrid, spec: StructureSpec, *, phi: float):
    rid = _axis_regions(grid, spec.region_axis, spec.n_regions)
    if spec.kind == "contrast":
        if spec.flip:
            rid = (rid == 0).astype(np.int64)
        return ContrastParameterization(rid, phi=float(phi))
    return RegionParameterization(rid, phi=float(phi))


def run_structure_search(
    twin,
    *,
    specs: list[StructureSpec] | None = None,
    time_limit_s: float | None = None,
) -> tuple[object, list[dict]]:
    """Calibrate each hypothesis. Keep the lowest hold-out. Leaves twin.parameterization on the winner."""
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    if specs is None:
        names = twin.inverse.structure_candidates or list(DEFAULT_NAMES)
        specs = specs_from_names(names)
    original = twin.parameterization
    rows: list[dict] = []
    best = None
    winner_param = original
    winner_name = "original"
    t0 = time.perf_counter()
    for i, spec in enumerate(specs):
        if time_limit_s is not None and time.perf_counter() - t0 >= float(time_limit_s):
            rows.append({"name": spec.name, "skipped": True, "reason": "time_limit", "selected": False})
            continue
        twin.parameterization = parameterization_for(twin.grid, spec, phi=phi)
        post = twin._calibrate_candidate(seed=int(twin.inverse.seed) + 17 + i)
        score = float(post.holdout_rmse)
        selected = best is None or (np.isfinite(score) and score < float(best.holdout_rmse))
        if selected:
            if best is not None:
                for prev in rows:
                    prev["selected"] = False
            best = post
            winner_param = twin.parameterization
            winner_name = spec.name
        rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "n_regions": spec.n_regions,
                "n_theta": int(twin.parameterization.n_params),
                "holdout_rmse": score,
                "assimilate_rmse": float(post.assimilate_rmse),
                "selected": selected,
            }
        )
    twin.parameterization = winner_param
    if best is None:
        twin.parameterization = original
        raise ValueError("structure search produced no posteriors")
    best.notes = list(best.notes) + [f"structure={winner_name}"]
    return best, rows
