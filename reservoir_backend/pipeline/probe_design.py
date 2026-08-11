"""Probe / monitor placement: uniform grid and adaptive (DOE-like) design.

Product API for recommending exclusive ``observer_p`` / ``observer_s`` locations.
Does not modify CMG decks; validation scripts sample virtual readings from truth
fields at the recommended cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.esmda import generate_logk_ensemble
from reservoir_backend.pipeline.state import MeshBundle
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d

# Built-in design constants (not sensor-case YAML)
DEFAULT_HYBRID_ALPHA = 0.6
DEFAULT_ENSEMBLE_SIZE = 12
DEFAULT_CORR_LEN_CELLS = 3.0
EPS = 1.0e-30


@dataclass(frozen=True)
class ProbeSpec:
    """One recommended exclusive probe."""

    name: str
    x: float
    y: float
    z: float
    role: str  # observer_p | observer_s
    cell_id: int
    ijk: tuple[int, int, int] | None = None
    score: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_well_point_kwargs(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "role": self.role,
        }


def place_uniform_probes(
    mesh: MeshBundle,
    n_p: int,
    n_s: int,
    *,
    exclude_cell_ids: set[int] | None = None,
    seed: int = 0,
) -> list[ProbeSpec]:
    """Space-filling lattice-style placement (geometric DOE baseline).

    Builds a roughly uniform subset of cells (stratified on flattened index
    after sorting by space-filling curve-like order: z then y then x rank),
    then assigns first ``n_p`` to pressure and next ``n_s`` to saturation,
    skipping excluded cells (e.g. injectors/producers).
    """
    n_p = max(0, int(n_p))
    n_s = max(0, int(n_s))
    need = n_p + n_s
    if need == 0:
        return []

    exclude = set(exclude_cell_ids or ())
    # also exclude known active wells by role
    for name, cid in mesh.well_cell_id.items():
        role = mesh.well_role.get(name, "")
        if role in ("injector", "producer"):
            exclude.add(int(cid))

    order = _space_filling_order(mesh)
    candidates = [c for c in order if c not in exclude]
    if len(candidates) < need:
        # fall back: allow any non-excluded; if still short, take what we have
        pass
    selected = _pick_strided(candidates, need)
    specs: list[ProbeSpec] = []
    for rank, cid in enumerate(selected):
        role = "observer_p" if rank < n_p else "observer_s"
        name = f"P{rank + 1}" if role == "observer_p" else f"S{rank - n_p + 1}"
        specs.append(_spec_from_cell(mesh, name, role, int(cid), score=None, note="uniform"))
    return specs


def recommend_probes(
    mesh: MeshBundle,
    *,
    n_p: int,
    n_s: int,
    mode: str = "hybrid",
    exclude_cell_ids: set[int] | None = None,
    prior_k_ensemble: NDArray[np.float64] | None = None,
    prior_var_p: NDArray[np.float64] | None = None,
    prior_var_s: NDArray[np.float64] | None = None,
    viscosity_pa_s: float = 1.0e-3,
    p_left: float = 12.0e6,
    p_right: float = 10.0e6,
    k_mean: float = 1.0e-13,
    seed: int = 0,
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[ProbeSpec]:
    """Adaptive DOE-like sequential design for exclusive p/S probes.

    Modes
    -----
    - ``maximin``: maximize minimum distance to existing wells/probes
    - ``variance``: maximize prior variance of p (or s)
    - ``hybrid``: α·var_norm + (1−α)·dmin_norm (default α=0.6)

    If ``prior_var_p`` / ``prior_var_s`` are provided (e.g. multi-time CMG
    variance), they are used directly; otherwise p-variance is estimated from
    a small log-k ensemble + steady TPFA, and s-variance falls back to maximin
    only (or ``prior_var_s`` if given).
    """
    n_p = max(0, int(n_p))
    n_s = max(0, int(n_s))
    mode = str(mode).lower().strip()
    if mode not in ("maximin", "variance", "hybrid"):
        raise ValueError(f"unsupported probe design mode: {mode}")
    if n_p + n_s == 0:
        return []

    exclude = set(int(c) for c in (exclude_cell_ids or set()))
    for name, cid in mesh.well_cell_id.items():
        role = mesh.well_role.get(name, "")
        if role in ("injector", "producer"):
            exclude.add(int(cid))

    # seed anchors: existing well cell centers
    anchors = _anchor_xyz(mesh, exclude_only=False)

    var_p = prior_var_p
    var_s = prior_var_s
    if var_p is None and mode in ("variance", "hybrid"):
        var_p = _pressure_ensemble_variance(
            mesh,
            prior_k_ensemble=prior_k_ensemble,
            viscosity_pa_s=viscosity_pa_s,
            p_left=p_left,
            p_right=p_right,
            k_mean=k_mean,
            seed=seed,
        )
    if var_p is None:
        var_p = np.ones(mesh.grid.shape, dtype=float)
    if var_s is None:
        # without multi-time Sw, use flat variance → distance-driven for s
        var_s = np.ones(mesh.grid.shape, dtype=float)

    var_p_flat = np.asarray(var_p, dtype=float).ravel()
    var_s_flat = np.asarray(var_s, dtype=float).ravel()
    if var_p_flat.size != mesh.n_cells:
        raise ValueError("prior_var_p shape must match mesh cells")
    if var_s_flat.size != mesh.n_cells:
        raise ValueError("prior_var_s shape must match mesh cells")

    # normalize variances to [0,1]
    def _norm(v: NDArray[np.float64]) -> NDArray[np.float64]:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or hi - lo < EPS:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo + EPS)

    var_p_n = _norm(var_p_flat)
    var_s_n = _norm(var_s_flat)

    chosen: list[ProbeSpec] = []
    taken = set(exclude)
    alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))

    # alternate p,s while quotas remain (p first when both need)
    queue: list[str] = []
    pi = si = 0
    while pi < n_p or si < n_s:
        if pi < n_p and (si >= n_s or pi <= si):
            queue.append("observer_p")
            pi += 1
        elif si < n_s:
            queue.append("observer_s")
            si += 1
        else:
            break

    p_count = 0
    s_count = 0
    for role in queue:
        best_cid = -1
        best_score = -1.0
        var_n = var_p_n if role == "observer_p" else var_s_n
        for cid in range(mesh.n_cells):
            if cid in taken:
                continue
            xyz = np.array([mesh.x[cid], mesh.y[cid], mesh.z[cid]], dtype=float)
            dmin = _min_dist(xyz, anchors)
            # also distance to already chosen
            for sp in chosen:
                d = float(
                    np.linalg.norm(
                        xyz - np.array([sp.x, sp.y, sp.z], dtype=float)
                    )
                )
                dmin = min(dmin, d)
            # normalize distance later across candidates via running max — use raw then
            score_parts = {"dmin": dmin, "var": float(var_n[cid])}
            if mode == "maximin":
                score = dmin
            elif mode == "variance":
                score = float(var_n[cid])
            else:
                # hybrid: will re-score with normalized d after first pass — two-pass
                score = alpha * float(var_n[cid]) + (1.0 - alpha) * dmin
                score_parts["hybrid_raw"] = score
            if score > best_score:
                best_score = score
                best_cid = cid

        if best_cid < 0:
            break

        # For hybrid, re-pick with normalized dmin among remaining (better scale)
        if mode == "hybrid":
            best_cid, best_score = _hybrid_pick(
                mesh,
                taken=taken,
                anchors=anchors,
                chosen=chosen,
                var_n=var_n,
                alpha=alpha,
            )
            if best_cid < 0:
                break

        if role == "observer_p":
            p_count += 1
            name = f"P{p_count}"
        else:
            s_count += 1
            name = f"S{s_count}"
        note = f"adaptive mode={mode} score={best_score:.4g}"
        specs = _spec_from_cell(
            mesh, name, role, best_cid, score=float(best_score), note=note
        )
        chosen.append(specs)
        taken.add(best_cid)
        anchors.append(np.array([specs.x, specs.y, specs.z], dtype=float))

    return chosen


def split_n_probes(n_total: int) -> tuple[int, int]:
    """Split total exclusive probes into (n_p, n_s); p gets the extra if odd."""
    n = max(0, int(n_total))
    n_p = (n + 1) // 2
    n_s = n // 2
    return n_p, n_s


def field_variance_over_time(
    series: list[tuple[float, NDArray[np.float64]]],
) -> NDArray[np.float64]:
    """Cell-wise variance across multi-time grids (for CMG-informed DOE)."""
    if not series:
        raise ValueError("empty series")
    stack = np.stack([np.asarray(a, dtype=float) for _, a in series], axis=0)
    return np.nanvar(stack, axis=0)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _spec_from_cell(
    mesh: MeshBundle,
    name: str,
    role: str,
    cid: int,
    *,
    score: float | None,
    note: str,
) -> ProbeSpec:
    i = int(mesh.i[cid])
    j = int(mesh.j[cid])
    k = int(mesh.k[cid])
    return ProbeSpec(
        name=name,
        x=float(mesh.x[cid]),
        y=float(mesh.y[cid]),
        z=float(mesh.z[cid]),
        role=role,
        cell_id=int(cid),
        ijk=(i, j, k),
        score=score,
        notes=(note,),
    )


def _space_filling_order(mesh: MeshBundle) -> list[int]:
    # sort by k, then j, then i for a regular scan; stride pick spreads them
    idx = np.arange(mesh.n_cells, dtype=int)
    order = np.lexsort((mesh.i, mesh.j, mesh.k))
    return [int(idx[o]) for o in order]


def _pick_strided(candidates: list[int], n: int) -> list[int]:
    if n <= 0 or not candidates:
        return []
    if n >= len(candidates):
        return list(candidates)
    # even spacing along candidate list
    pos = np.linspace(0, len(candidates) - 1, n)
    picked = []
    used = set()
    for p in pos:
        i = int(round(p))
        # walk to nearest free
        for d in range(len(candidates)):
            for cand in (i + d, i - d):
                if 0 <= cand < len(candidates) and candidates[cand] not in used:
                    used.add(candidates[cand])
                    picked.append(candidates[cand])
                    break
            else:
                continue
            break
        if len(picked) >= n:
            break
    return picked[:n]


def _anchor_xyz(mesh: MeshBundle, *, exclude_only: bool) -> list[NDArray[np.float64]]:
    pts: list[NDArray[np.float64]] = []
    for name, cid in mesh.well_cell_id.items():
        role = mesh.well_role.get(name, "")
        if exclude_only or role in ("injector", "producer", "observer_p", "observer_s", "observer"):
            pts.append(
                np.array([mesh.x[cid], mesh.y[cid], mesh.z[cid]], dtype=float)
            )
    if not pts:
        # domain corners as weak anchors
        pts.append(np.array([mesh.x[0], mesh.y[0], mesh.z[0]], dtype=float))
    return pts


def _min_dist(xyz: NDArray[np.float64], anchors: list[NDArray[np.float64]]) -> float:
    if not anchors:
        return 1.0
    return float(min(np.linalg.norm(xyz - a) for a in anchors))


def _hybrid_pick(
    mesh: MeshBundle,
    *,
    taken: set[int],
    anchors: list[NDArray[np.float64]],
    chosen: list[ProbeSpec],
    var_n: NDArray[np.float64],
    alpha: float,
) -> tuple[int, float]:
    dmins = []
    cids = []
    for cid in range(mesh.n_cells):
        if cid in taken:
            continue
        xyz = np.array([mesh.x[cid], mesh.y[cid], mesh.z[cid]], dtype=float)
        dmin = _min_dist(xyz, anchors)
        for sp in chosen:
            d = float(np.linalg.norm(xyz - np.array([sp.x, sp.y, sp.z])))
            dmin = min(dmin, d)
        dmins.append(dmin)
        cids.append(cid)
    if not cids:
        return -1, -1.0
    darr = np.asarray(dmins, dtype=float)
    dlo, dhi = float(darr.min()), float(darr.max())
    if dhi - dlo < EPS:
        d_n = np.zeros_like(darr)
    else:
        d_n = (darr - dlo) / (dhi - dlo + EPS)
    best_cid = -1
    best_score = -1.0
    for i, cid in enumerate(cids):
        score = alpha * float(var_n[cid]) + (1.0 - alpha) * float(d_n[i])
        if score > best_score:
            best_score = score
            best_cid = cid
    return best_cid, best_score


def _pressure_ensemble_variance(
    mesh: MeshBundle,
    *,
    prior_k_ensemble: NDArray[np.float64] | None,
    viscosity_pa_s: float,
    p_left: float,
    p_right: float,
    k_mean: float,
    seed: int,
) -> NDArray[np.float64]:
    if prior_k_ensemble is None:
        ens = generate_logk_ensemble(
            mesh.grid.shape,
            ne=DEFAULT_ENSEMBLE_SIZE,
            k_mean=k_mean,
            logk_std=1.0,
            corr_len_cells=DEFAULT_CORR_LEN_CELLS,
            seed=seed,
        )
    else:
        ens = np.asarray(prior_k_ensemble, dtype=float)
        if ens.ndim != 4:
            raise ValueError("prior_k_ensemble must be (ne,nz,ny,nx)")

    # cell Dirichlet at injectors/producers if present
    cell_bc: dict[int, float] = {}
    for name, cid in mesh.well_cell_id.items():
        role = mesh.well_role.get(name, "")
        if role == "injector":
            cell_bc[int(cid)] = float(p_left)
        elif role == "producer":
            cell_bc[int(cid)] = float(p_right)

    pressures = []
    for m in range(ens.shape[0]):
        k = ens[m]
        try:
            res = solve_steady_state_pressure_3d(
                mesh.grid,
                k,
                k,
                k,
                float(viscosity_pa_s),
                dirichlet_boundaries={"left": float(p_left), "right": float(p_right)},
                cell_dirichlet=cell_bc or None,
            )
            pressures.append(np.asarray(res.pressure.values, dtype=float))
        except Exception:
            continue
    if not pressures:
        return np.ones(mesh.grid.shape, dtype=float)
    stack = np.stack(pressures, axis=0)
    return np.var(stack, axis=0)
