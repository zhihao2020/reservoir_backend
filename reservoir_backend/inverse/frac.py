"""CMOST-style fracture-strip parameterization for shale depletion twins."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN

MD_TO_M2 = 9.869233e-16


@dataclass(frozen=True)
class WellTrack:
    """Horizontal completion segment in 0-based IJK indices."""

    name: str
    j: int
    k: int
    i0: int
    i1: int
    open_from_day: float = 0.0


def _frac_plane_indices(i0: int, i1: int, n_frac: int, phase: float) -> list[int]:
    n_frac = int(max(1, min(int(round(n_frac)), max(i1 - i0 + 1, 1))))
    phase = float(phase) % 1.0
    if n_frac == 1:
        return [int((i0 + i1) // 2)]
    out: list[int] = []
    for idx in range(n_frac):
        t = phase + idx / float(n_frac - 1)
        ii = int(round(i0 + t * (i1 - i0)))
        ii = max(i0, min(i1, ii))
        out.append(ii)
    return sorted(set(out))


def paint_fracture_strips(
    grid: CartesianGrid,
    wells: tuple[WellTrack, ...],
    *,
    log_k_m: float,
    log_k_f: float,
    log_k_srv: float,
    x_f_m: float,
    n_frac: int,
    frac_phase: float,
    frac_k_layers: tuple[int, ...] = (1, 2, 3),
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Return (k, frac_mask, srv_mask) on the simulation grid."""
    k = np.full(grid.n_cells, float(np.exp(np.clip(log_k_m, LOGK_MIN, LOGK_MAX))), dtype=float)
    frac = np.zeros(grid.n_cells, dtype=bool)
    srv = np.zeros(grid.n_cells, dtype=bool)
    k_frac = float(np.exp(np.clip(log_k_f, LOGK_MIN, LOGK_MAX)))
    k_srv = float(np.exp(np.clip(log_k_srv, LOGK_MIN, LOGK_MAX)))

    for well in wells:
        j_well = int(well.j)
        dy = float(grid.dy[j_well])
        half = max(1, int(round(float(x_f_m) / max(dy, 1.0e-30))))
        j_lo = max(0, j_well - half)
        j_hi = min(grid.ny, j_well + half + 1)
        layers = tuple(int(kk) for kk in frac_k_layers if 0 <= int(kk) < grid.nz)
        if not layers:
            layers = (max(0, grid.nz // 2),)

        for ii in _frac_plane_indices(int(well.i0), int(well.i1), n_frac, frac_phase):
            for j in range(j_lo, j_hi):
                for kk in layers:
                    c = grid.index(ii, j, kk)
                    frac[c] = True
                    srv[c] = False
                    k[c] = k_frac
            for di in (-1, 1):
                ia = ii + di
                if not (0 <= ia < grid.nx):
                    continue
                for j in range(j_lo, j_hi):
                    for kk in layers:
                        c = grid.index(ia, j, kk)
                        if frac[c]:
                            continue
                        srv[c] = True
                        k[c] = k_srv

    return k, frac, srv


@dataclass
class FractureStripParameterization:
    """Fracture-strip θ for shale depletion.

    Default (``free_geometry=False``): θ = [log k_m, log k_f, log k_srv, log x_f].
    Stage count and phase are fixed from completion design — BHP alone cannot
    identify the discrete ``n_frac`` (FD Jacobian is ~0 under rounding).

    Expert (``free_geometry=True``): also free [n_frac, frac_phase] (6-D).
    """

    grid: CartesianGrid
    wells: tuple[WellTrack, ...]
    phi: float = 0.08
    frac_k_layers: tuple[int, ...] = (1, 2, 3)
    prior_mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    prior_std: NDArray[np.float64] = field(default_factory=lambda: np.full(4, 0.5))
    n_frac_min: int = 1
    n_frac_max: int = 12
    log_x_f_min: float = float(np.log(5.0))
    log_x_f_max: float = float(np.log(250.0))
    frac_aperture_m: float | None = None
    free_geometry: bool = False
    fixed_n_frac: float = 5.0
    fixed_phase: float = 0.0

    def __post_init__(self) -> None:
        if not self.wells:
            raise ValueError("FractureStripParameterization needs at least one well track")
        n = 6 if self.free_geometry else 4
        self.prior_mean = np.asarray(self.prior_mean, dtype=float).ravel()
        self.prior_std = np.asarray(self.prior_std, dtype=float).ravel()
        if self.prior_mean.size == 6 and n == 4:
            object.__setattr__(self, "fixed_n_frac", float(self.prior_mean[4]))
            object.__setattr__(self, "fixed_phase", float(self.prior_mean[5] % 1.0))
            self.prior_mean = self.prior_mean[:4].copy()
        if self.prior_std.size == 6 and n == 4:
            self.prior_std = self.prior_std[:4].copy()
        if self.prior_mean.size == 4 and n == 6:
            self.prior_mean = np.concatenate(
                [self.prior_mean, [self.fixed_n_frac, self.fixed_phase]]
            )
        if self.prior_std.size == 4 and n == 6:
            self.prior_std = np.concatenate([self.prior_std, [0.75, 0.15]])
        if self.prior_mean.size == 0:
            self.prior_mean = np.zeros(n, dtype=float)
        if self.prior_mean.size != n:
            raise ValueError(f"prior_mean must have length {n}")
        if self.prior_std.size == 0:
            self.prior_std = np.full(n, 0.5)
        if self.prior_std.size == 1:
            object.__setattr__(self, "prior_std", np.full(n, float(self.prior_std[0])))
        elif self.prior_std.size != n:
            raise ValueError(f"prior_std must be scalar or length {n}")
        object.__setattr__(
            self,
            "fixed_n_frac",
            float(np.clip(self.fixed_n_frac, self.n_frac_min, self.n_frac_max)),
        )
        object.__setattr__(self, "fixed_phase", float(self.fixed_phase % 1.0))

    @property
    def n_params(self) -> int:
        return 6 if self.free_geometry else 4

    def _to_full(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.asarray(theta, dtype=float).ravel()
        if th.size == 6:
            return th.copy()
        if th.size == 4:
            return np.array(
                [th[0], th[1], th[2], th[3], self.fixed_n_frac, self.fixed_phase],
                dtype=float,
            )
        raise ValueError(f"theta size {th.size} not in {{4, 6}}")

    def project(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        full = self._to_full(theta)
        full[0:3] = np.clip(full[0:3], LOGK_MIN, LOGK_MAX)
        full[3] = float(np.clip(full[3], self.log_x_f_min, self.log_x_f_max))
        full[4] = float(np.clip(full[4], self.n_frac_min, self.n_frac_max))
        full[5] = float(full[5] % 1.0)
        if self.free_geometry:
            return full
        return full[:4].copy()

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = self.project(theta)
        full = self._to_full(th)
        if not self.free_geometry:
            full[4] = self.fixed_n_frac
            full[5] = self.fixed_phase
        k, _, _ = paint_fracture_strips(
            self.grid,
            self.wells,
            log_k_m=float(full[0]),
            log_k_f=float(full[1]),
            log_k_srv=float(full[2]),
            x_f_m=float(np.exp(full[3])),
            n_frac=int(round(full[4])),
            frac_phase=float(full[5]),
            frac_k_layers=self.frac_k_layers,
        )
        return k


def decode_frac_theta(
    parameterization: FractureStripParameterization,
    theta: NDArray[np.float64],
) -> dict[str, float]:
    """Engineering decode for ruler reports."""
    th4 = parameterization.project(theta)
    full = parameterization._to_full(th4)
    if not parameterization.free_geometry:
        full[4] = parameterization.fixed_n_frac
        full[5] = parameterization.fixed_phase
    k_m = float(np.exp(full[0]))
    k_f = float(np.exp(full[1]))
    grid = parameterization.grid
    well = parameterization.wells[0]
    di = float(np.mean(grid.dx[max(well.i0, 0) : min(well.i1 + 1, grid.nx)]))
    if parameterization.frac_aperture_m is not None:
        aperture = float(parameterization.frac_aperture_m)
    else:
        aperture = di
    return {
        "log_k_m": float(full[0]),
        "log_k_f": float(full[1]),
        "log_k_srv": float(full[2]),
        "x_f_m": float(np.exp(full[3])),
        "n_frac": float(int(round(full[4]))),
        "frac_phase": float(full[5]),
        "F_cd_m3": float(k_f * aperture),
        "k_frac_over_matrix": float(k_f / max(k_m, 1.0e-30)),
        "free_geometry": bool(parameterization.free_geometry),
    }


def default_shale_prior(
    truth: dict | None = None,
    *,
    free_geometry: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SI prior mean/std for frac θ from optional truth JSON."""
    k_m = 0.001 * MD_TO_M2
    k_f = 8000.0 * MD_TO_M2
    k_srv = 0.4 * MD_TO_M2
    x_f = 40.0
    n_frac = 5.0
    if truth is not None:
        k_m = float(truth.get("matrix_perm_md", {}).get("kx_geo", 0.001)) * MD_TO_M2
        k_f = float(truth.get("frac_perm_md", 8000.0)) * MD_TO_M2
        k_srv = float(truth.get("srv_perm_md", 0.4)) * MD_TO_M2
        planes = truth.get("frac_i_planes") or []
        n_frac = float(max(len(planes), 1))
        grid = truth.get("grid") or {}
        dj = float(grid.get("dj_ft", 50.0)) * 0.3048
        x_f = float(truth.get("frac_half_length_m") or (5.0 * dj))
        if "frac_half_length_ft" in truth:
            x_f = float(truth["frac_half_length_ft"]) * 0.3048
        # half-length from high-K block j-span when present
        blocks = truth.get("high_k_blocks_ijk") or truth.get("channel_blocks_ijk") or []
        if blocks:
            js = [int(b[1]) for b in blocks if len(b) >= 2]
            if js:
                x_f = 0.5 * (float(max(js) - min(js)) + 1.0) * dj
    mean6 = np.array(
        [np.log(k_m), np.log(k_f), np.log(k_srv), np.log(max(x_f, 1.0)), n_frac, 0.0],
        dtype=float,
    )
    # Broader continuous priors so LM can move; geometry fixed by default
    std6 = np.array([0.8, 0.6, 0.8, 0.40, 0.75, 0.15], dtype=float)
    if free_geometry:
        return mean6, std6
    return mean6[:4].copy(), std6[:4].copy()


def wells_from_truth(truth: dict) -> tuple[WellTrack, ...]:
    out: list[WellTrack] = []
    for w in truth.get("wells") or []:
        out.append(
            WellTrack(
                name=str(w["name"]),
                j=int(w["j"]) - 1,
                k=int(w["k"]) - 1,
                i0=int(w["i0"]) - 1,
                i1=int(w["i1"]) - 1,
                open_from_day=float(w.get("open_from_day") or 0.0),
            )
        )
    if not out:
        raise ValueError("truth JSON has no wells")
    return tuple(out)
