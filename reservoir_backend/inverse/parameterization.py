"""Fixed-dimension parameterizations. Default is not one K per cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN, exp_permeability


@dataclass
class RegionParameterization:
    """One log-k parameter per integer region."""

    region_id: NDArray[np.int64]
    phi: float = 0.20

    def __post_init__(self) -> None:
        self.region_id = np.asarray(self.region_id, dtype=np.int64).ravel()
        if self.region_id.size == 0:
            raise ValueError("region_id is empty")
        if np.any(self.region_id < 0):
            raise ValueError("region ids must be >= 0")

    @property
    def n_params(self) -> int:
        return int(self.region_id.max()) + 1

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.clip(np.asarray(theta, dtype=float).ravel(), LOGK_MIN, LOGK_MAX)
        if th.size != self.n_params:
            raise ValueError(f"theta size {th.size} != {self.n_params}")
        return exp_permeability(th[self.region_id])

    def sample_prior(
        self,
        n_ensemble: int,
        mean: NDArray[np.float64] | float,
        std: NDArray[np.float64] | float,
        seed: int,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mu = np.broadcast_to(np.asarray(mean, dtype=float), (self.n_params,)).copy()
        sig = np.broadcast_to(np.asarray(std, dtype=float), (self.n_params,)).copy()
        ens = rng.normal(mu[None, :], sig[None, :], size=(int(n_ensemble), self.n_params))
        return np.clip(ens, LOGK_MIN, LOGK_MAX)


@dataclass
class ContrastParameterization:
    """θ = [log k_background, log(k_body / k_background)].

    Region 1 is a known high-K body (channel, top layer). The sign of the
    contrast is structure, like PVT — not inverted. Magnitudes are inverted.
    """

    region_id: NDArray[np.int64]
    phi: float = 0.20
    log_contrast_mean: float = float(np.log(20.0))
    log_contrast_std: float = 1.00
    log_contrast_min: float = 0.0
    log_contrast_max: float = float(np.log(200.0))

    def __post_init__(self) -> None:
        self.region_id = np.asarray(self.region_id, dtype=np.int64).ravel()
        if self.region_id.size == 0:
            raise ValueError("region_id is empty")
        if int(self.region_id.min()) != 0 or int(self.region_id.max()) != 1:
            raise ValueError("contrast parameterization needs region ids {0, 1}")

    @property
    def n_params(self) -> int:
        return 2

    def project(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.asarray(theta, dtype=float).ravel()
        if th.size != 2:
            raise ValueError(f"theta size {th.size} != 2")
        th[0] = float(np.clip(th[0], LOGK_MIN, LOGK_MAX))
        th[1] = float(np.clip(th[1], self.log_contrast_min, self.log_contrast_max))
        return th

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        log_k0, log_c = self.project(theta)
        log_k1 = float(np.clip(log_k0 + log_c, LOGK_MIN, LOGK_MAX))
        vals = np.array([log_k0, log_k1], dtype=float)
        return exp_permeability(vals[self.region_id])

    def sample_prior(
        self,
        n_ensemble: int,
        mean: NDArray[np.float64] | float,
        std: NDArray[np.float64] | float,
        seed: int,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mu0 = float(np.mean(np.asarray(mean, dtype=float)))
        sig0 = float(np.mean(np.asarray(std, dtype=float)))
        logk = rng.normal(mu0, max(sig0, 1.0e-8), size=int(n_ensemble))
        logc = rng.normal(self.log_contrast_mean, self.log_contrast_std, size=int(n_ensemble))
        ens = np.stack([logk, logc], axis=1)
        return np.stack([self.project(row) for row in ens], axis=0)


@dataclass
class CoarseFieldParameterization:
    """log-k on a coarse Cartesian lattice, nearest-cell map to the fine grid."""

    grid: CartesianGrid
    nx: int
    ny: int
    nz: int
    phi: float = 0.20

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError("coarse dimensions must be positive")

    @property
    def n_params(self) -> int:
        return int(self.nx * self.ny * self.nz)

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.clip(np.asarray(theta, dtype=float).ravel(), LOGK_MIN, LOGK_MAX)
        if th.size != self.n_params:
            raise ValueError(f"theta size {th.size} != {self.n_params}")
        coarse = th.reshape(self.nz, self.ny, self.nx)
        centers = self.grid.cell_centers()
        lx, ly, lz = self.grid.size_m()
        ox, oy, oz = self.grid.origin
        fi = np.clip(((centers[:, 0] - ox) / max(lx, 1.0e-30)) * self.nx, 0, self.nx - 1e-9)
        fj = np.clip(((centers[:, 1] - oy) / max(ly, 1.0e-30)) * self.ny, 0, self.ny - 1e-9)
        fk = np.clip(((centers[:, 2] - oz) / max(lz, 1.0e-30)) * self.nz, 0, self.nz - 1e-9)
        ii = np.floor(fi).astype(int)
        jj = np.floor(fj).astype(int)
        kk = np.floor(fk).astype(int)
        return exp_permeability(coarse[kk, jj, ii])

    def sample_prior(
        self,
        n_ensemble: int,
        mean: NDArray[np.float64] | float,
        std: NDArray[np.float64] | float,
        seed: int,
        corr_cells: float = 1.5,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mu = np.broadcast_to(np.asarray(mean, dtype=float), (self.n_params,)).reshape(
            self.nz, self.ny, self.nx
        )
        sig = float(np.mean(np.asarray(std, dtype=float)))
        ens = []
        for _ in range(int(n_ensemble)):
            noise = rng.normal(0.0, 1.0, size=mu.shape)
            if corr_cells > 0.5 and min(self.nx, self.ny, self.nz) > 1:
                noise = _smooth3(noise, passes=max(1, int(round(corr_cells))))
                noise = noise / (float(np.std(noise)) + 1.0e-30)
            ens.append(np.clip((mu + sig * noise).ravel(), LOGK_MIN, LOGK_MAX))
        return np.stack(ens, axis=0)


def _smooth3(arr: NDArray[np.float64], passes: int) -> NDArray[np.float64]:
    out = arr.astype(float, copy=True)
    for _ in range(int(passes)):
        padded = np.pad(out, 1, mode="edge")
        acc = padded[1:-1, 1:-1, 1:-1] * 6.0
        acc += padded[1:-1, 1:-1, 0:-2] + padded[1:-1, 1:-1, 2:]
        acc += padded[1:-1, 0:-2, 1:-1] + padded[1:-1, 2:, 1:-1]
        acc += padded[0:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1]
        out = acc / 12.0
    return out
