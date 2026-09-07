"""Cartesian linear isotropic elasticity for optional flow coupling.

``*NOCOUPERM``: permeability is never updated. Pore volume follows
``V_p = φ_ref V_cell exp(cpor Δp) (1 + α θ)`` with θ = div u.
``boundary: unconstrained`` is GEM ``*GCFACTOR 0`` (zero incremental traction).
Enable only from YAML ``geomech.enabled``. Product DPDP stays off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from reservoir_backend.grid.cartesian import CartesianGrid


@dataclass
class GeomechSpec:
    enabled: bool = False
    nocouperm: bool = True
    boundary: str = "unconstrained"
    biot: float = 1.0
    E: float = 0.0
    nu: float = 0.22
    k_dry: float = 0.0
    p_ref: float = 1.0e5

    def __post_init__(self) -> None:
        b = str(self.boundary).strip().lower()
        if b in {"unconstrained", "free", "gcfactor0", "gfactor0"}:
            b = "unconstrained"
        elif b in {"confined", "fixed", "jacket"}:
            b = "confined"
        else:
            raise ValueError(f"geomech.boundary must be unconstrained or confined, got {self.boundary!r}")
        object.__setattr__(self, "boundary", b)
        if self.enabled:
            if float(self.E) <= 0.0 and float(self.k_dry) <= 0.0:
                raise ValueError("geomech.enabled needs E or K_dry")
            if not (0.0 <= float(self.nu) < 0.5):
                raise ValueError("geomech.nu must be in [0, 0.5)")
            if not bool(self.nocouperm):
                raise ValueError("permeability coupling is not implemented; set geomech.nocouperm: true")

    @property
    def K_dr(self) -> float:
        lam, mu = self.lame
        return lam + 2.0 * mu / 3.0

    @property
    def lame(self) -> tuple[float, float]:
        E = float(self.E)
        nu = float(self.nu)
        if E <= 0.0:
            K = float(self.k_dry)
            E = 3.0 * K * (1.0 - 2.0 * nu)
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lam, mu


def geomech_from_cfg(cfg: dict[str, Any], *, p_ref: float = 1.0e5) -> GeomechSpec:
    """Read a top-level or ``physics.geomech`` mapping. Missing/false → disabled."""
    raw = cfg.get("geomech")
    if raw is None:
        raw = (cfg.get("physics") or {}).get("geomech")
    if raw is None or raw is False:
        return GeomechSpec(enabled=False, p_ref=float(p_ref))
    if raw is True:
        raise ValueError("geomech: true needs E_GPa or K_dry_GPa")
    if not isinstance(raw, dict):
        raise ValueError("geomech must be a mapping")
    enabled = bool(raw.get("enabled", False))
    E = 0.0
    if raw.get("E") is not None:
        E = float(raw["E"])
    if raw.get("E_Pa") is not None:
        E = float(raw["E_Pa"])
    if raw.get("E_GPa") is not None:
        E = float(raw["E_GPa"]) * 1.0e9
    if raw.get("elastmod_kPa") is not None:
        E = float(raw["elastmod_kPa"]) * 1.0e3
    k_dry = 0.0
    if raw.get("k_dry") is not None:
        k_dry = float(raw["k_dry"])
    if raw.get("K_dry_Pa") is not None:
        k_dry = float(raw["K_dry_Pa"])
    if raw.get("K_dry_GPa") is not None:
        k_dry = float(raw["K_dry_GPa"]) * 1.0e9
    nu = float(raw.get("nu", raw.get("poissratio", raw.get("poisson", 0.22))))
    biot = float(raw.get("biot", raw.get("biotscoef", 1.0)))
    boundary = str(raw.get("boundary", "unconstrained"))
    if raw.get("gcfactor") is not None and float(raw["gcfactor"]) == 0.0:
        boundary = "unconstrained"
    p = float(raw.get("p_ref", raw.get("prpor", p_ref)))
    return GeomechSpec(
        enabled=enabled,
        nocouperm=bool(raw.get("nocouperm", True)),
        boundary=boundary,
        biot=biot,
        E=E,
        nu=nu,
        k_dry=k_dry,
        p_ref=p,
    )


_HEX = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)
_GAUSS = 1.0 / np.sqrt(3.0)


def _hex_nodes(ci: int, cj: int, ck: int, nxp: int, nyp: int) -> NDArray[np.int64]:
    corners = (
        (ci, cj, ck),
        (ci + 1, cj, ck),
        (ci + 1, cj + 1, ck),
        (ci, cj + 1, ck),
        (ci, cj, ck + 1),
        (ci + 1, cj, ck + 1),
        (ci + 1, cj + 1, ck + 1),
        (ci, cj + 1, ck + 1),
    )
    return np.array([k * nyp * nxp + j * nxp + i for i, j, k in corners], dtype=np.int64)


def _dN_dxyz(xi: float, eta: float, zeta: float, hx: float, hy: float, hz: float) -> NDArray[np.float64]:
    """8x3 derivatives of hex shape functions in physical coords (origin at cell centre)."""
    a = _HEX
    dN_dxi = 0.125 * a[:, 0] * (1.0 + a[:, 1] * eta) * (1.0 + a[:, 2] * zeta)
    dN_det = 0.125 * a[:, 1] * (1.0 + a[:, 0] * xi) * (1.0 + a[:, 2] * zeta)
    dN_dze = 0.125 * a[:, 2] * (1.0 + a[:, 0] * xi) * (1.0 + a[:, 1] * eta)
    dN_dx = dN_dxi * (2.0 / hx)
    dN_dy = dN_det * (2.0 / hy)
    dN_dz = dN_dze * (2.0 / hz)
    return np.stack([dN_dx, dN_dy, dN_dz], axis=1)


def _B_matrix(dN: NDArray[np.float64]) -> NDArray[np.float64]:
    B = np.zeros((6, 24))
    ax = np.arange(8) * 3
    B[0, ax] = dN[:, 0]
    B[1, ax + 1] = dN[:, 1]
    B[2, ax + 2] = dN[:, 2]
    B[3, ax + 1] = dN[:, 2]
    B[3, ax + 2] = dN[:, 1]
    B[4, ax] = dN[:, 2]
    B[4, ax + 2] = dN[:, 0]
    B[5, ax] = dN[:, 1]
    B[5, ax + 1] = dN[:, 0]
    return B


def _hex_ke_and_q(hx: float, hy: float, hz: float, C: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ke = np.zeros((24, 24))
    q = np.zeros(24)
    detJ = 0.125 * hx * hy * hz
    gps = (-_GAUSS, _GAUSS)
    for z in gps:
        for e in gps:
            for x in gps:
                dN = _dN_dxyz(float(x), float(e), float(z), hx, hy, hz)
                B = _B_matrix(dN)
                ke += (B.T @ C @ B) * detJ
                q[0::3] += dN[:, 0] * detJ
                q[1::3] += dN[:, 1] * detJ
                q[2::3] += dN[:, 2] * detJ
    return ke, q


def _isotropic_C(lam: float, mu: float) -> NDArray[np.float64]:
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


class CartesianElasticity:
    """Factorized hex8 elasticity on a Cartesian grid. Reused every Newton step."""

    def __init__(self, grid: CartesianGrid, spec: GeomechSpec):
        self.grid = grid
        self.spec = spec
        self._factor, self._Q = self._assemble()
        self._p_last: NDArray[np.float64] | None = None
        self._theta_last: NDArray[np.float64] | None = None

    def volumetric_strain(self, pressure: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(pressure, dtype=float).ravel()
        if self._p_last is not None and np.array_equal(self._p_last, p):
            return self._theta_last
        if self._factor is None:
            theta = np.zeros(self.grid.n_cells)
            self._p_last = np.asarray(p, dtype=float).copy()
            self._theta_last = theta
            return theta
        dp = p - float(self.spec.p_ref)
        rhs = float(self.spec.biot) * (self._Q @ dp)
        u = self._factor.solve(rhs)
        vol = self.grid.cell_volumes()
        theta = (self._Q.T @ u) / np.maximum(vol, 1.0e-30)
        self._p_last = np.asarray(p, dtype=float).copy()
        self._theta_last = theta
        return theta

    def _nxyz(self) -> tuple[int, int, int, int]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        return nx, ny, nz, (nx + 1) * (ny + 1) * (nz + 1)

    def _assemble(self):
        grid = self.grid
        nx, ny, nz, nnode = self._nxyz()
        nxp, nyp = nx + 1, ny + 1
        ndof = 3 * nnode
        n_cells = grid.n_cells
        lam, mu = self.spec.lame
        C = _isotropic_C(lam, mu)
        nnz = n_cells * 24 * 24
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float64)
        q_r = np.empty(n_cells * 24, dtype=np.int64)
        q_c = np.empty(n_cells * 24, dtype=np.int64)
        q_d = np.empty(n_cells * 24, dtype=np.float64)
        cache: dict[tuple[float, float, float], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        idx = 0
        qidx = 0
        cell = 0
        for ck in range(nz):
            hz = float(grid.dz[ck])
            for cj in range(ny):
                hy = float(grid.dy[cj])
                for ci in range(nx):
                    hx = float(grid.dx[ci])
                    key = (hx, hy, hz)
                    hit = cache.get(key)
                    if hit is None:
                        hit = _hex_ke_and_q(hx, hy, hz, C)
                        cache[key] = hit
                    ke, q = hit
                    nodes = _hex_nodes(ci, cj, ck, nxp, nyp)
                    dofs = np.repeat(nodes * 3, 3) + np.tile(np.arange(3), 8)
                    rows[idx : idx + 576] = np.repeat(dofs, 24)
                    cols[idx : idx + 576] = np.tile(dofs, 24)
                    data[idx : idx + 576] = ke.ravel()
                    idx += 576
                    q_r[qidx : qidx + 24] = dofs
                    q_c[qidx : qidx + 24] = cell
                    q_d[qidx : qidx + 24] = q
                    qidx += 24
                    cell += 1
        K = sparse.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()
        Q = sparse.coo_matrix((q_d, (q_r, q_c)), shape=(ndof, n_cells)).tocsc()
        if self.spec.boundary == "confined":
            keep = np.ones(ndof, dtype=bool)
            for k in range(nz + 1):
                for j in range(nyp):
                    for i in range(nxp):
                        if i in (0, nx) or j in (0, ny) or k in (0, nz):
                            n = k * nyp * nxp + j * nxp + i
                            keep[3 * n : 3 * n + 3] = False
            idx_keep = np.where(keep)[0]
            if idx_keep.size == 0:
                return None, sparse.csr_matrix((0, n_cells))
            K = K[idx_keep][:, idx_keep]
            Q = Q[idx_keep]
        else:
            diag = np.asarray(K.diagonal(), dtype=float)
            scale = max(float(np.mean(np.abs(diag))), 1.0)
            K = K + sparse.diags(1.0e-8 * scale * np.ones(ndof))
        return splu(K.tocsc()), Q.tocsr()
