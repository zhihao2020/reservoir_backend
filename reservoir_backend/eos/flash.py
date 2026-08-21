"""Isothermal two-phase VLE flash (successive substitution + Rachford–Rice).

Wilson K-value start, Michelsen TPD stability (before / after), bracketed
Newton RR, and damped successive substitution on fugacity equality
``φ_i^L x_i = φ_i^V y_i``. Single-phase when the feed is TPD-stable, when
RR has no root in (0, 1), or when K collapses to the trivial solution K = 1.

Units: T in K, p in Pa. Standalone; not wired into the reservoir residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.peng_robinson import (
    EosMixture,
    _g_res_over_rt,
    _normalize_composition,
    compressibility_roots,
    fugacity_coefficients,
    ln_fugacity,
    mixture_AB,
    phase_zv_rho,
)

_TRIVIAL_K = 1.0e-8
_K_MIN = 1.0e-8
_K_MAX = 1.0e8


@dataclass(frozen=True)
class FlashResult:
    """TP-flash outcome. ``V`` is the vapor mole fraction in ``[0, 1]``."""

    T: float
    p: float
    z: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    V: float
    K: NDArray[np.float64]
    n_iter: int
    converged: bool
    phase_state: str  # "two-phase" | "liquid" | "vapor"
    tpd_min: float | None = None
    Z_liquid: float | None = None
    Z_vapor: float | None = None
    v_liquid: float | None = None  # m³/mol
    v_vapor: float | None = None
    rho_liquid: float | None = None  # kg/m³
    rho_vapor: float | None = None

    @property
    def L(self) -> float:
        return 1.0 - self.V

    def material_balance_residual(self) -> NDArray[np.float64]:
        return self.z - ((1.0 - self.V) * self.x + self.V * self.y)


def wilson_k(mixture: EosMixture, T: float, p: float) -> NDArray[np.float64]:
    """Wilson (1968) K-value estimate. ``T`` in K, ``p`` in Pa."""
    if T <= 0.0 or p <= 0.0:
        raise ValueError("T (K) and p (Pa) must be positive")
    return (mixture.Pc / p) * np.exp(5.373 * (1.0 + mixture.omega) * (1.0 - mixture.Tc / T))


def _rr_value(z: NDArray[np.float64], K: NDArray[np.float64], V: float) -> float:
    return float(np.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0))))


def solve_rachford_rice(
    z: NDArray[np.float64] | float,
    K: NDArray[np.float64] | float,
    *,
    allow_negative: bool = False,
) -> tuple[float, str]:
    """Solve Rachford–Rice for vapor fraction ``V``.

    Returns ``(V, phase_state)``. Unless ``allow_negative`` is set, ``V`` is
    clipped to ``[0, 1]`` and a root outside ``(0, 1)`` is reported as
    single-phase (``liquid`` or ``vapor``), never as two-phase.
    """
    z_arr = np.asarray(z, dtype=float).ravel()
    k_arr = np.clip(np.asarray(K, dtype=float).ravel(), _K_MIN, _K_MAX)
    if z_arr.size != k_arr.size:
        raise ValueError("z and K size mismatch")
    total = float(z_arr.sum())
    if total <= 0.0:
        raise ValueError("feed composition sums to zero")
    z_arr = z_arr / total

    if np.all(k_arr <= 1.0 + 1.0e-14):
        return 0.0, "liquid"
    if np.all(k_arr >= 1.0 - 1.0e-14):
        return 1.0, "vapor"

    f0 = _rr_value(z_arr, k_arr, 0.0)
    f1 = _rr_value(z_arr, k_arr, 1.0)
    if f0 <= 0.0:
        return 0.0, "liquid"
    if f1 >= 0.0:
        return 1.0, "vapor"

    lo, hi = 0.0, 1.0
    V = 0.5
    for _ in range(80):
        num = z_arr * (k_arr - 1.0)
        den = 1.0 + V * (k_arr - 1.0)
        fv = float(np.sum(num / den))
        df = float(-np.sum(num * (k_arr - 1.0) / den**2))
        if abs(fv) < 1.0e-14:
            break
        step = fv / df if abs(df) > 1.0e-30 else 0.0
        trial = V - step
        if fv > 0.0:
            lo = V
        else:
            hi = V
        if not (lo < trial < hi):
            trial = 0.5 * (lo + hi)
        if abs(trial - V) < 1.0e-14:
            V = trial
            break
        V = trial

    V = float(V)
    if allow_negative:
        return V, "two-phase"
    if V <= 0.0 or V >= 1.0:
        return (0.0, "liquid") if V <= 0.0 else (1.0, "vapor")
    return V, "two-phase"


def _phase_compositions(
    z: NDArray[np.float64], K: NDArray[np.float64], V: float, state: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if state != "two-phase":
        return z.copy(), z.copy()
    x = z / (1.0 + V * (K - 1.0))
    y = K * x
    x = np.clip(x, 0.0, None)
    y = np.clip(y, 0.0, None)
    x_sum = float(x.sum())
    y_sum = float(y.sum())
    if x_sum <= 0.0 or y_sum <= 0.0:
        return z.copy(), z.copy()
    return x / x_sum, y / y_sum


def _single_phase_by_gibbs(z: NDArray[np.float64], T: float, p: float, mixture: EosMixture) -> str:
    A, B, *_ = mixture_AB(z, T, p, mixture)
    roots = compressibility_roots(A, B)
    if roots.size == 1:
        return "vapor" if float(roots[0]) >= 0.35 else "liquid"
    z_liq = float(roots[0])
    z_vap = float(roots[-1])
    g_l = _g_res_over_rt(z_liq, A, B)
    g_v = _g_res_over_rt(z_vap, A, B)
    return "liquid" if g_l <= g_v else "vapor"


def _phase_densities(
    T: float,
    p: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    phase_state: str,
    mixture: EosMixture | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if mixture is None or mixture.Mw is None:
        return None, None, None, None, None, None
    z_l = v_l = rho_l = z_v = v_v = rho_v = None
    try:
        if phase_state in ("two-phase", "liquid"):
            z_l, v_l, rho_l = phase_zv_rho(x, T, p, mixture, phase="liquid")
        if phase_state in ("two-phase", "vapor"):
            z_v, v_v, rho_v = phase_zv_rho(y, T, p, mixture, phase="vapor")
    except ValueError:
        return None, None, None, None, None, None
    return z_l, z_v, v_l, v_v, rho_l, rho_v


def _result(
    T: float,
    p: float,
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    V: float,
    K: NDArray[np.float64],
    n_iter: int,
    converged: bool,
    phase_state: str,
    *,
    mixture: EosMixture | None = None,
    tpd_min: float | None = None,
) -> FlashResult:
    if phase_state != "two-phase":
        V = 0.0 if phase_state == "liquid" else 1.0
        x = z.copy()
        y = z.copy()
    z_l, z_v, v_l, v_v, rho_l, rho_v = _phase_densities(T, p, x, y, phase_state, mixture)
    return FlashResult(
        T=float(T),
        p=float(p),
        z=z.copy(),
        x=x.copy(),
        y=y.copy(),
        V=float(V),
        K=K.copy(),
        n_iter=int(n_iter),
        converged=bool(converged),
        phase_state=phase_state,
        tpd_min=tpd_min,
        Z_liquid=z_l,
        Z_vapor=z_v,
        v_liquid=v_l,
        v_vapor=v_v,
        rho_liquid=rho_l,
        rho_vapor=rho_v,
    )


def flash_tp(
    z: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    tol: float = 1.0e-8,
    max_iter: int = 200,
    damping: float = 1.0,
) -> FlashResult:
    """Isothermal two-phase flash at ``(T [K], p [Pa])`` for feed ``z``.

    Returns liquid ``x``, vapor ``y``, vapor fraction ``V``, and ``K = y/x``.
    Material balance: ``z = (1−V) x + V y``. Out-of-range / negative-flash
    ``V`` is never labelled two-phase.
    """
    from reservoir_backend.eos.stability import michelsen_stability

    z_arr = _normalize_composition(z, mixture.n_components)
    stab = michelsen_stability(z_arr, T, p, mixture)
    pack = dict(mixture=mixture, tpd_min=stab.tpd_min)
    if stab.stable:
        state = _single_phase_by_gibbs(z_arr, T, p, mixture)
        return _result(T, p, z_arr, z_arr, z_arr, 0.0, wilson_k(mixture, T, p), 0, True, state, **pack)

    if (not stab.trivial_vapor) and (not stab.trivial_liquid):
        K = np.clip(stab.w_vapor / np.clip(stab.w_liquid, 1.0e-16, None), _K_MIN, _K_MAX)
    else:
        K = np.clip(wilson_k(mixture, T, p), _K_MIN, _K_MAX)
    damp = float(np.clip(damping, 0.05, 1.0))
    best_res = np.inf
    x = z_arr.copy()
    y = z_arr.copy()
    V = 0.5
    state = "vapor"
    K_prev = K.copy()

    for it in range(1, max_iter + 1):
        if float(np.max(np.abs(K - 1.0))) < _TRIVIAL_K:
            state = _single_phase_by_gibbs(z_arr, T, p, mixture)
            return _result(T, p, z_arr, z_arr, z_arr, 0.0, K, it, True, state, **pack)

        V, state = solve_rachford_rice(z_arr, K)
        x, y = _phase_compositions(z_arr, K, V, state)
        if state != "two-phase":
            # Wilson already single-phase, or a damped step still left (0, 1).
            if it == 1 or damp <= 0.16:
                return _result(T, p, z_arr, x, y, V, K, it, True, state, **pack)
            K = K_prev.copy()
            damp = max(0.15, 0.5 * damp)
            continue

        phi_l = fugacity_coefficients(x, T, p, mixture, phase="liquid")
        phi_v = fugacity_coefficients(y, T, p, mixture, phase="vapor")
        ln_f_l = np.log(phi_l) + np.log(np.clip(x, 1.0e-16, None))
        ln_f_v = np.log(phi_v) + np.log(np.clip(y, 1.0e-16, None))
        residual = float(np.max(np.abs(ln_f_l - ln_f_v)))
        if residual < tol:
            return _result(T, p, z_arr, x, y, V, K, it, True, "two-phase", **pack)

        K_ss = np.clip(phi_l / np.clip(phi_v, 1.0e-30, None), _K_MIN, _K_MAX)
        if residual > best_res * 1.01:
            damp = max(0.15, 0.5 * damp)
        else:
            damp = min(1.0, damp * 1.1)
            best_res = residual
        K_prev = K.copy()
        ln_k = np.log(K) + damp * (np.log(K_ss) - np.log(K))
        K = np.clip(np.exp(np.clip(ln_k, np.log(_K_MIN), np.log(_K_MAX))), _K_MIN, _K_MAX)

    # Last iterate: accept two-phase only if RR stayed inside (0, 1) and
    # fugacity residual is modest; otherwise drop to single-phase Gibbs.
    V, state = solve_rachford_rice(z_arr, K)
    x, y = _phase_compositions(z_arr, K, V, state)
    if state == "two-phase":
        ln_f_l = ln_fugacity(x, T, p, mixture, phase="liquid")
        ln_f_v = ln_fugacity(y, T, p, mixture, phase="vapor")
        if float(np.max(np.abs(ln_f_l - ln_f_v))) < 50.0 * tol:
            return _result(T, p, z_arr, x, y, V, K, max_iter, True, "two-phase", **pack)
    state = _single_phase_by_gibbs(z_arr, T, p, mixture)
    return _result(T, p, z_arr, z_arr, z_arr, 0.0, K, max_iter, False, state, **pack)
