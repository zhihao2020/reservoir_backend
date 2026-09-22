"""Stateful online reconstruction: bounded window + non-blocking rock inversion.

The naive per-step ``run_pipeline`` re-solves interpolation *and* rock inversion
over the whole accumulated history, so its cost grows O(N) and every step blocks
for the slow inversion. This session keeps that cost bounded instead:

- a **sliding window** (``window`` most recent times) bounds interpolation,
  smoothing, inversion, and the UDP re-send to O(window), independent of how long
  the session runs;
- pressure/saturation interpolation is **incremental** — only the new time is
  interpolated and causally smoothed (``smooth_fields`` is a causal recurrence,
  so smoothing the last row against the previous one reproduces the batch result
  exactly) — so the real-time path is O(n_cells) per step;
- the **rock inversion** (k/phi) runs on a **background thread**, coalesced
  (latest-wins). ``step()`` returns fresh p/sw/so/sg immediately; the k/phi from
  a finished inversion is picked up by the next ``step()`` (or a control-port
  RESEND), so the caller never blocks on it.

``step()`` returns the current :class:`ProgramFields` with the windowed fields and
the latest (possibly prior) k/phi, so the existing emit/UDP path is unchanged.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.lab_case import LabCase
from ..exceptions import CaseSchemaError
from .mesh import MeshResult
from .pipeline import ProgramFields, run_mesh
from .pressure import interpolate_pressure
from .rock import RockDiagnostics, invert_rock, invert_rock_three_phase, transient_weights
from .saturation import (
    interpolate_saturation,
    project_saturations3,
)

_JUMP_REL = 0.2
_ALPHA = 0.3


def _saturation_slice(case: LabCase, t: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """One time slice of the phase saturations, carrying forward the previous
    time when a phase is unobserved. Matches ``pipeline._saturation_slice``."""
    sw = np.array(case.sw[t], dtype=float, copy=True)
    so = np.array(case.so[t], dtype=float, copy=True)
    sg = np.array(case.sg[t], dtype=float, copy=True)
    if not np.isfinite(sw).any() and t > 0 and np.isfinite(case.sw[t - 1]).any():
        sw = np.array(case.sw[t - 1], dtype=float, copy=True)
    if not np.isfinite(so).any() and t > 0 and np.isfinite(case.so[t - 1]).any():
        so = np.array(case.so[t - 1], dtype=float, copy=True)
    if not np.isfinite(sg).any() and t > 0 and np.isfinite(case.sg[t - 1]).any():
        sg = np.array(case.sg[t - 1], dtype=float, copy=True)
    return sw, so, sg


def _smooth_last(arr: NDArray[np.float64]) -> None:
    """Causal weak temporal smoothing of the last row against its predecessor.

    This is exactly ``saturation.smooth_fields`` restricted to the final step; the
    recurrence is causal so the result equals the batch smoother at that row.
    """
    if arr.shape[0] < 2:
        return
    prev = arr[-2]
    cur = arr[-1]
    scale = float(np.sqrt(np.mean(prev ** 2)))
    rel = float(np.sqrt(np.mean((cur - prev) ** 2))) / scale if scale > 0.0 else 0.0
    if rel < _JUMP_REL:
        arr[-1] = (1.0 - _ALPHA) * cur + _ALPHA * prev


# 滑动窗口：只看上下相邻时刻（当前 + 上一拍）
_WINDOW = 2


@dataclass
class InversionSession:
    """Stateful reconstruction with bounded window and background inversion."""

    case: LabCase
    mesh: MeshResult
    window: int = _WINDOW
    _pressure_method: str | None = None
    _sat_method: str | None = None
    _p: NDArray[np.float64] = field(default=None, repr=False)
    _sw: NDArray[np.float64] = field(default=None, repr=False)
    _so: NDArray[np.float64] = field(default=None, repr=False)
    _sg: NDArray[np.float64] = field(default=None, repr=False)
    _phi: NDArray[np.float64] | None = field(default=None, repr=False)
    _k: NDArray[np.float64] | None = field(default=None, repr=False)
    _diag: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _worker: threading.Thread | None = field(default=None, repr=False)
    _pending: bool = field(default=False, repr=False)

    @classmethod
    def open(cls, case: LabCase, *, window: int = _WINDOW) -> InversionSession:
        issues = case.static_issues()
        if issues:
            raise CaseSchemaError(issues)
        mesh = run_mesh(case)
        n_c = mesh.grid.n_cells
        sess = cls(case=case, mesh=mesh, window=int(window))
        sess._p = np.zeros((0, n_c))
        sess._sw = np.zeros((0, n_c))
        sess._so = np.zeros((0, n_c))
        sess._sg = np.zeros((0, n_c))
        # 预热惰性 scipy/numpy 导入（克里金里的 lu_factor/lu_solve 首次调用约
        # 1.3 s），让第一拍 STEP 也能在亚秒内回 ACK，不拖慢联调。值用非零的
        # 占位（全零会命中 var==0 的早退分支，不起预热作用）。
        if len(case.probes):
            n_p, n_w = len(case.probes), len(case.wells)
            interpolate_pressure(
                mesh.grid,
                case.probe_xyz,
                np.linspace(1.0, 2.0, n_p),
                case.well_xyz,
                np.linspace(1.0, 2.0, n_w),
                method="kriging",
            )
        return sess

    def step(
        self,
        time_s: float,
        pressure: NDArray[np.float64],
        sw: NDArray[np.float64],
        so: NDArray[np.float64],
        sg: NDArray[np.float64],
        well_pw: NDArray[np.float64],
        well_q: NDArray[np.float64],
        well_qw: NDArray[np.float64],
        well_qo: NDArray[np.float64],
        well_qg: NDArray[np.float64],
    ) -> ProgramFields:
        # Mutate observations and windowed fields together under the lock so the
        # background worker always snapshots a consistent state.
        with self._lock:
            self.case.append_step(
                time_s, pressure, sw, so, sg, well_pw, well_q, well_qw, well_qo, well_qg
            )
            if self.window > 0:
                self.case.trim_to(self.window)
            self._interpolate_latest()
        self._request_invert()
        return self.snapshot_fields()

    def snapshot_fields(self) -> ProgramFields:
        with self._lock:
            return self._build_fields()

    def _build_fields(self) -> ProgramFields:
        n_t = int(self._p.shape[0])
        n_c = self.mesh.grid.n_cells
        phi = self._phi if self._phi is not None else np.full(n_c, float(self.case.phi0))
        k = self._k if self._k is not None else np.full(n_c, float(self.case.k0))
        return ProgramFields(
            mesh=self.mesh,
            times=np.array(self.case.times, dtype=float, copy=True),
            p=self._p.copy(),
            sw=self._sw.copy(),
            so=self._so.copy(),
            sg=self._sg.copy(),
            phi=np.repeat(phi[None, :], n_t, axis=0),
            k=np.repeat(k[None, :], n_t, axis=0),
            diagnostics=dict(self._diag),
        )

    def _interpolate_latest(self) -> None:
        """Interpolate and smooth only the newest time (caller holds ``_lock``)."""
        t = int(self.case.times.size) - 1
        grid = self.mesh.grid
        if self._pressure_method is None:
            self._pressure_method = str(self.case.method).strip().lower()
            self._sat_method = str(self.case.method).strip().lower()
        p_new = interpolate_pressure(
            grid,
            self.case.probe_xyz,
            self.case.pressure[t],
            self.case.well_xyz,
            self.case.well_pw[t],
            method=self._pressure_method,
        )
        sw_obs, so_obs, sg_obs = _saturation_slice(self.case, t)
        sw_new, so_new, sg_new = interpolate_saturation(
            grid,
            self.case.probe_xyz,
            sw_obs,
            so_obs,
            sg_obs,
            method=self._sat_method,
            swc=self.case.black_oil.swc,
            sgc=self.case.black_oil.sgc,
        )
        self._p = np.vstack([self._p, p_new[None, :]])
        self._sw = np.vstack([self._sw, sw_new[None, :]])
        self._so = np.vstack([self._so, so_new[None, :]])
        self._sg = np.vstack([self._sg, sg_new[None, :]])
        if self.window > 0 and self._p.shape[0] > self.window:
            self._p = self._p[-self.window :]
            self._sw = self._sw[-self.window :]
            self._so = self._so[-self.window :]
            self._sg = self._sg[-self.window :]
        _smooth_last(self._p)
        _smooth_last(self._sw)
        _smooth_last(self._so)
        _smooth_last(self._sg)
        last = int(self._p.shape[0]) - 1
        self._sw[last], self._so[last], self._sg[last] = project_saturations3(
            self._sw[last], self._so[last], self._sg[last]
        )

    def _request_invert(self) -> None:
        with self._lock:
            self._pending = True
            running = self._worker is not None and self._worker.is_alive()
        if not running:
            self._worker = threading.Thread(target=self._invert_loop, daemon=True)
            self._worker.start()

    def _invert_loop(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    return
                self._pending = False
                snapshot = self._snapshot()
            phi, k, diag = self._run_inversion(snapshot)
            with self._lock:
                self._phi = phi
                self._k = k
                self._diag = diag.as_dict()

    def _snapshot(self) -> dict:
        return {
            "p": self._p.copy(),
            "sw": self._sw.copy(),
            "so": self._so.copy(),
            "sg": self._sg.copy(),
            "times": np.array(self.case.times, dtype=float, copy=True),
            "well_q": self.case.well_q.copy(),
            "well_qw": self.case.well_qw.copy(),
            "well_qo": self.case.well_qo.copy(),
            "well_qg": self.case.well_qg.copy(),
            "well_pw": self.case.well_pw.copy(),
        }

    def _run_inversion(self, snap: dict):
        case = self.case
        grid = self.mesh.grid
        p, sw, so, sg = snap["p"], snap["sw"], snap["so"], snap["sg"]
        times = snap["times"]
        if case.k_homogeneous:
            # 岩石均质：跳过病态 k 反演，直接用常数 k0/phi0（与离线
            # ``pipeline._invert_static`` 一致）。
            n_c = grid.n_cells
            diag = RockDiagnostics()
            diag.k_mean_error = 0.0
            diag.phi_mean_error = 0.0
            diag.converged = True
            return np.full(n_c, float(case.phi0)), np.full(n_c, float(case.k0)), diag
        if case.rock_model == "black_oil_3phase":
            tw = transient_weights(times) if case.transient else None
            return invert_rock_three_phase(
                grid,
                p,
                sw,
                so,
                sg,
                times,
                self.mesh.probes,
                self.mesh.wells,
                snap["well_qw"],
                snap["well_qo"],
                snap["well_qg"],
                phi0=case.phi0,
                k0=case.k0,
                params=case.black_oil,
                relperm=case.relperm,
                time_weights=tw,
                fractional_weight=case.fractional_weight,
                well_bhp=snap["well_pw"],
                well_params=case.well,
                smoothness=case.k_smoothness,
            )
        return invert_rock(
            grid,
            p,
            sw,
            so,
            sg,
            times,
            self.mesh.probes,
            self.mesh.wells,
            snap["well_q"],
            phi0=case.phi0,
            k0=case.k0,
            params=case.black_oil,
            well_bhp=snap["well_pw"],
            well_params=case.well,
        )
