"""Bind a DigitalTwin / TwinLoops to control, observation, and field I/O.

UDP handlers enqueue work here. This module may run Newton; the socket
layer must not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.domain.types import ControlSeries, ObservationSeries
from reservoir_backend.observation.qc import classify_observations
from reservoir_backend.runtime.command_queue import CommandQueue
from reservoir_backend.runtime.field_store import FieldStore
from reservoir_backend.twin.loops import TwinLoops
from reservoir_backend.twin.offline import DigitalTwin, stack_observations


def _append_control(series: ControlSeries, time_s: float, value: float) -> ControlSeries:
    t = np.asarray(series.times_s, dtype=float)
    v = np.asarray(series.values, dtype=float)
    time_s = float(time_s)
    value = float(value)
    match = np.flatnonzero(np.abs(t - time_s) < 1.0e-12)
    if match.size:
        v = v.copy()
        v[int(match[-1])] = value
        return ControlSeries(series.port_name, series.kind, t, v)
    idx = int(np.searchsorted(t, time_s))
    return ControlSeries(
        series.port_name,
        series.kind,
        np.insert(t, idx, time_s),
        np.insert(v, idx, value),
    )


def _append_observation(series: ObservationSeries, time_s: float, value: float, sigma: float) -> ObservationSeries:
    t = np.asarray(series.times_s, dtype=float)
    v = np.asarray(series.values, dtype=float)
    s = np.broadcast_to(np.asarray(series.sigma, dtype=float), t.shape).copy()
    return ObservationSeries(
        series.sensor_name,
        series.kind,
        np.append(t, float(time_s)),
        np.append(v, float(value)),
        np.append(s, float(sigma)),
        holdout=bool(series.holdout),
    )


class TwinRuntime:
    """Experiment-facing twin service: controls, observations, snapshots."""

    def __init__(
        self,
        twin: DigitalTwin,
        *,
        loops: TwinLoops | None = None,
        field_folder: str | Path = "results/fields",
    ) -> None:
        self.twin = twin
        self.loops = loops
        self.queue = CommandQueue()
        self.fields = FieldStore(field_folder)
        self.running = True
        self.notes: list[str] = []
        self.last_observe_time_s = 0.0
        self.last_cf_update_s = 0.0
        self.last_full_time_s = 0.0 if loops is None else float(loops.last_slow_s)
        self.assimilated_times: set[tuple[str, float]] = set()

    def update_control(self, port: str, kind: str, value: float, time_s: float) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("runtime is stopped")
        names = {p.name for p in self.twin.ports}
        if str(port) not in names:
            raise ValueError(f"unknown port {port!r}")
        found = False
        new_ctrls: list[ControlSeries] = []
        for c in self.twin.experiment.controls:
            if c.port_name == port and c.kind == kind:
                new_ctrls.append(_append_control(c, time_s, value))
                found = True
            else:
                new_ctrls.append(c)
        if not found:
            new_ctrls.append(
                ControlSeries(str(port), str(kind), np.array([float(time_s)]), np.array([float(value)]))
            )
        self.twin.experiment.controls = new_ctrls
        if self.loops is not None:
            self.loops._frozen = None
        self.notes.append(f"control {port}/{kind}={value} at t={time_s}")
        return {"ok": True, "cmd": "update_control", "port": port, "kind": kind, "time_s": float(time_s)}

    def observe(
        self,
        *,
        sensor_id: str,
        kind: str,
        value: float,
        sigma: float,
        time_s: float,
    ) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("runtime is stopped")
        known = {s.name for s in self.twin.experiment.sensors}
        if str(sensor_id) not in known:
            raise ValueError(f"unknown sensor {sensor_id!r}")
        key = (str(sensor_id), float(time_s))
        reused = key in self.assimilated_times
        if reused:
            self.notes.append(f"skip reused observation {sensor_id} t={time_s}")
            return {"ok": True, "cmd": "observe", "reused": True, "sensor_id": sensor_id}
        self.assimilated_times.add(key)
        found = False
        new_obs: list[ObservationSeries] = []
        for o in self.twin.experiment.observations:
            if o.sensor_name == sensor_id:
                new_obs.append(_append_observation(o, time_s, value, sigma))
                found = True
            else:
                new_obs.append(o)
        if not found:
            new_obs.append(
                ObservationSeries(
                    str(sensor_id),
                    str(kind),
                    np.array([float(time_s)]),
                    np.array([float(value)]),
                    np.array([float(sigma)]),
                    False,
                )
            )
        self.twin.experiment.observations = new_obs
        self.last_observe_time_s = float(time_s)
        triggered = False
        if self.loops is not None:
            post = self.loops.maybe_slow(float(time_s), observations=new_obs)
            triggered = post is not None
            if triggered:
                self.last_full_time_s = float(time_s)
                self.last_cf_update_s = float(time_s)
        return {
            "ok": True,
            "cmd": "observe",
            "sensor_id": sensor_id,
            "time_s": float(time_s),
            "slow_triggered": bool(triggered),
            "reused": False,
        }

    def request_field(self, *, time_s: float | None = None) -> dict[str, Any]:
        t = float(self.last_observe_time_s if time_s is None else time_s)
        pressure_source = "fast"
        saturations_held = True
        st = None
        if self.loops is not None and self.loops.last_traj is not None:
            try:
                st = self.loops.fast_state(t)
                if abs(float(self.loops.last_slow_s) - t) < 1.0e-9:
                    pressure_source = "full"
                    saturations_held = False
            except Exception:
                st = self.loops.last_traj.states[-1] if self.loops.last_traj.states else None
                pressure_source = "full"
                saturations_held = False
        elif getattr(self.twin, "_last_dual", None) is not None and self.twin._last_dual is not None:
            from reservoir_backend.solver.fi_comp_dual import dual_to_state

            st = dual_to_state(self.twin.physics.fluid, self.twin._last_dual, self.twin._last_dual_rock)
            pressure_source = "full"
            saturations_held = False
        if st is None and self.twin.uses_dpdp() and self.twin.physics.fluid is not None:
            from reservoir_backend.solver.fi_comp_dual import dual_to_state, initialize_dual_state

            theta = np.asarray(self.twin.parameterization.prior_mean, dtype=float).ravel()
            rock = self.twin.dual_rock_from_theta(theta)
            dual0 = initialize_dual_state(self.twin.grid, rock, self.twin.physics.fluid, float(self.twin.physics.p_init))
            st = dual_to_state(self.twin.physics.fluid, dual0, rock)
            pressure_source = "full"
            saturations_held = False
            self.last_full_time_s = 0.0
        if st is None:
            raise ValueError("no field available; run a forward first")
        cf_mean = None
        cf_std = None
        if self.loops is not None and self.loops.members is not None:
            cfs = np.array(
                [cf for cf in (self.twin.parameterization.decode(self.loops.members[:, j])[0] for j in range(self.loops.members.shape[1]))],
                dtype=float,
            )
            cf_mean = float(np.mean(cfs))
            cf_std = float(np.std(cfs, ddof=1)) if cfs.size > 1 else 0.0
        payload = self.fields.write(
            {
                "pressure_fracture": st.pressure,
                "pressure_matrix": st.pressure_matrix,
                "sw": st.sw,
                "so": st.so(),
                "sg": st.sg,
                "Cf_mean": np.array([cf_mean if cf_mean is not None else np.nan]),
                "Cf_std": np.array([cf_std if cf_std is not None else np.nan]),
            },
            time_s=t,
            metadata={
                "pressure_source": pressure_source,
                "saturation_source": "last_full",
                "saturations_held": saturations_held,
                "last_full_time_s": self.last_full_time_s,
                "Cf_update_time_s": self.last_cf_update_s,
            },
        )
        return {"ok": True, "cmd": "request_field", **payload, "transport": "npz"}

    def qc_window(self, t_lo: float, t_hi: float) -> NDArray[np.str_]:
        from reservoir_backend.twin.offline import window_observations

        series = window_observations(self.twin.experiment.observations, t_lo, t_hi)
        if not series:
            return np.zeros(0, dtype=object)
        d = stack_observations(series)
        y = np.repeat(d.values.reshape(-1, 1), 2, axis=1)
        return classify_observations(y, d.values, d.sigma)

    def start(self) -> dict[str, Any]:
        self.running = True
        return {"ok": True, "cmd": "start"}

    def stop(self) -> dict[str, Any]:
        self.running = False
        return {"ok": True, "cmd": "stop"}

    def status(self) -> dict[str, Any]:
        n = 0
        t = None
        if self.loops is not None and self.loops.members is not None:
            n = int(self.loops.members.shape[1])
            t = float(self.loops.last_slow_s)
        return {
            "ok": True,
            "cmd": "status",
            "running": bool(self.running),
            "n_theta_ensemble": n,
            "time_s": t,
            "pressure_source": "fast",
            "saturation_source": "last_full",
            "saturations_held": True,
            "last_full_time_s": self.last_full_time_s,
            "Cf_update_time_s": self.last_cf_update_s,
            "queue_len": len(self.queue),
        }
