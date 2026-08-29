"""Probe coordinates + p(t) in, batch invert K, full-grid pressure p(t) out.

Wrapper around ``DigitalTwin.calibrate`` (LM on θ) and ``DigitalTwin.simulate``.
Not a 1 Hz online invert. After K is known the same forward reports cell
pressure at the requested times. ``invert --output`` / ``apply --output`` write
a last-time snapshot; this module writes p(t) with shape (n_times, n_cells).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.domain.types import ObservationSeries, Sensor, State
from reservoir_backend.io.case import load_case
from reservoir_backend.observation.operator import ObservationOperator
from reservoir_backend.twin.offline import DigitalTwin


@dataclass
class PressureField:
    """Full-grid pressure at report times. ``pressure`` is (n_times, n_cells)."""

    times_s: NDArray[np.float64]
    pressure: NDArray[np.float64]
    k: NDArray[np.float64]
    posterior: Any | None = None
    I: NDArray[np.int64] | None = None
    xyz: NDArray[np.float64] | None = None
    sw: NDArray[np.float64] | None = None
    so: NDArray[np.float64] | None = None
    sg: NDArray[np.float64] | None = None
    phi: float | None = None

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "times_s.npy", self.times_s)
        np.save(folder / "pressure.npy", self.pressure)
        np.save(folder / "k.npy", self.k)
        packed: dict[str, Any] = {
            "times_s": self.times_s,
            "pressure": self.pressure,
            "k": self.k,
        }
        if self.I is not None:
            np.save(folder / "I.npy", self.I)
            packed["I"] = self.I
        if self.xyz is not None:
            np.save(folder / "xyz.npy", self.xyz)
            packed["xyz"] = self.xyz
        if self.sw is not None:
            np.save(folder / "sw.npy", self.sw)
            packed["sw"] = self.sw
        if self.so is not None:
            np.save(folder / "so.npy", self.so)
            packed["so"] = self.so
        if self.sg is not None:
            np.save(folder / "sg.npy", self.sg)
            packed["sg"] = self.sg
        if self.phi is not None:
            packed["phi"] = np.asarray(self.phi, dtype=float)
            np.save(folder / "phi.npy", packed["phi"])
        np.savez(folder / "pressure_field.npz", **packed)
        lines = ["time_s"]
        lines.extend(f"{float(t):.9g}" for t in self.times_s)
        (folder / "times_s.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write_csv(folder / "field.csv")

    def write_csv(self, path: str | Path) -> None:
        """Stacked rows: I,x,y,z,p[,sw,so,sg],k[,phi],time_s."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pressure = np.asarray(self.pressure, dtype=float)
        n_t, n_cells = pressure.shape
        cell_I = self.I if self.I is not None else np.arange(n_cells, dtype=np.int64)
        xyz = self.xyz if self.xyz is not None else np.full((n_cells, 3), np.nan)
        times = np.asarray(self.times_s, dtype=float).ravel()
        header = ["I", "x", "y", "z", "p"]
        if self.sw is not None:
            header.append("sw")
        if self.so is not None:
            header.append("so")
        if self.sg is not None:
            header.append("sg")
        header.append("k")
        if self.phi is not None:
            header.append("phi")
        header.append("time_s")
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for it in range(n_t):
                p_row = pressure[it].ravel()
                sw_row = None if self.sw is None else np.asarray(self.sw[it], dtype=float).ravel()
                so_row = None if self.so is None else np.asarray(self.so[it], dtype=float).ravel()
                sg_row = None if self.sg is None else np.asarray(self.sg[it], dtype=float).ravel()
                for ic in range(n_cells):
                    row = [
                        int(cell_I[ic]),
                        f"{float(xyz[ic, 0]):.16g}",
                        f"{float(xyz[ic, 1]):.16g}",
                        f"{float(xyz[ic, 2]):.16g}",
                        f"{float(p_row[ic]):.16g}",
                    ]
                    if sw_row is not None:
                        row.append(f"{float(sw_row[ic]):.16g}")
                    if so_row is not None:
                        row.append(f"{float(so_row[ic]):.16g}")
                    if sg_row is not None:
                        row.append(f"{float(sg_row[ic]):.16g}")
                    row.append(f"{float(self.k[ic]):.16g}")
                    if self.phi is not None:
                        row.append(f"{float(self.phi):.16g}")
                    row.append(f"{float(times[it]):.16g}")
                    writer.writerow(row)


def _as_twin(case: str | Path | DigitalTwin) -> DigitalTwin:
    if isinstance(case, DigitalTwin):
        return case
    return load_case(case)


def _parse_probe(item: Sensor | Sequence[Any] | Mapping[str, Any], index: int) -> Sensor:
    if isinstance(item, Sensor):
        return item
    if isinstance(item, Mapping):
        return Sensor(
            name=str(item.get("name") or f"P{index}"),
            kind=str(item.get("kind") or "pressure"),
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
            volume_m3=float(item.get("volume_m3") or 0.0),
            probe_diameter_m=float(item.get("probe_diameter_m") or 0.0),
            sigma=float(item.get("sigma") or 0.0),
        )
    seq = list(item)
    if len(seq) == 3:
        name = f"P{index}"
        x, y, z = float(seq[0]), float(seq[1]), float(seq[2])
    elif len(seq) >= 4:
        name = str(seq[0])
        x, y, z = float(seq[1]), float(seq[2]), float(seq[3])
    else:
        raise ValueError("probe must be Sensor, (name, x, y, z), (x, y, z), or a dict")
    return Sensor(name=name, kind="pressure", x=x, y=y, z=z)


def _read_probes_csv(path: Path) -> list[Sensor]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"probes CSV is empty: {path}")
    out: list[Sensor] = []
    for i, row in enumerate(rows):
        name = str(row.get("name") or row.get("sensor") or f"P{i}").strip()
        out.append(
            Sensor(
                name=name,
                kind=str(row.get("kind") or "pressure"),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                probe_diameter_m=float(row.get("probe_diameter_m") or 0.0),
                sigma=float(row.get("sigma") or 0.0),
            )
        )
    return out



def _observation_kind(raw: object, sensor_name: str, sensors: Sequence[Sensor]) -> str:
    """Use an explicit series kind, else the matching sensor kind, else pressure."""
    text = str(raw).strip() if raw not in (None, "") else ""
    if text:
        return text
    for s in sensors:
        if s.name == sensor_name:
            return s.kind
    return "pressure"


def attach_probe_series(
    twin: DigitalTwin,
    probes: Sequence[Sensor | Sequence[Any] | Mapping[str, Any]] | None = None,
    series: str | Path | Mapping[str, Any] | Sequence[ObservationSeries] | None = None,
    *,
    sigma: float = 2.0e3,
) -> None:
    """Put probe xyz and/or p(t) onto ``twin.experiment``. Rebuilds H if probes change."""
    if probes is not None:
        sensors = [_parse_probe(p, i) for i, p in enumerate(probes)]
        twin.experiment.sensors = sensors
        twin.operator = ObservationOperator(twin.grid, sensors, twin.ports)

    if series is None:
        return

    if isinstance(series, (str, Path)):
        from reservoir_backend.io.case import _read_observation_csv

        rows = _read_observation_csv(Path(series))
        obs = [
            ObservationSeries(
                sensor_name=str(o["sensor"]),
                kind=_observation_kind(o.get("kind"), str(o["sensor"]), twin.experiment.sensors),
                times_s=np.asarray(o["times"], dtype=float),
                values=np.asarray(o["values"], dtype=float),
                sigma=np.asarray(o.get("sigma", sigma), dtype=float),
                holdout=bool(o.get("holdout", False)),
            )
            for o in rows
        ]
        known = {s.name for s in twin.experiment.sensors}
        unknown = sorted({o.sensor_name for o in obs} - known)
        if unknown:
            raise ValueError(f"series names sensors not on the twin: {unknown}")
        twin.experiment.observations = obs
        return

    if isinstance(series, Sequence) and not isinstance(series, (str, bytes, Mapping)):
        if series and isinstance(series[0], ObservationSeries):
            twin.experiment.observations = list(series)
            return

    if not isinstance(series, Mapping):
        raise ValueError("series must be a CSV path, ObservationSeries list, or mapping")

    if "times_s" in series or "times" in series:
        times = np.asarray(series.get("times_s", series.get("times")), dtype=float).ravel()
        values = np.asarray(series["values"], dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] != times.size:
            if values.shape[1] == times.size:
                values = values.T
            else:
                raise ValueError("series values must be (n_times, n_probes)")
        sensors = twin.experiment.sensors
        if values.shape[1] != len(sensors):
            raise ValueError(
                f"series values have {values.shape[1]} columns, but {len(sensors)} probes"
            )
        sig = series.get("sigma", sigma)
        obs = []
        for j, s in enumerate(sensors):
            sig_j = np.asarray(sig, dtype=float)
            if sig_j.ndim == 2:
                sj = sig_j[:, j] if sig_j.shape[1] == values.shape[1] else sig_j[j, :]
            elif sig_j.size == times.size:
                sj = sig_j
            else:
                sj = np.full(times.size, float(sig_j.ravel()[0]) if sig_j.size else float(sigma))
            obs.append(
                ObservationSeries(
                    sensor_name=s.name,
                    kind=s.kind,
                    times_s=times,
                    values=values[:, j],
                    sigma=sj,
                )
            )
        twin.experiment.observations = obs
        return

    obs = []
    for name, payload in series.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"series[{name!r}] must be a mapping with times and values")
        times = np.asarray(payload.get("times_s", payload.get("times")), dtype=float)
        vals = np.asarray(payload["values"], dtype=float)
        sig = np.asarray(payload.get("sigma", sigma), dtype=float)
        kind = _observation_kind(payload.get("kind"), str(name), twin.experiment.sensors)
        obs.append(ObservationSeries(str(name), kind, times, vals, sig))
    twin.experiment.observations = obs


def _forward_pressure(
    twin: DigitalTwin,
    k: NDArray[np.float64],
    report_times: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    times = np.asarray(report_times, dtype=float).ravel()
    if times.size == 0:
        raise ValueError("report_times is empty")
    t_end = float(np.max(times))
    rock = twin.rock_from_k(np.asarray(k, dtype=float).ravel())
    traj = twin.simulate(rock, t_end=t_end, report_times=times)
    p_rows: list[NDArray[np.float64]] = []
    sw_rows: list[NDArray[np.float64]] = []
    so_rows: list[NDArray[np.float64]] = []
    sg_rows: list[NDArray[np.float64]] = []
    have_sg = True
    for t in times:
        st = traj.state_at(float(t))
        p_rows.append(np.asarray(st.pressure, dtype=float).ravel())
        sw_rows.append(np.asarray(st.sw, dtype=float).ravel())
        so_rows.append(np.asarray(st.so(), dtype=float).ravel())
        if st.sg is None:
            have_sg = False
        else:
            sg_rows.append(np.asarray(st.sg, dtype=float).ravel())
    sw = np.stack(sw_rows, axis=0)
    if have_sg and len(sg_rows) == times.size:
        sg = np.stack(sg_rows, axis=0)
        so = np.stack(so_rows, axis=0)
    else:
        sg = None
        so = None
    return times, np.stack(p_rows, axis=0), sw, so, sg


def pressure_field(
    case: str | Path | DigitalTwin,
    *,
    probes: Sequence[Any] | None = None,
    series: str | Path | Mapping[str, Any] | Sequence[ObservationSeries] | None = None,
    k: ArrayLike | None = None,
    posterior: Any | None = None,
    report_times: ArrayLike | None = None,
    output: str | Path | None = None,
    invert_kwargs: Mapping[str, Any] | None = None,
) -> PressureField:
    """Batch invert (or skip if ``k`` / ``posterior`` given), then full-grid p(t).

    Parameters
    ----------
    case:
        YAML path (grid / PVT / controls) or a loaded ``DigitalTwin``.
    probes:
        Probe coordinates: ``(name, x, y, z)``, ``(x, y, z)``, dict, or ``Sensor``.
        Omit to keep sensors already on the case.
    series:
        Pressure and/or known-Sw time series. CSV (``time_s,sensor,kind,value,sigma``), array
        mapping ``{"times_s": (n_t,), "values": (n_t, n_probes)}``, per-sensor
        dict, or ``ObservationSeries`` list. Omit to keep case observations.
    k:
        Cell permeability ``(n_cells,)``. Skips invert.
    posterior:
        Result of ``twin.calibrate()``. Skips invert; uses ``k``.
    report_times:
        Times (s) at which to dump the field. Default: unique observation times.
    output:
        If set, write npy/npz plus ``field.csv`` rows
        ``I,x,y,z,p[,sw,so,sg],k[,phi],time_s``.
    invert_kwargs:
        Forwarded to ``DigitalTwin.calibrate`` when invert runs.

    Invert is one LM fit over the series, not once per sample. After K
    is known the forward reports p on every cell at ``report_times``.
    """
    twin = _as_twin(case)
    if probes is not None or series is not None:
        attach_probe_series(twin, probes, series)

    post = posterior
    if k is not None:
        k_use = np.asarray(k, dtype=float).ravel()
    elif post is not None:
        k_use = np.asarray(post.k, dtype=float).ravel()
    else:
        if not twin.experiment.observations:
            raise ValueError(
                "pressure_field needs series (or case observations), or pass k / posterior"
            )
        post = twin.calibrate(**dict(invert_kwargs or {}))
        k_use = np.asarray(post.k, dtype=float).ravel()

    if k_use.size != twin.grid.n_cells:
        raise ValueError(f"k size {k_use.size} != n_cells {twin.grid.n_cells}")

    if report_times is None:
        chunks = [o.times_s for o in twin.experiment.observations if np.asarray(o.times_s).size]
        if not chunks:
            chunks = [c.times_s for c in twin.experiment.controls if np.asarray(c.times_s).size]
        if not chunks:
            raise ValueError("no report_times, observations, or controls")
        times = np.unique(np.concatenate([np.asarray(c, dtype=float).ravel() for c in chunks]))
    else:
        times = np.asarray(report_times, dtype=float).ravel()

    times, pressure, sw, so, sg = _forward_pressure(twin, k_use, times)
    result = PressureField(
        times_s=times,
        pressure=pressure,
        k=k_use,
        posterior=post,
        I=np.arange(twin.grid.n_cells, dtype=np.int64),
        xyz=np.asarray(twin.grid.cell_centers(), dtype=float),
        sw=sw,
        so=so,
        sg=sg,
        phi=float(getattr(twin.parameterization, "phi", 0.20)),
    )
    if output is not None:
        result.save(output)
    return result


def step_pressure(
    twin: DigitalTwin,
    k: ArrayLike,
    *,
    state: State | None = None,
    dt: float | None = None,
) -> State:
    """Advance one solver dt on an already-inverted twin; return the new state.

    Uses ``DigitalTwin.simulate`` from ``state`` (or IC) to ``t + dt`` with K
    held fixed. This is not a 1 Hz invert.
    """
    k_use = np.asarray(k, dtype=float).ravel()
    if k_use.size != twin.grid.n_cells:
        raise ValueError(f"k size {k_use.size} != n_cells {twin.grid.n_cells}")
    state0 = state.copy() if state is not None else twin.initial_state()
    step = float(twin.physics.dt_init if dt is None else dt)
    if step <= 0.0:
        raise ValueError("dt must be positive")
    t_end = float(state0.time_s) + step
    rock = twin.rock_from_k(k_use)
    traj = twin.simulate(
        rock,
        t_end=t_end,
        state0=state0,
        report_times=np.array([t_end], dtype=float),
    )
    return traj.states[-1]
