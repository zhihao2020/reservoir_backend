"""Experiment domain: controls, observations, state, sensors. No PDE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Sensor:
    """Measurement geometry. Coordinates are SI metres, independent of the grid."""

    name: str
    kind: str  # pressure | saturation | oil_saturation | gas_saturation | phase_rate | bhp | q_oil | q_gas | q_inj
    # acoustic / em / resistivity alias saturation once xyz exists; do not invent coordinates.
    x: float
    y: float
    z: float
    volume_m3: float = 0.0
    probe_diameter_m: float = 0.0
    port_name: str | None = None
    sigma: float = 0.0
    medium: str = "fracture"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind in {"p", "pressure"}:
            kind = "pressure"
        elif kind in {
            "s",
            "sw",
            "saturation",
            "water_saturation",
            "acoustic",
            "em",
            "electromagnetic",
            "resistivity",
        }:
            kind = "saturation"
        elif kind in {"so", "oil", "oil_saturation"}:
            kind = "oil_saturation"
        elif kind in {"sg", "gas", "gas_saturation"}:
            kind = "gas_saturation"
        elif kind in {"qw", "phase_rate", "water_rate", "rate"}:
            kind = "phase_rate"
        elif kind in {"bhp", "pwf", "well_pressure", "bottomhole"}:
            kind = "bhp"
        elif kind in {"q_oil", "oil_rate", "qo"}:
            kind = "q_oil"
        elif kind in {"q_gas", "gas_rate", "qg"}:
            kind = "q_gas"
        elif kind in {"q_inj", "inj_rate", "injection_rate"}:
            kind = "q_inj"
        else:
            raise ValueError(f"unsupported sensor kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        medium = str(self.medium).strip().lower()
        if medium in {"f", "frac", "fracture"}:
            medium = "fracture"
        elif medium in {"m", "mat", "matrix"}:
            medium = "matrix"
        elif medium in {"b", "bulk", "both", "pv"}:
            medium = "bulk"
        else:
            raise ValueError(f"unsupported sensor medium: {self.medium}")
        object.__setattr__(self, "medium", medium)
        if self.volume_m3 < 0.0:
            raise ValueError("sensor volume_m3 must be >= 0")
        if self.probe_diameter_m < 0.0:
            raise ValueError("sensor probe_diameter_m must be >= 0")


def column_sensors(
    prefix: str,
    kind: str,
    x: float,
    y: float,
    depths_z: list[float] | tuple[float, ...] | NDArray[np.float64],
    *,
    sigma: float,
    volume_m3: float = 0.0,
    probe_diameter_m: float = 0.0,
    labels: list[str] | tuple[str, ...] | None = None,
    port_name: str | None = None,
) -> list[Sensor]:
    """Gauges on one (x, y) column at different z (lab height or field TVD).

    Depths do not have to sit on grid planes. ``H`` interpolates.
    """
    zs = [float(z) for z in np.asarray(depths_z, dtype=float).ravel()]
    if not zs:
        raise ValueError("column_sensors needs at least one depth")
    if labels is None:
        tags = [f"{i}" for i in range(len(zs))]
    else:
        tags = [str(s) for s in labels]
        if len(tags) != len(zs):
            raise ValueError("labels must match depths_z")
    return [
        Sensor(
            name=f"{prefix}_{tag}",
            kind=kind,
            x=float(x),
            y=float(y),
            z=z,
            volume_m3=volume_m3,
            probe_diameter_m=float(probe_diameter_m),
            port_name=port_name,
            sigma=float(sigma),
        )
        for tag, z in zip(tags, zs)
    ]


@dataclass(frozen=True)
class ControlSeries:
    """A single control channel applied to a port. SI values, times in seconds."""

    port_name: str
    kind: str  # rate | pressure | composition
    times_s: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"rate", "pressure", "composition", "gas_composition"}:
            raise ValueError(f"unsupported control kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        times = np.asarray(self.times_s, dtype=float).ravel()
        vals = np.asarray(self.values, dtype=float).ravel()
        if times.size != vals.size or times.size == 0:
            raise ValueError("control times and values must be non-empty and aligned")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("control times must be non-decreasing")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "values", vals)

    def value_at(self, t: float) -> float:
        """Piecewise-constant hold of the last sample at or before ``t``."""
        t = float(t)
        idx = int(np.searchsorted(self.times_s, t, side="right") - 1)
        idx = max(0, min(idx, self.times_s.size - 1))
        return float(self.values[idx])


@dataclass(frozen=True)
class ObservationSeries:
    """A single observation channel. SI values, times in seconds."""

    sensor_name: str
    kind: str  # same aliases as Sensor (acoustic/em/resistivity -> saturation)
    times_s: NDArray[np.float64]
    values: NDArray[np.float64]
    sigma: NDArray[np.float64]
    holdout: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind in {"p", "pressure"}:
            kind = "pressure"
        elif kind in {
            "s",
            "sw",
            "saturation",
            "water_saturation",
            "acoustic",
            "em",
            "electromagnetic",
            "resistivity",
        }:
            kind = "saturation"
        elif kind in {"so", "oil", "oil_saturation"}:
            kind = "oil_saturation"
        elif kind in {"sg", "gas", "gas_saturation"}:
            kind = "gas_saturation"
        elif kind in {"qw", "phase_rate", "water_rate", "rate"}:
            kind = "phase_rate"
        elif kind in {"bhp", "pwf", "well_pressure", "bottomhole"}:
            kind = "bhp"
        elif kind in {"q_oil", "oil_rate", "qo"}:
            kind = "q_oil"
        elif kind in {"q_gas", "gas_rate", "qg"}:
            kind = "q_gas"
        elif kind in {"q_inj", "inj_rate", "injection_rate"}:
            kind = "q_inj"
        else:
            raise ValueError(f"unsupported observation kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        times = np.asarray(self.times_s, dtype=float).ravel()
        vals = np.asarray(self.values, dtype=float).ravel()
        sig = np.asarray(self.sigma, dtype=float).ravel()
        if sig.size == 1:
            sig = np.full(vals.size, float(sig[0]))
        if times.size != vals.size or times.size != sig.size or times.size == 0:
            raise ValueError("observation times, values, and sigma must align")
        if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
            raise ValueError("observation sigma must be positive and finite")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "sigma", sig)


@dataclass
class State:
    """Dynamic state on the simulation grid. Fields are ``(n_cells,)``."""

    pressure: NDArray[np.float64]
    sw: NDArray[np.float64]
    sg: NDArray[np.float64] | None = None
    rs: NDArray[np.float64] | None = None
    moles: NDArray[np.float64] | None = None
    time_s: float = 0.0
    pressure_matrix: NDArray[np.float64] | None = None
    sw_matrix: NDArray[np.float64] | None = None
    sg_matrix: NDArray[np.float64] | None = None
    phi_fracture: NDArray[np.float64] | None = None
    phi_matrix: NDArray[np.float64] | None = None

    def so(self) -> NDArray[np.float64]:
        sg = self.sg if self.sg is not None else 0.0
        return 1.0 - np.asarray(self.sw, dtype=float) - np.asarray(sg, dtype=float)

    def copy(self) -> State:
        return State(
            pressure=np.asarray(self.pressure, dtype=float).copy(),
            sw=np.asarray(self.sw, dtype=float).copy(),
            sg=None if self.sg is None else np.asarray(self.sg, dtype=float).copy(),
            rs=None if self.rs is None else np.asarray(self.rs, dtype=float).copy(),
            moles=None if self.moles is None else np.asarray(self.moles, dtype=float).copy(),
            time_s=float(self.time_s),
            pressure_matrix=None if self.pressure_matrix is None else np.asarray(self.pressure_matrix, dtype=float).copy(),
            sw_matrix=None if self.sw_matrix is None else np.asarray(self.sw_matrix, dtype=float).copy(),
            sg_matrix=None if self.sg_matrix is None else np.asarray(self.sg_matrix, dtype=float).copy(),
            phi_fracture=None if self.phi_fracture is None else np.asarray(self.phi_fracture, dtype=float).copy(),
            phi_matrix=None if self.phi_matrix is None else np.asarray(self.phi_matrix, dtype=float).copy(),
        )


@dataclass
class Experiment:
    """Laboratory experiment: geometry, ports, controls, observations."""

    size_m: tuple[float, float, float] = (0.30, 0.30, 0.30)
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sensors: list[Sensor] = field(default_factory=list)
    controls: list[ControlSeries] = field(default_factory=list)
    observations: list[ObservationSeries] = field(default_factory=list)
    history_end_s: float | None = None

    def sensor_map(self) -> dict[str, Sensor]:
        return {s.name: s for s in self.sensors}

    def assimilate_observations(self) -> list[ObservationSeries]:
        return [o for o in self.observations if not o.holdout]

    def holdout_observations(self) -> list[ObservationSeries]:
        return [o for o in self.observations if o.holdout]

    def all_times_s(self) -> NDArray[np.float64]:
        chunks = [c.times_s for c in self.controls] + [o.times_s for o in self.observations]
        if not chunks:
            return np.zeros(0, dtype=float)
        return np.unique(np.concatenate(chunks))
