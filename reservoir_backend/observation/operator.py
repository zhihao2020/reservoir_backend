"""Observation operator H: state on the grid -> sensor values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import Sensor, State
from reservoir_backend.exceptions import InvalidObservation
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.ports.flow import FlowPort


def _trilinear(grid: CartesianGrid, field: NDArray[np.float64], x: float, y: float, z: float) -> float:
    """Trilinear interpolation on the cell-center lattice."""
    values = np.asarray(field, dtype=float).ravel()
    if values.size != grid.n_cells:
        raise InvalidObservation(f"field size {values.size} != n_cells {grid.n_cells}")
    cx, cy, cz = grid.center_axis("x"), grid.center_axis("y"), grid.center_axis("z")
    ijk = grid.reshape_ijk(values)

    def bracket(coord: float, axis: NDArray[np.float64]) -> tuple[int, int, float]:
        if axis.size == 1:
            return 0, 0, 0.0
        if coord <= axis[0]:
            return 0, 0, 0.0
        if coord >= axis[-1]:
            last = axis.size - 1
            return last, last, 0.0
        hi = int(np.searchsorted(axis, coord, side="right"))
        lo = hi - 1
        w = (coord - axis[lo]) / (axis[hi] - axis[lo])
        return lo, hi, float(w)

    i0, i1, wx = bracket(x, cx)
    j0, j1, wy = bracket(y, cy)
    k0, k1, wz = bracket(z, cz)
    c000 = ijk[k0, j0, i0]
    c100 = ijk[k0, j0, i1]
    c010 = ijk[k0, j1, i0]
    c110 = ijk[k0, j1, i1]
    c001 = ijk[k1, j0, i0]
    c101 = ijk[k1, j0, i1]
    c011 = ijk[k1, j1, i0]
    c111 = ijk[k1, j1, i1]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return float(c0 * (1.0 - wz) + c1 * wz)


def _sphere_offsets(radius_m: float) -> NDArray[np.float64]:
    """Deterministic stencil inside a sphere. Includes the center."""
    r = float(radius_m)
    if r <= 0.0:
        return np.zeros((1, 3), dtype=float)
    lin = np.linspace(-1.0, 1.0, 5)
    pts = np.array([(x, y, z) for x in lin for y in lin for z in lin], dtype=float) * r
    keep = np.sum(pts * pts, axis=1) <= r * r + 1.0e-16
    return pts[keep]


def _interpolated_average(
    grid: CartesianGrid,
    field: NDArray[np.float64],
    x: float,
    y: float,
    z: float,
    offsets: NDArray[np.float64],
) -> float:
    acc = 0.0
    for dx, dy, dz in offsets:
        acc += _trilinear(grid, field, x + float(dx), y + float(dy), z + float(dz))
    return float(acc / max(offsets.shape[0], 1))


def _volume_average(
    grid: CartesianGrid,
    field: NDArray[np.float64],
    x: float,
    y: float,
    z: float,
    volume_m3: float,
) -> float:
    """Average the interpolated field over a cube of volume ``volume_m3``."""
    side = float(volume_m3) ** (1.0 / 3.0)
    half = 0.5 * side
    lin = np.linspace(-half, half, 4)
    offsets = np.array([(dx, dy, dz) for dx in lin for dy in lin for dz in lin], dtype=float)
    return _interpolated_average(grid, field, x, y, z, offsets)


def _probe_average(
    grid: CartesianGrid,
    field: NDArray[np.float64],
    x: float,
    y: float,
    z: float,
    diameter_m: float,
) -> float:
    """Average the interpolated field over a sphere of the probe diameter."""
    return _interpolated_average(grid, field, x, y, z, _sphere_offsets(0.5 * float(diameter_m)))


@dataclass
class ObservationOperator:
    """Maps ``State`` (+ optional port rates) to a sensor reading."""

    grid: CartesianGrid
    sensors: list[Sensor]
    ports: list[FlowPort] | None = None

    def sample_field(self, sensor: Sensor, field: NDArray[np.float64]) -> float:
        diameter = float(getattr(sensor, "probe_diameter_m", 0.0) or 0.0)
        if diameter > 0.0:
            return _probe_average(self.grid, field, sensor.x, sensor.y, sensor.z, diameter)
        if sensor.volume_m3 > 0.0:
            return _volume_average(self.grid, field, sensor.x, sensor.y, sensor.z, sensor.volume_m3)
        return _trilinear(self.grid, field, sensor.x, sensor.y, sensor.z)

    def sample(
        self,
        sensor: Sensor,
        state: State,
        port_rates: dict[str, float] | None = None,
        port_bhp: dict[str, float] | None = None,
    ) -> float:
        if sensor.kind == "pressure":
            return self.sample_field(sensor, state.pressure)
        if sensor.kind == "saturation":
            return self.sample_field(sensor, state.sw)
        if sensor.kind == "oil_saturation":
            return self.sample_field(sensor, state.so())
        if sensor.kind == "gas_saturation":
            sg = state.sg if state.sg is not None else np.zeros_like(state.sw)
            return self.sample_field(sensor, np.asarray(sg, dtype=float))
        if sensor.kind == "phase_rate":
            if not sensor.port_name:
                raise InvalidObservation(f"phase_rate sensor {sensor.name} needs port_name")
            rates = port_rates or {}
            if sensor.port_name not in rates:
                raise InvalidObservation(
                    f"phase_rate sensor {sensor.name}: missing rate for port {sensor.port_name}"
                )
            return float(rates[sensor.port_name])
        if sensor.kind in {"q_oil", "q_gas", "q_inj"}:
            if not sensor.port_name:
                raise InvalidObservation(f"{sensor.kind} sensor {sensor.name} needs port_name")
            rates = port_rates or {}
            key = str(sensor.port_name) + ":" + sensor.kind
            if key in rates:
                return float(rates[key])
            if sensor.port_name in rates and sensor.kind == "phase_rate":
                return float(rates[sensor.port_name])
            raise InvalidObservation(
                f"{sensor.kind} sensor {sensor.name}: missing rate for port {sensor.port_name}"
            )
        if sensor.kind == "bhp":
            if not sensor.port_name:
                raise InvalidObservation(f"bhp sensor {sensor.name} needs port_name")
            bhps = port_bhp or {}
            if sensor.port_name not in bhps:
                raise InvalidObservation(
                    f"bhp sensor {sensor.name}: missing BHP for port {sensor.port_name}"
                )
            return float(bhps[sensor.port_name])
        raise InvalidObservation(f"unknown sensor kind {sensor.kind}")

    def vector(
        self,
        sensors: list[Sensor],
        state: State,
        port_rates: dict[str, float] | None = None,
        port_bhp: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        return np.asarray(
            [self.sample(s, state, port_rates=port_rates, port_bhp=port_bhp) for s in sensors],
            dtype=float,
        )
