"""Raw sensor signal → saturation observation. Independent of the DPDP solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ObservationSeries


@dataclass
class SaturationObservation:
    """One inverted saturation sample with its own uncertainty."""

    value: float
    sigma: float
    x: float
    y: float
    z: float
    t: float
    sensor_name: str = ""


class SignalInversionModel:
    def invert(self, raw: NDArray[np.float64], *, x: float, y: float, z: float, times_s: NDArray[np.float64], name: str) -> list[SaturationObservation]:
        raise NotImplementedError


@dataclass
class LinearSaturationMap(SignalInversionModel):
    """Sw = a * raw + b. σ is supplied, never invented as zero."""

    a: float = 1.0
    b: float = 0.0
    sigma: float = 0.04

    def invert(self, raw, *, x, y, z, times_s, name) -> list[SaturationObservation]:
        raw = np.asarray(raw, dtype=float).ravel()
        times_s = np.asarray(times_s, dtype=float).ravel()
        if raw.size != times_s.size:
            raise ValueError("raw signal length must match times")
        sig = max(float(self.sigma), 1.0e-12)
        out = []
        for t, v in zip(times_s, raw):
            sw = float(np.clip(self.a * float(v) + self.b, 0.0, 1.0))
            out.append(SaturationObservation(sw, sig, float(x), float(y), float(z), float(t), name))
        return out


def observations_from_saturation(samples: list[SaturationObservation]) -> ObservationSeries:
    if not samples:
        raise ValueError("no saturation samples")
    name = samples[0].sensor_name or "S"
    return ObservationSeries(
        name,
        "saturation",
        np.array([s.t for s in samples], dtype=float),
        np.array([s.value for s in samples], dtype=float),
        np.array([s.sigma for s in samples], dtype=float),
        False,
    )
