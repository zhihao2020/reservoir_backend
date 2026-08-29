"""Concept-lab sensors: one (x, y, z) maps to exactly one kind."""

from __future__ import annotations

from typing import Any

XY_CM = (5.0, 10.0, 15.0, 20.0, 25.0)
LAYERS = ("bottom", "interface", "top")
LAYER_OFFSET_CM = {"bottom": 0.0, "interface": 8.0, "top": 16.0}
P_SIGMA_PA = 2.0e3
S_SIGMA = 0.04
PROBE_DIAMETER_M = 0.006

# 75 mm lattice, not on the resistivity 5×5.
ACOUSTIC_XY_CM = (
    (7.5, 7.5),
    (7.5, 22.5),
    (22.5, 7.5),
    (22.5, 22.5),
    (15.0, 7.5),
)
ACOUSTIC_Z_CM = (7.5, 22.5)


def z_horizon_cm(x_cm: float) -> float:
    return 3.0 + 8.0 * (1.0 - abs(float(x_cm) - 15.0) / 15.0)


def _xyz_key(x_m: float, y_m: float, z_m: float) -> tuple[int, int, int]:
    return (round(x_m * 1.0e9), round(y_m * 1.0e9), round(z_m * 1.0e9))


def resistivity_sensors() -> list[dict[str, Any]]:
    sensors: list[dict[str, Any]] = []
    for layer in LAYERS:
        off = LAYER_OFFSET_CM[layer]
        for iy, y_cm in enumerate(XY_CM):
            for ix, x_cm in enumerate(XY_CM):
                z_cm = z_horizon_cm(x_cm) + off
                kind = "pressure" if (ix + iy) % 2 == 0 else "saturation"
                tag = "p" if kind == "pressure" else "s"
                sensors.append(
                    {
                        "name": f"R_{layer}_{ix}{iy}_{tag}",
                        "kind": kind,
                        "source": "resistivity",
                        "layer": layer,
                        "x_m": x_cm / 100.0,
                        "y_m": y_cm / 100.0,
                        "z_m": z_cm / 100.0,
                        "sigma": P_SIGMA_PA if kind == "pressure" else S_SIGMA,
                    }
                )
    return sensors


def acoustic_sensors() -> list[dict[str, Any]]:
    sensors: list[dict[str, Any]] = []
    n = 0
    for x_cm, y_cm in ACOUSTIC_XY_CM:
        for z_cm in ACOUSTIC_Z_CM:
            n += 1
            sensors.append(
                {
                    "name": f"A_{n:02d}_s",
                    "kind": "saturation",
                    "source": "acoustic",
                    "layer": None,
                    "x_m": x_cm / 100.0,
                    "y_m": y_cm / 100.0,
                    "z_m": z_cm / 100.0,
                    "sigma": S_SIGMA,
                }
            )
    if len(sensors) != 10:
        raise ValueError(f"expected 10 acoustic sensors, got {len(sensors)}")
    return sensors


def all_sensors() -> list[dict[str, Any]]:
    sensors = resistivity_sensors() + acoustic_sensors()
    seen: dict[tuple[int, int, int], str] = {}
    for s in sensors:
        key = _xyz_key(float(s["x_m"]), float(s["y_m"]), float(s["z_m"]))
        prev = seen.get(key)
        if prev is not None:
            raise ValueError(f"duplicate xyz {s['name']} vs {prev}")
        seen[key] = str(s["name"])
        kinds = {t["kind"] for t in sensors if _xyz_key(t["x_m"], t["y_m"], t["z_m"]) == key}
        if len(kinds) != 1:
            raise ValueError(f"{s['name']} has mixed kinds at one xyz")
    n_p = sum(1 for s in sensors if s["kind"] == "pressure")
    n_s = sum(1 for s in sensors if s["kind"] == "saturation")
    if n_p != 39 or n_s != 46:
        raise ValueError(f"expected 39 p + 46 Sw, got {n_p} p + {n_s} Sw")
    return sensors
