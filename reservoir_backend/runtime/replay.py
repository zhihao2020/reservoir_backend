"""Replay an experiments/EXP00N directory through TwinRuntime."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reservoir_backend.io.case import load_case
from reservoir_backend.runtime.twin_runtime import TwinRuntime


def _read_wide(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: list[dict[str, Any]] = []
    for row in rows:
        t = float(row["time_s"])
        for key, val in row.items():
            if key == "time_s" or val in (None, ""):
                continue
            out.append({"time_s": t, "sensor_id": key, "value": float(val)})
    out.sort(key=lambda r: (r["time_s"], r["sensor_id"]))
    return out


def replay_experiment(folder: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    folder = Path(folder)
    twin = load_case(folder / "case.yaml")
    dest = Path(output or folder / "results")
    runtime = TwinRuntime(twin, field_folder=dest / "fields")
    ctrl_path = folder / "controls.csv"
    if ctrl_path.is_file():
        for row in csv.DictReader(ctrl_path.open(encoding="utf-8")):
            runtime.update_control(row["port"], row["kind"], float(row["value"]), float(row["time_s"]))
    sensors = {s.name: s for s in twin.experiment.sensors}
    events: list[dict[str, Any]] = []
    for row in _read_wide(folder / "pressure.csv"):
        row["kind"] = "pressure"
        row["sigma"] = float(sensors[row["sensor_id"]].sigma) if row["sensor_id"] in sensors else 2.0e3
        events.append(row)
    for row in _read_wide(folder / "saturation.csv"):
        row["kind"] = "sw"
        row["sigma"] = float(sensors[row["sensor_id"]].sigma) if row["sensor_id"] in sensors else 0.03
        events.append(row)
    events.sort(key=lambda r: r["time_s"])
    n_obs = 0
    for ev in events:
        runtime.observe(
            sensor_id=ev["sensor_id"],
            kind=ev["kind"],
            value=ev["value"],
            sigma=ev["sigma"],
            time_s=ev["time_s"],
        )
        n_obs += 1
    snap = None
    if n_obs:
        try:
            snap = runtime.request_field(time_s=events[-1]["time_s"])
        except ValueError:
            snap = None
    report = {
        "experiment": str(folder),
        "n_controls": len(twin.experiment.controls),
        "n_observations_appended": n_obs,
        "snapshot": snap,
        "notes": runtime.notes[-20:],
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
