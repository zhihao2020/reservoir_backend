"""Concept-lab sensors (xlsx coords). Does not run a full invert."""

from __future__ import annotations

import csv
from pathlib import Path

from reservoir_backend.io.case import load_case


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "lab" / "lab_concept.yaml"
CSV = ROOT / "examples" / "lab" / "concept_probes.csv"


def test_concept_probes_csv_counts() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    n_r = sum(1 for r in rows if r["source"] == "resistivity")
    n_g = sum(1 for r in rows if r["source"] == "added_7p5cm")
    assert n_r == 75
    assert n_g == 16
    assert len(rows) == 91


def test_concept_sensors_load_count_and_xyz() -> None:
    twin = load_case(CASE)
    sensors = twin.experiment.sensors
    assert len(sensors) == 91
    by = {s.name: s for s in sensors}
    assert len(by) == 91

    r1 = by["R_bottom_01"]
    assert r1.kind == "saturation"
    assert abs(r1.x - 0.05) < 1.0e-9
    assert abs(r1.y - 0.05) < 1.0e-9
    assert abs(r1.z - 0.056667) < 1.0e-9

    r2 = by["R_bottom_02"]
    assert abs(r2.x - 0.05) < 1.0e-9
    assert abs(r2.y - 0.10) < 1.0e-9
    assert abs(r2.z - 0.056667) < 1.0e-9

    r13 = by["R_bottom_13"]
    assert abs(r13.x - 0.15) < 1.0e-9
    assert abs(r13.y - 0.15) < 1.0e-9
    assert abs(r13.z - 0.11) < 1.0e-9

    r26 = by["R_interface_26"]
    assert abs(r26.x - 0.05) < 1.0e-9
    assert abs(r26.z - 0.136667) < 1.0e-9

    r75 = by["R_top_75"]
    assert abs(r75.x - 0.25) < 1.0e-9
    assert abs(r75.y - 0.25) < 1.0e-9
    assert abs(r75.z - 0.216667) < 1.0e-9

    g1 = by["G7_01"]
    assert abs(g1.x - 0.075) < 1.0e-9
    assert abs(g1.y - 0.075) < 1.0e-9
    assert abs(g1.z - 0.07) < 1.0e-9

    assert all(abs(a - 0.3) < 1.0e-12 for a in twin.grid.size_m())
    assert twin.grid.nx == 30 and twin.grid.ny == 30 and twin.grid.nz == 30
    roles = {p.name: (p.role, p.control) for p in twin.ports}
    assert roles["INJ"] == ("injector", "rate")
    assert roles["PROD"] == ("producer", "pressure")

