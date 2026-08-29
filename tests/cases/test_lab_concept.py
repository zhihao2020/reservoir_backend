"""Concept-lab sensors (xlsx coords) and waterflood similarity keys.

Does not run a full invert. Does not import references/.
"""

from __future__ import annotations

import csv
from pathlib import Path

from reservoir_backend.io.case import load_case
from reservoir_backend.twin.similarity import REQUIRED_KEYS, attach_displacement, waterflood_groups


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


def test_similarity_report_required_keys() -> None:
    twin = load_case(CASE)
    rep = waterflood_groups(twin)
    for key in REQUIRED_KEYS:
        assert key in rep
    geo = rep["geometric"]
    assert all(abs(a - 0.3) < 1.0e-12 for a in geo["size_m"])
    assert geo["well_spacing_m"] is not None
    assert abs(float(geo["well_spacing_m"]) - 0.29) < 0.02
    assert geo["well_radius_m"] is None
    assert geo["size_ratio_lab_over_field"] is None
    assert geo["well_spacing_ratio_lab_over_field"] is None
    assert geo["well_radius_ratio_lab_over_field"] is None
    assert geo["field_ratios"] == "unknown"

    assert abs(float(rep["reservoir"]["phi"]) - 0.20) < 1.0e-12
    assert rep["reservoir"]["k_m2"] > 0.0
    assert abs(float(rep["fluid"]["mu_o_over_mu_w"]) - 5.0) < 1.0e-12
    assert abs(float(rep["fluid"]["rho_o_over_rho_w"]) - 0.8) < 1.0e-12
    assert abs(float(rep["saturation"]["swc"]) - 0.20) < 1.0e-12
    assert abs(float(rep["saturation"]["sor"]) - 0.20) < 1.0e-12
    assert abs(float(rep["saturation"]["movable"]) - 0.60) < 1.0e-12
    assert rep["dynamic"]["capillary_model"] == "brooks_corey"
    assert rep["dynamic"]["pc_entry_pa"] is not None
    assert rep["dynamic"]["capillary_over_viscous"] is not None
    assert rep["dynamic"]["gravity_on"] is False
    assert rep["dynamic"]["gravity_over_viscous"] is None
    assert abs(float(rep["dynamic"]["compressibility_ct_1_pa"])) < 1.0e-30
    assert rep["displacement"]["comparison"] == "F(m_post) vs F(m_true)"
    assert rep["displacement"]["not"] == "CMG"
    assert rep["displacement"]["sw_field_nrmse"] is None
    assert "field_geometric_ratios" in rep["skipped"]
    assert "thermal" in rep["skipped"]
    assert "polymer" in rep["skipped"]
    assert "shale" in rep["skipped"]
    assert "gravity_over_viscous" in rep["skipped"]


def test_displacement_nrmse_filled_without_invert() -> None:
    import numpy as np

    twin = load_case(CASE)
    n = twin.grid.n_cells
    sw_t = np.full(n, 0.25)
    sw_p = np.full(n, 0.25)
    p_t = np.full(n, 1.0e5)
    p_p = np.full(n, 1.0e5)
    filled = attach_displacement(
        waterflood_groups(twin),
        sw_post=sw_p,
        sw_true=sw_t,
        p_post=p_p,
        p_true=p_t,
    )
    assert filled["displacement"]["sw_field_nrmse"] == 0.0
    assert filled["displacement"]["p_field_nrmse"] == 0.0
    assert filled["displacement"]["not"] == "CMG"
