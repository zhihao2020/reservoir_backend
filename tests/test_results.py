"""Per-probe / per-well results derivation."""

from pathlib import Path

import numpy as np

from src.core.lab_case import load_lab_case
from src.programs.pipeline import field_scale_view, run_pipeline
from src.programs.results import summarize_results

ROOT = Path(__file__).resolve().parents[1]
SHALE = ROOT / "example" / "case.yaml"


def _case(n_times: int = 3):
    case = load_lab_case(SHALE)
    case.times = case.times[:n_times]
    case.pressure = case.pressure[:n_times]
    case.sw = case.sw[:n_times]
    case.so = case.so[:n_times]
    case.sg = case.sg[:n_times]
    case.well_pw = case.well_pw[:n_times]
    case.well_q = case.well_q[:n_times]
    case.well_qw = case.well_qw[:n_times]
    case.well_qo = case.well_qo[:n_times]
    case.well_qg = case.well_qg[:n_times]
    return case


def test_summarize_results_structure():
    case = _case()
    fields = run_pipeline(case)
    out = summarize_results(case, fields.mesh, fields)

    assert out["n_times"] == case.times.size
    assert out["times_s"] == [float(t) for t in case.times]
    # one entry per probe, in id order
    assert [p["id"] for p in out["probes"]] == case.probe_ids
    for p in out["probes"]:
        assert set(p) == {"id", "cell", "observed", "reconstructed", "rmse"}
        assert set(p["rmse"]) == {"pressure_pa", "sw", "so", "sg"}
        assert np.isfinite(p["rmse"]["pressure_pa"])

    # one entry per well
    assert [w["id"] for w in out["wells"]] == case.well_ids
    for w in out["wells"]:
        assert set(w) == {"id", "kind", "n_cells", "bhp_reconstructed_pa", "sw", "so", "sg", "control"}
        assert len(w["bhp_reconstructed_pa"]) == out["n_times"]
        assert set(w["control"]) == {"pw_pa", "q_m3s", "qw_m3s", "qo_m3s", "qg_m3s"}

    assert "diagnostics" in out


def test_summarize_results_field_times_override():
    case = _case()
    fields = run_pipeline(case)
    _, field_times, _ = field_scale_view(case, fields)
    lab = summarize_results(case, fields.mesh, fields)
    field = summarize_results(case, fields.mesh, fields, times=field_times)

    # field stream's time axis must match its (scaled) MANIFEST.times …
    assert field["times_s"] == [float(t) for t in field_times]
    # … while the index-aligned observed/reconstructed values are unchanged.
    assert (
        field["probes"][0]["reconstructed"]["pressure_pa"]
        == lab["probes"][0]["reconstructed"]["pressure_pa"]
    )
    assert field["n_times"] == lab["n_times"]
