"""Example case library: shale_oil parses and runs end-to-end."""

from pathlib import Path

from src.core.lab_case import load_lab_case
from src.programs.pipeline import run_pipeline, write_output

ROOT = Path(__file__).resolve().parents[1]
SHALE = ROOT / "example" / "case.yaml"
EXAMPLE = ROOT / "example"


def test_example_bundle_files():
    for name in (
        "case.yaml",
        "probes.csv",
        "wells.csv",
        "observations.csv",
        "series.csv",
        "send_steps.py",
        "receive_fields.py",
        "reservoir.py",
        "requirements.txt",
        "README.md",
    ):
        assert (EXAMPLE / name).is_file(), name


def test_shale_oil_loads_and_runs(tmp_path):
    case = load_lab_case(SHALE)
    assert case.nx == 15 and case.ny == 15 and case.nz == 15
    assert len(case.probe_ids) == 27
    assert len(case.well_ids) == 5
    assert case.times.size == 120
    fields = run_pipeline(case)
    assert fields.p.shape == (case.times.size, case.nx * case.ny * case.nz)
    summary = write_output(tmp_path, case, fields)
    assert (tmp_path / "fields.npz").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "results.json").exists()
    assert summary["n_cells"] == case.nx * case.ny * case.nz
