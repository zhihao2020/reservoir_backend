"""Example case library: every case parses; the fast ones run end-to-end."""

from pathlib import Path

import numpy as np
import pytest

from src.core.lab_case import load_lab_case
from src.programs.pipeline import run_pipeline, write_output

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# (case.yaml relative path, run_pipeline?)
CASES = [
    ("small/case.yaml", True),
    ("twod/case.yaml", True),
    ("model_compare/total_mobility.yaml", True),
    ("model_compare/three_phase.yaml", True),
    ("offline/case.yaml", False),  # 20^3, load only to keep the suite fast
    ("online/case.yaml", False),  # 60^3, load only
]


@pytest.mark.parametrize("rel,run", CASES)
def test_case_loads(rel, run):
    case = load_lab_case(EXAMPLES / rel)
    assert case.nx > 0 and case.ny > 0 and case.nz > 0
    assert case.probe_ids
    assert case.well_ids
    assert case.times.size > 0


@pytest.mark.parametrize("rel", ["small/case.yaml", "twod/case.yaml"])
def test_fast_cases_run_and_write(rel, tmp_path):
    case = load_lab_case(EXAMPLES / rel)
    fields = run_pipeline(case)
    assert fields.p.shape == (case.times.size, case.nx * case.ny * case.nz)
    summary = write_output(tmp_path, case, fields)
    assert (tmp_path / "fields.npz").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "results.json").exists()
    assert summary["n_cells"] == case.nx * case.ny * case.nz


def test_model_compare_both_models_run():
    tm = load_lab_case(EXAMPLES / "model_compare" / "total_mobility.yaml")
    tp = load_lab_case(EXAMPLES / "model_compare" / "three_phase.yaml")
    assert tm.rock_model == "total_mobility"
    assert tp.rock_model == "black_oil_3phase"
    for case in (tm, tp):
        fields = run_pipeline(case)
        assert fields.p.shape[0] == case.times.size
