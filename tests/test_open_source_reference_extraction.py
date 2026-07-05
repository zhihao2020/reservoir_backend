from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
FIXTURE_JSON = REFERENCES / "fixtures" / "open_source_adapted_cases.json"
FIXTURE_NPZ = REFERENCES / "fixtures" / "open_source_adapted_arrays.npz"


def _load_summary() -> dict:
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def _case(name: str) -> dict:
    return next(case for case in _load_summary()["cases"] if case["case_name"] == name)


def test_reference_upstream_files_exist():
    paths = [
        REFERENCES / "upstream" / "opm-tests" / "water-1ph" / "WATER2F.DATA",
        REFERENCES / "upstream" / "opm-tests" / "spe1" / "SPE1CASE1.DATA",
        REFERENCES / "upstream" / "mrst" / "modules" / "book" / "examples" / "1phase" / "src" / "simpleIncompTPFA.m",
        REFERENCES / "upstream" / "mrst" / "modules" / "book" / "examples" / "in2ph" / "buckleyLeverett1D.m",
    ]
    assert all(path.exists() for path in paths)


def test_reference_fixture_json_exists():
    assert FIXTURE_JSON.exists()


def test_reference_fixture_npz_exists():
    assert FIXTURE_NPZ.exists()


def test_extract_reference_cases_script_runs():
    spec = importlib.util.spec_from_file_location("extract_reference_cases", REFERENCES / "extract_reference_cases.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    summary = module.extract_reference_cases()
    assert summary["success"] is True


def test_opm_water_1ph_extracted_properties():
    case = _case("opm_water_1ph_single_cell")
    assert case["grid"] == [1, 1, 1]
    assert case["porosity"] == 0.1
    assert case["permeability_md"]["kx"] == 1000.0
    assert case["permeability_md"]["kz"] == 100.0


def test_opm_spe1_layered_subset_extracted():
    case = _case("opm_spe1_case1_layered_subset")
    assert case["grid"] == [10, 10, 3]
    assert case["porosity_min"] == 0.3
    assert case["porosity_max"] == 0.3
    assert case["permeability_contrast"] == 10.0


def test_mrst_tpfa_reference_extracted():
    case = _case("mrst_simple_incomp_tpfa_reference")
    assert case["mentions_tpfa"] is True
    assert case["mentions_boundary_conditions"] is True
    assert case["mentions_sources"] is True


def test_mrst_buckley_leverett_reference_extracted():
    case = _case("mrst_buckley_leverett_1d_reference")
    assert case["grid"] == [100, 1]
    assert case["permeability_md"] == 100.0
    assert case["porosity"] == 0.2
    assert case["mentions_explicit_transport"] is True


def test_npz_contains_spe1_arrays():
    data = np.load(FIXTURE_NPZ)
    assert data["spe1_permx_md"].shape == (3, 10, 10)
    assert data["spe1_dz_ft"].shape == (3, 10, 10)
    assert np.min(data["spe1_permx_md"]) == 50.0
    assert np.max(data["spe1_permx_md"]) == 500.0


def test_reference_policy_disclaims_runtime_dependency_and_equivalence():
    policy = _load_summary()["policy"]
    assert policy["runtime_dependency"] is False
    assert policy["full_spe10_reproduction"] is False
    assert policy["opm_flow_equivalence"] is False
    assert policy["mrst_runtime_integration"] is False
    assert policy["commercial_simulator_equivalence"] is False


def test_references_readme_records_policy():
    text = (REFERENCES / "README.md").read_text(encoding="utf-8")
    assert "not imported as runtime dependencies" in text
    assert "full SPE10 reproduction" in text
    assert "OPM Flow equivalence" in text
