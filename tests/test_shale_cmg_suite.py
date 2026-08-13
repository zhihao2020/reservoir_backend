"""Deck + optional IMEX-output smoke for the five shale analog rulers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "shale_oil" / "validation" / "cmg_shale_suite"
VAL = SUITE.parent
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))
from build_shale_suite import CASE_DIR, CASES, NX, NY, NZ, frac_i_list, wells_for  # noqa: E402

_CMG_IO = ROOT / "black_oil" / "validation"
if str(_CMG_IO) not in sys.path:
    sys.path.insert(0, str(_CMG_IO))
from cmg_io.grid_parse import parse_grid_series  # noqa: E402


def _dat(case: str) -> Path:
    return VAL / CASE_DIR[case] / f"mxshale_{case.lower()}.dat"


def _out(case: str) -> Path:
    return VAL / CASE_DIR[case] / f"mxshale_{case.lower()}.out"


def _truth(case: str) -> dict:
    path = VAL / CASE_DIR[case] / f"truth_{case.lower()}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_truth_and_deck_match_scenario(case: str) -> None:
    truth = _truth(case)
    dat = _dat(case).read_text(encoding="latin-1")
    wells = wells_for(case)
    assert truth["scenario"] == case
    assert truth["grid"]["nx"] == NX
    assert "Not GEM" in truth["analog_note"]
    assert "*INJECTOR" not in dat
    assert "*PRODUCER" in dat
    assert "*OUTPRN *GRID *SO *SG *SW *PRES" in dat
    assert "*WPRN   *GRID 5" in dat
    assert "*MIN       *BHP" in dat
    assert "HW1" in dat
    assert f"*GRID *CART {NX} {NY} {NZ}" in dat
    assert truth["frac_perm_md"] == 8000.0
    assert truth["n_frac_blocks"] > 0
    assert len(truth["wells"]) == len(wells)
    assert truth["frac_i_planes"] == frac_i_list(case if case != "S3" else "S1")


def test_s3_s4_have_two_wells_s4_child_starts_shut() -> None:
    assert [w["name"] for w in wells_for("S3")] == ["HW1", "HW2"]
    assert wells_for("S4")[1]["open_from_day"] == 365.0
    s4 = _dat("S4").read_text(encoding="latin-1")
    assert "HW2" in s4
    assert "*SHUTIN  2" in s4
    assert "*OPEN    2" in s4


def test_s5_has_midlife_shutin_and_reopen() -> None:
    s5 = _dat("S5").read_text(encoding="latin-1")
    assert "*SHUTIN  1" in s5
    assert "*OPEN    1" in s5
    assert s5.find("*SHUTIN  1") < s5.find("*OPEN    1")


def test_s2_has_more_frac_stages_than_s1() -> None:
    assert len(frac_i_list("S2")) == 9
    assert len(frac_i_list("S1")) == 5
    assert _truth("S2")["n_frac_blocks"] > _truth("S1")["n_frac_blocks"]


@pytest.mark.parametrize("case", CASES)
def test_parse_pres_sw_if_out_present(case: str) -> None:
    out_path = _out(case)
    if not out_path.is_file():
        pytest.skip(f"{out_path.name} not present")
    text = out_path.read_text(encoding="latin-1", errors="replace")
    assert "Normal Termination" in text
    truth = _truth(case)
    nx, ny, nz = truth["grid"]["nx"], truth["grid"]["ny"], truth["grid"]["nz"]
    sw = parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)
    pr = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    assert len(sw) >= 2
    assert len(pr) >= 2
    _, p_last = pr[-1]
    _, sw_last = sw[-1]
    assert p_last.shape == (nz, ny, nx)
    assert float(np.mean(np.isfinite(p_last))) > 0.99
    assert float(np.mean(np.isfinite(sw_last))) > 0.99
    assert float(np.nanstd(p_last)) > 1.0
    assert 0.0 <= float(np.nanmin(sw_last)) <= float(np.nanmax(sw_last)) <= 1.0
    frac = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in truth["high_k_blocks_ijk"]:
        frac[k - 1, j - 1, i - 1] = True
    srv = np.zeros((nz, ny, nx), dtype=bool)
    for i, j, k in truth.get("srv_blocks_ijk", []):
        srv[k - 1, j - 1, i - 1] = True
    matrix = ~(frac | srv)
    dp = float(np.nanmean(p_last[matrix]) - np.nanmean(p_last[frac]))
    assert dp > 10.0
