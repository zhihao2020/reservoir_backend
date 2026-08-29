"""CMG shale S1–S5 case assembly (skip without IMEX .out)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reservoir_backend.io.shale_case import invert_shale_case, twin_from_shale_truth

ROOT = Path(__file__).resolve().parents[2]
SHALE = ROOT / "validation" / "shale_oil"
CASE_DIRS = {
    "S1": "cmg_s1_hw5frac",
    "S2": "cmg_s2_hw9frac",
    "S3": "cmg_s3_twohw",
    "S4": "cmg_s4_parent_child",
    "S5": "cmg_s5_shutin",
}


@pytest.mark.parametrize("case", list(CASE_DIRS))
def test_shale_cmg_twin_builds(case: str) -> None:
    c = case.lower()
    truth_path = SHALE / CASE_DIRS[case.upper()] / f"truth_{c}.json"
    out_path = SHALE / CASE_DIRS[case.upper()] / f"mxshale_{c}.out"
    if not out_path.is_file():
        pytest.skip(f"missing IMEX .out for {case}")
    twin = twin_from_shale_truth(truth_path, out_path=out_path, n_times=3, max_iter=4)
    assert twin.parameterization.n_params == 4
    assert twin.physics.fully_implicit is False
    assert len(twin.ports) >= 3
    assert len(twin.experiment.observations) >= 3
    assert all(o.sigma.size == o.values.size for o in twin.experiment.observations)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["S1"])
def test_shale_cmg_inversion_smoke(case: str) -> None:
    """One-case LM smoke; full S1–S5 gates via validation script."""
    c = case.lower()
    truth_path = SHALE / CASE_DIRS[case.upper()] / f"truth_{c}.json"
    out_path = SHALE / CASE_DIRS[case.upper()] / f"mxshale_{c}.out"
    if not out_path.is_file():
        pytest.skip(f"missing IMEX .out for {case}")
    rec = invert_shale_case(
        truth_path,
        out_path=out_path,
        n_times=3,
        max_iter=3,
        time_limit_s=600.0,
    )
    assert rec.get("ok"), rec.get("error", "invert failed")
    assert rec.get("frac_theta") is True
    assert float(rec.get("assimilate_nrmse", 99.0)) < 20.0
