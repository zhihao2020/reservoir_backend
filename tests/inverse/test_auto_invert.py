import numpy as np

from reservoir_backend.inverse.lm import (
    HOLDOUT_WHITENED_ESCALATE,
    IDENT_RATIO_ESCALATE,
    should_run_ensemble,
)
from reservoir_backend.io.case import inverse_spec_from_cfg, load_case


def test_should_run_ensemble_lm_when_pinned() -> None:
    go, reason = should_run_ensemble([0.2], assimilate_rmse=0.8, holdout_rmse=0.9, uq=False)
    assert go is False
    assert reason == "lm_sufficient"


def test_should_run_ensemble_uq_uses_lm_interval_when_pinned() -> None:
    go, reason = should_run_ensemble([0.2], assimilate_rmse=0.8, holdout_rmse=0.9, uq=True)
    assert go is False
    assert reason == "interval_from_lm"


def test_should_run_ensemble_when_weak_identifiability() -> None:
    go, reason = should_run_ensemble(
        [IDENT_RATIO_ESCALATE + 0.05], assimilate_rmse=0.5, holdout_rmse=0.6, uq=False
    )
    assert go is True
    assert reason == "weak_identifiability"


def test_should_run_ensemble_when_holdout_blows_up() -> None:
    go, reason = should_run_ensemble(
        [0.1], assimilate_rmse=0.4, holdout_rmse=HOLDOUT_WHITENED_ESCALATE + 0.5, uq=False
    )
    assert go is True
    assert reason == "holdout"


def test_cf_yaml_defaults_to_auto() -> None:
    spec = inverse_spec_from_cfg({"parameterization": "log_conductivity"})
    assert spec.algorithm == "auto"
    spec_j = inverse_spec_from_cfg({"parameterization": "log_cf_tmf"})
    assert spec_j.algorithm == "auto"
    spec_lm = inverse_spec_from_cfg({"parameterization": "region"})
    assert spec_lm.algorithm == "lm"


def test_explicit_esmda_still_selected() -> None:
    spec = inverse_spec_from_cfg({"parameterization": "log_cf_tmf", "algorithm": "esmda"})
    assert spec.algorithm == "esmda"


def test_lab_v1_and_physical_3d_use_auto() -> None:
    assert load_case("examples/lab_v1/case_dev.yaml").inverse.algorithm == "auto"
    assert load_case("examples/lab_v1/case.yaml").inverse.algorithm == "auto"
    assert load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml").inverse.algorithm == "auto"
    assert load_case("examples/lab/lab_cf.yaml").inverse.algorithm == "auto"
