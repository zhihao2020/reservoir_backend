from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.history_matching.synthetic_twin import (
    add_observation_noise,
    apply_baseline_parameter_update,
    compute_rmse,
    forward_simulate_observations,
    generate_truth_fields,
    run_synthetic_twin_history_matching,
)


def test_truth_field_generation() -> None:
    truth = generate_truth_fields((2, 3, 4))
    assert truth["permeability"].shape == (2, 3, 4)
    assert np.all(truth["porosity"] > 0.0)


def test_observation_generation() -> None:
    obs = forward_simulate_observations(generate_truth_fields())
    assert {"pressure", "liquid_rate", "water_cut", "saturation_proxy"} <= set(obs)
    assert obs["pressure"].shape == obs["time"].shape


def test_noise_injection() -> None:
    obs = forward_simulate_observations(generate_truth_fields())
    noisy = add_observation_noise(obs, noise_std=0.05, seed=3)
    assert np.array_equal(noisy["time"], obs["time"])
    assert not np.array_equal(noisy["pressure"], obs["pressure"])


def test_negative_noise_rejected() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        add_observation_noise(forward_simulate_observations(generate_truth_fields()), noise_std=-1.0)


def test_baseline_parameter_update() -> None:
    truth = generate_truth_fields()
    initial = {"permeability": truth["permeability"] * 0.8, "porosity": truth["porosity"] + 0.02}
    updated = apply_baseline_parameter_update(initial, truth, update_fraction=0.5)
    assert compute_rmse(truth, updated, ("permeability", "porosity")) < compute_rmse(truth, initial, ("permeability", "porosity"))


def test_invalid_update_fraction_rejected() -> None:
    truth = generate_truth_fields()
    with pytest.raises(ValueError, match="update_fraction"):
        apply_baseline_parameter_update(truth, truth, update_fraction=1.5)


def test_rmse_before_after(tmp_path: Path) -> None:
    summary = run_synthetic_twin_history_matching(output_dir=tmp_path)
    assert summary["rmse_after"] < summary["rmse_before"]


def test_prediction_error_before_after(tmp_path: Path) -> None:
    summary = run_synthetic_twin_history_matching(output_dir=tmp_path)
    assert summary["prediction_rmse_after"] < summary["prediction_rmse_before"]


def test_uncertainty_summary(tmp_path: Path) -> None:
    summary = run_synthetic_twin_history_matching(output_dir=tmp_path)
    uncertainty = summary["uncertainty_summary"]
    assert uncertainty["permeability_abs_error_max"] >= uncertainty["permeability_abs_error_min"]


def test_summary_json_serializable(tmp_path: Path) -> None:
    json.dumps(run_synthetic_twin_history_matching(output_dir=tmp_path))


def test_summary_reports_generated(tmp_path: Path) -> None:
    run_synthetic_twin_history_matching(output_dir=tmp_path)
    assert (tmp_path / "synthetic_twin_history_matching_summary.json").exists()
    assert (tmp_path / "synthetic_twin_history_matching_summary.md").exists()


def test_no_real_field_history_matching_claim(tmp_path: Path) -> None:
    summary = run_synthetic_twin_history_matching(output_dir=tmp_path)
    assert "No real field history matching claim." in summary["non_claims"]


def test_no_complete_enkf_claim(tmp_path: Path) -> None:
    summary = run_synthetic_twin_history_matching(output_dir=tmp_path)
    assert "No complete EnKF implementation." in summary["non_claims"]
    assert "No complete ES-MDA implementation." in summary["non_claims"]


def test_report_markdown_contains_limitations(tmp_path: Path) -> None:
    run_synthetic_twin_history_matching(output_dir=tmp_path)
    text = (tmp_path / "synthetic_twin_history_matching_summary.md").read_text(encoding="utf-8")
    assert "Known Limitations" in text
    assert "Non-Claims" in text
