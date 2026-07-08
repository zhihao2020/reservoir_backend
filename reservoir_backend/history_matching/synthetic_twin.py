"""Synthetic twin history-matching prototype.

This module is deliberately synthetic-only. It demonstrates an error-reduction
baseline using known truth fields and generated observations; it is not a real
field history-matching product, EnKF/ES-MDA implementation, or automatic
calibration workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.project.project_registry import json_safe


@dataclass(frozen=True)
class SyntheticTwinHistoryResult:
    """JSON-ready result for the synthetic history-matching prototype."""

    success: bool
    shape: list[int]
    rmse_before: float
    rmse_after: float
    prediction_rmse_before: float
    prediction_rmse_after: float
    uncertainty_summary: dict[str, float]
    warnings: list[str]
    limitations: list[str]
    non_claims: list[str]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def generate_truth_fields(shape: tuple[int, ...] = (2, 4, 6), seed: int = 7) -> dict[str, NDArray[np.float64]]:
    """Generate deterministic synthetic truth permeability and porosity fields."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=float).reshape(shape)
    permeability = 100.0 + 40.0 * base + rng.normal(0.0, 2.0, size=shape)
    porosity = 0.18 + 0.06 * base + rng.normal(0.0, 0.002, size=shape)
    return {
        "permeability": np.clip(permeability, 1.0, None),
        "porosity": np.clip(porosity, 0.01, 0.8),
    }


def forward_simulate_observations(fields: Mapping[str, NDArray[np.float64]], times: NDArray[np.float64] | None = None) -> dict[str, NDArray[np.float64]]:
    """Generate synthetic pressure, production, and water-cut responses."""
    if times is None:
        times = np.linspace(0.0, 5.0, 6)
    k_mean = float(np.mean(fields["permeability"]))
    phi_mean = float(np.mean(fields["porosity"]))
    pressure = 10.0 - 0.08 * times + 0.001 * (k_mean - 100.0)
    liquid_rate = 40.0 + 0.05 * k_mean - 2.0 * phi_mean + 0.1 * times
    water_cut = np.clip(0.02 + 0.015 * times + 0.3 * (phi_mean - 0.2), 0.0, 1.0)
    saturation_proxy = np.clip(phi_mean + 0.01 * times, 0.0, 1.0)
    return {
        "time": np.asarray(times, dtype=float),
        "pressure": np.asarray(pressure, dtype=float),
        "liquid_rate": np.asarray(liquid_rate, dtype=float),
        "water_cut": np.asarray(water_cut, dtype=float),
        "saturation_proxy": np.asarray(saturation_proxy, dtype=float),
    }


def add_observation_noise(observations: Mapping[str, NDArray[np.float64]], noise_std: float = 0.01, seed: int = 11) -> dict[str, NDArray[np.float64]]:
    """Add deterministic Gaussian noise to synthetic observation arrays."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    rng = np.random.default_rng(seed)
    noisy: dict[str, NDArray[np.float64]] = {}
    for key, values in observations.items():
        array = np.asarray(values, dtype=float)
        if key == "time" or noise_std == 0.0:
            noisy[key] = array.copy()
        else:
            scale = max(float(np.std(array)), 1.0)
            noisy[key] = array + rng.normal(0.0, noise_std * scale, size=array.shape)
    return noisy


def apply_baseline_parameter_update(
    initial_fields: Mapping[str, NDArray[np.float64]],
    truth_fields: Mapping[str, NDArray[np.float64]],
    update_fraction: float = 0.6,
) -> dict[str, NDArray[np.float64]]:
    """Move synthetic initial fields toward known synthetic truth fields."""
    if not 0.0 <= update_fraction <= 1.0:
        raise ValueError("update_fraction must be within [0, 1]")
    updated: dict[str, NDArray[np.float64]] = {}
    for key in ("permeability", "porosity"):
        initial = np.asarray(initial_fields[key], dtype=float)
        truth = np.asarray(truth_fields[key], dtype=float)
        if initial.shape != truth.shape:
            raise ValueError(f"{key} shape mismatch")
        updated[key] = initial + update_fraction * (truth - initial)
    updated["permeability"] = np.clip(updated["permeability"], 1.0, None)
    updated["porosity"] = np.clip(updated["porosity"], 0.0, 1.0)
    return updated


def compute_rmse(reference: Mapping[str, NDArray[np.float64]], candidate: Mapping[str, NDArray[np.float64]], keys: tuple[str, ...]) -> float:
    """Compute combined RMSE across named arrays."""
    errors: list[NDArray[np.float64]] = []
    for key in keys:
        ref = np.asarray(reference[key], dtype=float)
        cand = np.asarray(candidate[key], dtype=float)
        if ref.shape != cand.shape:
            raise ValueError(f"{key} shape mismatch")
        errors.append((cand - ref).ravel())
    combined = np.concatenate(errors)
    return float(np.sqrt(np.mean(combined**2)))


def run_synthetic_twin_history_matching(
    *,
    output_dir: str | Path = "accuracy_reports",
    shape: tuple[int, ...] = (2, 4, 6),
    seed: int = 7,
) -> dict[str, Any]:
    """Run the synthetic prototype and write JSON/Markdown reports."""
    truth = generate_truth_fields(shape=shape, seed=seed)
    initial = {
        "permeability": truth["permeability"] * 0.78,
        "porosity": np.clip(truth["porosity"] + 0.025, 0.0, 1.0),
    }
    truth_obs = forward_simulate_observations(truth)
    noisy_obs = add_observation_noise(truth_obs, noise_std=0.02, seed=seed + 1)
    initial_obs = forward_simulate_observations(initial, noisy_obs["time"])
    updated = apply_baseline_parameter_update(initial, truth, update_fraction=0.65)
    updated_obs = forward_simulate_observations(updated, noisy_obs["time"])

    rmse_before = compute_rmse(truth, initial, ("permeability", "porosity"))
    rmse_after = compute_rmse(truth, updated, ("permeability", "porosity"))
    prediction_before = compute_rmse(noisy_obs, initial_obs, ("pressure", "liquid_rate", "water_cut", "saturation_proxy"))
    prediction_after = compute_rmse(noisy_obs, updated_obs, ("pressure", "liquid_rate", "water_cut", "saturation_proxy"))
    uncertainty = np.abs(updated["permeability"] - truth["permeability"])
    result = SyntheticTwinHistoryResult(
        success=bool(rmse_after < rmse_before and prediction_after < prediction_before),
        shape=list(shape),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        prediction_rmse_before=prediction_before,
        prediction_rmse_after=prediction_after,
        uncertainty_summary={
            "permeability_abs_error_min": float(np.min(uncertainty)),
            "permeability_abs_error_mean": float(np.mean(uncertainty)),
            "permeability_abs_error_max": float(np.max(uncertainty)),
        },
        warnings=["synthetic-only prototype; uses known truth fields"],
        limitations=_limitations(),
        non_claims=_non_claims(),
    )
    summary = {
        "summary_name": "synthetic_twin_history_matching_summary",
        "source_task": "IND-004",
        **result.to_dict(),
        "observation_generation": {
            "noise_injected": True,
            "noise_std": 0.02,
            "observation_keys": ["pressure", "liquid_rate", "water_cut", "saturation_proxy"],
        },
        "baseline_update": {
            "method": "known-truth fractional update for synthetic prototype",
            "update_fraction": 0.65,
        },
    }
    _write_reports(summary, output_dir)
    return summary


def _write_reports(summary: Mapping[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "synthetic_twin_history_matching_summary.json"
    md_path = root / "synthetic_twin_history_matching_summary.md"
    json_path.write_text(json.dumps(json_safe(dict(summary)), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Synthetic Twin History Matching Prototype Summary",
        "",
        "## Implemented Scope",
        "",
        f"- success: {summary['success']}",
        f"- shape: {summary['shape']}",
        f"- RMSE before: {summary['rmse_before']}",
        f"- RMSE after: {summary['rmse_after']}",
        f"- prediction RMSE before: {summary['prediction_rmse_before']}",
        f"- prediction RMSE after: {summary['prediction_rmse_after']}",
        "",
        "## Test Results",
        "",
        "- See `tests/test_synthetic_twin_history_matching.py` and `pytest -q`.",
        "",
        "## Known Limitations",
        "",
    ]
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Non-Claims", ""])
    for item in summary["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Steps", "", "- Keep real-field history matching out of the current mainline."])
    return "\n".join(lines) + "\n"


def _limitations() -> list[str]:
    return [
        "Synthetic truth fields only.",
        "Uses generated observations and deterministic noise.",
        "Baseline update is not a field calibration product.",
    ]


def _non_claims() -> list[str]:
    return [
        "No real field history matching claim.",
        "No complete EnKF implementation.",
        "No complete ES-MDA implementation.",
        "No automatic calibration product.",
        "No closed-loop digital twin product.",
    ]
