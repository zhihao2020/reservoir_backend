"""Synthetic benchmarks for saturation inversion hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reservoir_backend.inversion.acoustic import invert_saturation_acoustic
from reservoir_backend.inversion.electromagnetic import invert_saturation_em
from reservoir_backend.inversion.resistivity_archie import invert_saturation_archie
from reservoir_backend.inversion.saturation_fusion import fuse_saturation_estimates


def run_saturation_inversion_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run synthetic saturation inversion checks and write JSON/Markdown reports."""
    archie_error = _archie_formula_check()
    noise_report = _noise_sensitivity_check()
    fusion_report = _fusion_check()
    clipping_report = _clipping_check()

    checks = [
        archie_error < 1.0e-12,
        noise_report["monotonic_error_growth"],
        noise_report["bounded"],
        fusion_report["bounded"],
        fusion_report["closer_to_low_uncertainty_signal"],
        clipping_report["clipping_checked"],
    ]
    summary = {
        "benchmark_name": "saturation_inversion_benchmark",
        "success": bool(all(checks)),
        "num_cases": len(checks),
        "num_passed": int(sum(checks)),
        "num_failed": int(len(checks) - sum(checks)),
        "archie_formula_error": float(archie_error),
        "noise_sensitivity": noise_report,
        "fusion_error": float(fusion_report["fused_error"]),
        "clipping_checked": bool(clipping_report["clipping_checked"]),
        "warnings": [],
        "has_nan": bool(noise_report["has_nan"] or fusion_report["has_nan"]),
        "has_inf": bool(noise_report["has_inf"] or fusion_report["has_inf"]),
        "recommendations": [
            "Keep Archie analytical formula as a regression gate.",
            "Use uncertainty-weighted fusion when source uncertainty is known.",
            "Do not claim Bayesian inversion or commercial petrophysical interpretation.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _archie_formula_check() -> float:
    sw_true = np.linspace(0.2, 0.8, 9)
    phi = np.full_like(sw_true, 0.25)
    rw = 0.2
    a = 1.0
    m = 2.0
    n = 2.0
    rt = a * rw / ((phi**m) * (sw_true**n))
    sw_pred = invert_saturation_archie(rt, rw, phi, a=a, m=m, n=n)
    return float(np.max(np.abs(np.asarray(sw_pred) - sw_true)))


def _noise_sensitivity_check() -> dict:
    rng = np.random.default_rng(2026)
    sw_true = np.linspace(0.25, 0.75, 30)
    phi = np.full_like(sw_true, 0.24)
    rw = 0.15
    rt = rw / ((phi**2.0) * (sw_true**2.0))
    errors = []
    bounded = True
    for noise in (0.01, 0.05, 0.10):
        noisy_rt = rt * (1.0 + rng.normal(0.0, noise, size=rt.shape))
        sw_pred = np.asarray(invert_saturation_archie(noisy_rt, rw, phi), dtype=float)
        errors.append(float(np.mean(np.abs(sw_pred - sw_true))))
        bounded = bounded and bool(np.all((sw_pred >= 0.0) & (sw_pred <= 1.0)))
    return {
        "levels": [0.01, 0.05, 0.10],
        "mean_abs_errors": errors,
        "monotonic_error_growth": bool(errors[0] <= errors[1] <= errors[2]),
        "bounded": bounded,
        "has_nan": bool(np.isnan(errors).any()),
        "has_inf": bool(np.isinf(errors).any()),
    }


def _fusion_check() -> dict:
    truth = np.array([0.35, 0.45, 0.55])
    estimates = {
        "archie": truth + np.array([0.005, -0.004, 0.003]),
        "em": truth + np.array([0.03, -0.02, 0.025]),
        "acoustic": truth + np.array([0.08, -0.06, 0.07]),
    }
    uncertainties = {"archie": 0.01, "em": 0.05, "acoustic": 0.10}
    fused, report = fuse_saturation_estimates(estimates, uncertainties=uncertainties, return_report=True)
    fused_arr = np.asarray(fused, dtype=float)
    archie_error = float(np.mean(np.abs(estimates["archie"] - truth)))
    fused_error = float(np.mean(np.abs(fused_arr - truth)))
    return {
        "fused_error": fused_error,
        "archie_error": archie_error,
        "closer_to_low_uncertainty_signal": bool(
            np.mean(np.abs(fused_arr - estimates["archie"])) < np.mean(np.abs(fused_arr - estimates["acoustic"]))
        ),
        "bounded": bool(np.all((fused_arr >= 0.0) & (fused_arr <= 1.0))),
        "weights_normalized": bool(abs(sum(report["normalized_weights"].values()) - 1.0) < 1.0e-12),
        "has_nan": bool(np.isnan(fused_arr).any()),
        "has_inf": bool(np.isinf(fused_arr).any()),
    }


def _clipping_check() -> dict:
    _, em_report = invert_saturation_em(np.array([0.0, 1.0]), [-0.2, 1.5], return_report=True)
    _, acoustic_report = invert_saturation_acoustic(np.array([0.0, 1.0]), [1.2, -1.5], return_report=True)
    return {
        "clipping_checked": bool(
            em_report["num_clipped_low"] > 0
            and em_report["num_clipped_high"] > 0
            and acoustic_report["num_clipped_low"] > 0
            and acoustic_report["num_clipped_high"] > 0
        )
    }


def _write_reports(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "saturation_inversion_benchmark_summary.json"
    md_path = output_dir / "saturation_inversion_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Saturation Inversion Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- archie_formula_error: {summary['archie_formula_error']:.6e}",
        f"- fusion_error: {summary['fusion_error']:.6e}",
        f"- clipping_checked: {summary['clipping_checked']}",
        f"- has_nan: {summary['has_nan']}",
        f"- has_inf: {summary['has_inf']}",
        "",
        "## Noise Sensitivity",
        "",
        json.dumps(summary["noise_sensitivity"], indent=2),
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_saturation_inversion_benchmark(), indent=2))
