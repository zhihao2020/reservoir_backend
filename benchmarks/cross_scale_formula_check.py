"""Formula checks for cross-scale similarity, scale-effect, and validation."""

from __future__ import annotations

import numpy as np

from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.scale_effect import compute_scale_ratios
from reservoir_backend.cross_scale.similarity import (
    compute_capillary_number,
    compute_criterion_similarity_score,
    compute_peclet_number,
    compute_reynolds_number,
)
from reservoir_backend.cross_scale.validation import compute_mae, compute_r2, compute_rmse


def run_benchmark() -> dict:
    lab = _descriptor()
    field = ScaleDescriptor.from_dict({**lab.to_dict(), "length_scale_m": 20.0})
    checks = [
        compute_reynolds_number(lab).value - 20.0,
        compute_capillary_number(lab).value - (1.0e-3 * 1.0e-5 / 0.03),
        compute_peclet_number(lab).value - 2.0e4,
        compute_criterion_similarity_score(10.0, 10.0).value - 1.0,
        compute_scale_ratios(lab, field)["scale_ratio_length"] - 10.0,
        compute_rmse([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) - np.sqrt(4.0 / 3.0),
        compute_mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) - (2.0 / 3.0),
        compute_r2([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) - 0.5,
    ]
    errors = np.abs(np.asarray(checks, dtype=float))
    return {
        "benchmark_name": "cross_scale_formula_check",
        "success": bool(np.max(errors) < 1.0e-12),
        "formula_checks_passed": int(np.sum(errors < 1.0e-12)),
        "num_formula_checks": int(errors.size),
        "max_formula_error": float(np.max(errors)),
        "has_nan": bool(np.isnan(errors).any()),
        "has_inf": bool(np.isinf(errors).any()),
        "warnings": [],
    }


def _descriptor() -> ScaleDescriptor:
    return ScaleDescriptor.from_dict(
        {
            "length_scale_m": 2.0,
            "time_scale_s": 100.0,
            "pressure_scale_pa": 1.0e6,
            "permeability_scale_m2": 1.0e-12,
            "porosity": 0.25,
            "viscosity_pa_s": 1.0e-3,
            "density_kg_m3": 1000.0,
            "velocity_scale_m_s": 1.0e-5,
            "flow_rate_m3_s": 1.0e-8,
            "temperature_scale_k": 300.0,
            "interfacial_tension_n_m": 0.03,
            "diffusivity_m2_s": 1.0e-9,
            "delta_density_kg_m3": 200.0,
            "pressure_drop_pa": 2.0e5,
            "elapsed_time_s": 50.0,
            "mobility_displacing": 3.0,
            "mobility_displaced": 2.0,
        }
    )


if __name__ == "__main__":
    print(run_benchmark())
