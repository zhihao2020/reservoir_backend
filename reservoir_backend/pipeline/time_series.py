"""Multi-time reconstruction, shape inference, and refine loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.mesh_refine import map_field_to_mesh, refine_mesh_by_indicator
from reservoir_backend.pipeline.run import run_time_slice, save_fields
from reservoir_backend.pipeline.shape_indicator import (
    indicator_to_active_mask,
    infer_shape_indicator,
)
from reservoir_backend.pipeline.state import FieldBundle, MeshBundle, SensorSample


@dataclass
class DiscoveryResult:
    """Outputs of multi-time discovery + optional refine."""

    coarse_mesh: MeshBundle
    history: list[FieldBundle]
    indicator: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    indicator_stats: dict[str, float]
    fine_mesh: MeshBundle | None = None
    refine_stats: dict[str, float] = field(default_factory=dict)
    fine_history: list[FieldBundle] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_time_series(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    n_k_iterations: int = 2,
) -> list[FieldBundle]:
    """Run sequential time slices; carry k/φ arrays as priors between times."""
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: s.time)
    history: list[FieldBundle] = []
    prev: FieldBundle | None = None
    for sample in samples:
        dt = None
        if prev is not None:
            dt = float(sample.time - prev.time)
            if dt <= 0:
                dt = None
        # first step uses scalar/array prior; later steps inherit previous fields
        k_prior: float | NDArray[np.float64]
        phi_prior: float | NDArray[np.float64]
        if prev is None:
            k_prior = permeability_prior_m2
            phi_prior = porosity_prior
        else:
            k_prior = prev.permeability
            phi_prior = prev.porosity
        bundle = run_time_slice(
            mesh,
            sample,
            permeability_prior_m2=k_prior,
            porosity_prior=phi_prior,
            viscosity_pa_s=viscosity_pa_s,
            previous=prev,
            dt=dt,
            n_k_iterations=n_k_iterations,
        )
        history.append(bundle)
        prev = bundle
    return history


def run_shape_discovery(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    refine: bool = True,
    refine_factor: int = 2,
    indicator_threshold: float = 0.35,
    n_k_iterations: int = 2,
) -> DiscoveryResult:
    """Multi-time reconstruct → infer shape → optional mesh refine + re-run."""
    history = run_time_series(
        mesh,
        samples,
        permeability_prior_m2=permeability_prior_m2,
        porosity_prior=porosity_prior,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=n_k_iterations,
    )
    indicator, stats = infer_shape_indicator(mesh, history)
    active = indicator_to_active_mask(indicator, threshold=indicator_threshold, dilate=1)
    notes = [
        "multi-time reconstruction completed",
        f"shape indicator active_fraction={stats['active_fraction_at_0.4']:.3f}",
        f"k-pressure iterations per slice={n_k_iterations}",
    ]

    fine_mesh = None
    refine_stats: dict[str, float] = {}
    fine_history: list[FieldBundle] = []

    if refine:
        fine_mesh, refine_stats = refine_mesh_by_indicator(
            mesh,
            indicator,
            factor=refine_factor,
            threshold=indicator_threshold,
        )
        # map last coarse k field onto fine mesh as array prior
        k_mapped = map_field_to_mesh(mesh, history[-1].permeability, fine_mesh)
        phi_mapped = map_field_to_mesh(mesh, history[-1].porosity, fine_mesh)
        notes.append(
            f"refined mesh cells={int(refine_stats['fine_cells'])}; "
            f"mapped k mean={float(np.mean(k_mapped)):.3e}"
        )
        fine_history = run_time_series(
            fine_mesh,
            samples,
            permeability_prior_m2=k_mapped,
            porosity_prior=phi_mapped,
            viscosity_pa_s=viscosity_pa_s,
            n_k_iterations=n_k_iterations,
        )

    return DiscoveryResult(
        coarse_mesh=mesh,
        history=history,
        indicator=indicator,
        active_mask=active,
        indicator_stats=stats,
        fine_mesh=fine_mesh,
        refine_stats=refine_stats,
        fine_history=fine_history,
        notes=notes,
    )


def save_discovery(result: DiscoveryResult, output_dir: str) -> None:
    """Persist coarse history, indicator, and fine history if present."""
    from pathlib import Path

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for i, fb in enumerate(result.history):
        save_fields(result.coarse_mesh, fb, root / "coarse" / f"t_{i:04d}")
    np.save(root / "shape_indicator.npy", result.indicator)
    np.save(root / "active_mask.npy", result.active_mask.astype(np.uint8))
    (root / "indicator_stats.json").write_text(
        __import__("json").dumps(result.indicator_stats, indent=2),
        encoding="utf-8",
    )
    if result.fine_mesh is not None and result.fine_history:
        for i, fb in enumerate(result.fine_history):
            save_fields(result.fine_mesh, fb, root / "fine" / f"t_{i:04d}")
        (root / "refine_stats.json").write_text(
            __import__("json").dumps(result.refine_stats, indent=2),
            encoding="utf-8",
        )
    (root / "notes.json").write_text(
        __import__("json").dumps({"notes": result.notes}, indent=2),
        encoding="utf-8",
    )
