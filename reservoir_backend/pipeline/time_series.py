"""Multi-time reconstruction, shape inference, and refine loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.mesh_refine import map_field_to_mesh, refine_mesh_by_indicator
from reservoir_backend.pipeline.run import run_time_slice, save_fields
from reservoir_backend.pipeline.shape_indicator import (
    enhance_permeability_from_indicator,
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
    mode: str = "point_first",
    assimilate_k: bool = False,
    esmda_ne: int = 16,
    esmda_assimilations: int = 3,
    esmda_max_times: int = 5,
    refine_dynamic_k: bool = True,
    esmda_second_pass: bool = True,
) -> list[FieldBundle]:
    """Sequential multi-time inversion from wells + probes.

    Each ``SensorSample`` is one time stamp of:

    - injectors / producers: p and/or S, optional rates
    - ``observer_p``: pressure-only probes
    - ``observer_s``: saturation-only probes

    Default ``point_first`` workflow per time, carrying full-grid k/φ as the
    prior into the next time (time-series inversion).

    If ``assimilate_k`` is True, run ES-MDA on log(k) first (using all pressure
    hard data: wells + observer_p), then point-first series with the ensemble
    mean as permeability prior. Optional second ES-MDA pass and dynamic-k
    refinement from multi-time ΔSw / pressure indicator further improve paths.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: s.time)

    k_prior0: float | NDArray[np.float64] = permeability_prior_m2
    phi_prior0: float | NDArray[np.float64] = porosity_prior
    esmda_notes: list[str] = []
    phi0 = float(np.mean(np.asarray(porosity_prior, dtype=float)))
    if assimilate_k:
        from reservoir_backend.pipeline.esmda import run_esmda_permeability

        # subsample times for speed while spanning the series
        es_samples = _subsample_times(samples, int(esmda_max_times))
        k_mean = float(np.mean(np.asarray(permeability_prior_m2, dtype=float)))
        try:
            es = run_esmda_permeability(
                mesh,
                es_samples,
                ne=int(esmda_ne),
                n_assimilations=int(esmda_assimilations),
                k_mean=k_mean,
                logk_std=1.2,
                corr_len_cells=max(2.0, float(np.mean(mesh.grid.shape)) * 0.35),
                porosity_prior=phi0,
                viscosity_pa_s=viscosity_pa_s,
                n_k_iterations=1,
                seed=11,
                auto_localize=True,
            )
            k_prior0 = es.k_mean
            esmda_notes = [
                f"pre-series ES-MDA k assimilate (ne={esmda_ne}, "
                f"Na={esmda_assimilations}, n_times={len(es_samples)})",
                *es.notes[-4:],
            ]
            # second pass: tighter prior around first-pass mean
            if esmda_second_pass and len(es_samples) >= 2:
                try:
                    es2 = run_esmda_permeability(
                        mesh,
                        es_samples,
                        ne=max(8, int(esmda_ne) // 2),
                        n_assimilations=max(2, int(esmda_assimilations) - 1),
                        k_mean=float(np.exp(np.mean(np.log(np.clip(es.k_mean, 1e-30, None))))),
                        logk_std=0.65,
                        corr_len_cells=max(2.0, float(np.mean(mesh.grid.shape)) * 0.30),
                        porosity_prior=phi0,
                        viscosity_pa_s=viscosity_pa_s,
                        n_k_iterations=1,
                        seed=23,
                        auto_localize=True,
                        ensemble_inflation=1.01,
                    )
                    k_prior0 = 0.55 * es2.k_mean + 0.45 * es.k_mean
                    k_prior0 = np.clip(k_prior0, 1.0e-18, 1.0e-10)
                    esmda_notes.append("second-pass ES-MDA blended into k prior")
                except Exception as exc2:
                    esmda_notes.append(f"second-pass ES-MDA skipped ({exc2})")
        except Exception as exc:
            esmda_notes = [f"ES-MDA skipped ({exc}); plain time series"]

    history = _run_point_first_series(
        mesh,
        samples,
        k_prior0=k_prior0,
        phi_prior0=phi_prior0,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=n_k_iterations,
        mode=mode,
        assimilate_k=assimilate_k,
        esmda_notes=esmda_notes,
        k_esmda=(
            np.asarray(k_prior0, dtype=float)
            if assimilate_k and isinstance(k_prior0, np.ndarray)
            else None
        ),
    )

    if refine_dynamic_k and len(history) >= 2:
        history = _refine_tail_with_dynamic_k(
            mesh,
            samples,
            history,
            viscosity_pa_s=viscosity_pa_s,
            n_k_iterations=n_k_iterations,
            mode=mode,
        )
    return history


def _run_point_first_series(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    k_prior0: float | NDArray[np.float64],
    phi_prior0: float | NDArray[np.float64],
    viscosity_pa_s: float,
    n_k_iterations: int,
    mode: str,
    assimilate_k: bool,
    esmda_notes: list[str],
    k_esmda: NDArray[np.float64] | None,
) -> list[FieldBundle]:
    history: list[FieldBundle] = []
    prev: FieldBundle | None = None
    for sample in samples:
        dt = None
        if prev is not None:
            dt = float(sample.time - prev.time)
            if dt <= 0:
                dt = None
        if prev is None:
            k_prior: float | NDArray[np.float64] = k_prior0
            phi_prior: float | NDArray[np.float64] = phi_prior0
        else:
            if k_esmda is not None:
                k_prior = 0.45 * k_esmda + 0.55 * prev.permeability
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
            mode=mode,
        )
        if not any(n.startswith("time-series inversion") for n in bundle.notes):
            bundle.notes = (
                [
                    f"time-series inversion t={sample.time} "
                    f"(n_samples={len(samples)}, mode={mode}, "
                    f"assimilate_k={assimilate_k})"
                ]
                + esmda_notes
                + list(bundle.notes)
            )
        if k_esmda is not None:
            w = 0.55 if prev is None else 0.35
            bundle.permeability = (
                w * k_esmda + (1.0 - w) * np.asarray(bundle.permeability, dtype=float)
            )
            bundle.permeability = np.clip(bundle.permeability, 1.0e-18, 1.0e-10)
            bundle.notes = list(bundle.notes) + [f"k blended with ES-MDA prior (w={w})"]
        history.append(bundle)
        prev = bundle
    return history


def _refine_tail_with_dynamic_k(
    mesh: MeshBundle,
    samples: list[SensorSample],
    history: list[FieldBundle],
    *,
    viscosity_pa_s: float,
    n_k_iterations: int,
    mode: str,
) -> list[FieldBundle]:
    """Boost k on multi-time activity indicator and re-run last few slices."""
    ind, stats = infer_shape_indicator(
        mesh,
        history,
        sw_weight=1.6,
        k_weight=0.15,
        pressure_weight=0.75,
    )
    k_enh = enhance_permeability_from_indicator(
        history[-1].permeability, ind, strength=0.70
    )
    n = len(samples)
    start = max(0, n - 3)
    out = list(history[:start])
    prev = history[start - 1] if start > 0 else None
    for i in range(start, n):
        sample = samples[i]
        dt = None
        if prev is not None:
            dt = float(sample.time - prev.time)
            if dt <= 0:
                dt = None
        if prev is None:
            k_prior: float | NDArray[np.float64] = k_enh
            phi_prior: float | NDArray[np.float64] = history[0].porosity
        else:
            k_prior = 0.60 * k_enh + 0.40 * prev.permeability
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
            mode=mode,
        )
        bundle.permeability = 0.65 * k_enh + 0.35 * bundle.permeability
        bundle.permeability = np.clip(bundle.permeability, 1.0e-18, 1.0e-10)
        # keep multi-time stamp from the pre-refine history when present
        prior_notes = list(history[i].notes) if i < len(history) else []
        multi = [n for n in prior_notes if n.startswith("time-series inversion")]
        bundle.notes = (
            multi
            + list(bundle.notes)
            + [
                "dynamic-k refine from multi-time indicator",
                f"indicator_mean={stats.get('indicator_mean', float('nan')):.3f}",
            ]
        )
        out.append(bundle)
        prev = bundle
    return out


def _subsample_times(samples: list[SensorSample], max_times: int) -> list[SensorSample]:
    n = len(samples)
    if n <= max_times or max_times < 2:
        return list(samples)
    idx = np.linspace(0, n - 1, int(max_times))
    picked = sorted({int(round(i)) for i in idx})
    return [samples[i] for i in picked]


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
