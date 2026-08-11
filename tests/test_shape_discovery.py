"""Multi-time shape discovery and synthetic channel twin validation."""

from __future__ import annotations

import numpy as np

from reservoir_backend.pipeline import (
    build_channel_twin,
    build_faulted_channel_twin,
    indicator_to_active_mask,
    mask_overlap,
    run_shape_discovery,
    run_time_series,
)


def test_enhance_permeability_from_indicator() -> None:
    from reservoir_backend.pipeline.shape_indicator import enhance_permeability_from_indicator

    k = np.full((2, 3, 4), 1.0e-13)
    ind = np.zeros((2, 3, 4))
    ind[0, 1, 2] = 1.0
    k2 = enhance_permeability_from_indicator(k, ind, strength=0.8)
    assert k2[0, 1, 2] > k[0, 0, 0]
    assert np.all(k2 > 0.0)


def test_run_time_series_sorted_and_shapes() -> None:
    twin = build_channel_twin(nx=8, ny=6, nz=3, n_times=3)
    # shuffle times to ensure sorting
    samples = list(reversed(twin.samples))
    history = run_time_series(twin.mesh, samples, permeability_prior_m2=1.0e-13)
    assert len(history) == 3
    assert all(h.pressure.shape == twin.mesh.grid.shape for h in history)
    assert history[0].time <= history[1].time <= history[2].time


def test_shape_discovery_finds_channel_corridor() -> None:
    twin = build_channel_twin(nx=10, ny=8, nz=4, n_times=4)
    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=1.0e-13,
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
    )
    assert result.indicator.shape == twin.mesh.grid.shape
    assert result.active_mask.shape == twin.mesh.grid.shape
    assert np.any(result.active_mask)
    assert result.fine_mesh is not None
    assert result.fine_mesh.n_cells > twin.mesh.n_cells
    assert len(result.fine_history) == len(twin.samples)

    metrics = mask_overlap(result.active_mask, twin.true_channel_mask)
    # Under-determined inverse: require better-than-random corridor recovery.
    assert metrics["dice"] > 0.10, metrics
    assert metrics["recall"] > 0.15, metrics
    # high-indicator cells should not be worse than domain-average by much
    high = result.indicator >= 0.4
    if np.any(high):
        channel_frac_high = float(np.mean(twin.true_channel_mask[high]))
        channel_frac_all = float(np.mean(twin.true_channel_mask))
        assert channel_frac_high + 0.05 >= channel_frac_all


def test_indicator_mask_dilate_and_fallback() -> None:
    twin = build_channel_twin(nx=6, ny=5, nz=3, n_times=2)
    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        refine=False,
        indicator_threshold=0.99,
    )
    # extreme threshold still yields a non-empty mask via quantile fallback
    mask = indicator_to_active_mask(result.indicator, threshold=0.99, dilate=0)
    assert np.any(mask)


def test_save_discovery_writes_artifacts(tmp_path) -> None:
    twin = build_channel_twin(nx=6, ny=5, nz=3, n_times=2)
    result = run_shape_discovery(twin.mesh, twin.samples, refine=False)
    from reservoir_backend.pipeline import save_discovery

    out = tmp_path / "disc"
    save_discovery(result, str(out))
    assert (out / "shape_indicator.npy").is_file()
    assert (out / "active_mask.npy").is_file()
    assert (out / "indicator_stats.json").is_file()
    assert (out / "coarse" / "t_0000" / "summary.json").is_file()


def test_faulted_channel_twin_discovery() -> None:
    twin = build_faulted_channel_twin(nx=10, ny=8, nz=4, n_times=4)
    assert twin.true_fault_mask is not None
    assert np.any(twin.true_fault_mask)
    assert np.any(twin.true_channel_mask)
    # fault and channel should be mostly disjoint
    overlap_fc = float(np.mean(twin.true_fault_mask & twin.true_channel_mask))
    assert overlap_fc < 0.05

    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=1.0e-13,
        refine=False,
        indicator_threshold=0.30,
    )
    metrics = mask_overlap(result.active_mask, twin.true_channel_mask)
    assert metrics["dice"] > 0.08, metrics
    # when the active mask is selective, sealing cells should not dominate it
    active_frac = float(np.mean(result.active_mask.astype(float)))
    if active_frac < 0.85:
        fault_frac = float(np.mean(result.active_mask[twin.true_fault_mask].astype(float)))
        assert fault_frac < 0.9, (fault_frac, active_frac)
    else:
        # near-full domain mask: require channel enrichment over random
        ch_frac_active = float(np.mean(twin.true_channel_mask[result.active_mask].astype(float)))
        ch_frac_all = float(np.mean(twin.true_channel_mask.astype(float)))
        assert ch_frac_active + 1e-12 >= ch_frac_all
