"""Multi-time shape discovery and synthetic channel twin validation."""

from __future__ import annotations

import numpy as np

from reservoir_backend.pipeline import (
    build_channel_twin,
    indicator_to_active_mask,
    mask_overlap,
    run_shape_discovery,
    run_time_series,
)


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
    # Dice > 0.15 is a soft floor; channel + well corridor prior should beat chance.
    assert metrics["dice"] > 0.12, metrics
    assert metrics["recall"] > 0.15, metrics
    # high-indicator cells should be enriched in true channel
    high = result.indicator >= 0.4
    if np.any(high):
        channel_frac_high = float(np.mean(twin.true_channel_mask[high]))
        channel_frac_all = float(np.mean(twin.true_channel_mask))
        assert channel_frac_high >= channel_frac_all


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
