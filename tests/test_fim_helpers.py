"""FIM helpers and rename-guard checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reservoir_backend.solver.fi import (
    cell_cnv_ok,
    clip_saturation_increment,
    dt_from_newton_iters,
    global_mass_balance_ok,
    scale_newton_update,
)


def test_clip_and_scale_newton_update() -> None:
    n = 4
    unsat = np.array([True, True, False, False])
    du = np.concatenate([np.full(n, 1.0e6), np.full(n, 0.5), np.array([10.0, 10.0, 0.5, 0.5])])
    out = clip_saturation_increment(du, n, unsat, rs_ref=50.0, pref=1.0e5, ds_max=0.2, dp_rel_max=0.2)
    assert float(np.max(np.abs(out[:n]))) <= 0.2 * 1.0e5 + 1.0e-6
    assert float(np.max(np.abs(out[n : 2 * n]))) <= 0.2 + 1.0e-9
    scaled = scale_newton_update(out, alpha=0.5)
    assert np.allclose(scaled, 0.5 * out)


def test_cnv_and_mass_balance_helpers() -> None:
    n = 3
    scale = np.ones(3 * n)
    res_ok = np.full(3 * n, 1.0e-4)
    assert cell_cnv_ok(res_ok, scale, tol=1.0e-3)
    assert global_mass_balance_ok(res_ok, n, scale, tol=1.0e-3)
    res_bad = res_ok.copy()
    res_bad[0] = 10.0
    assert not cell_cnv_ok(res_bad, scale, tol=1.0e-3)


def test_dt_from_newton_iters_grows_when_easy() -> None:
    dt_new = dt_from_newton_iters(10.0, 2, dt_min=0.1, dt_max=100.0)
    assert dt_new > 10.0
    dt_hard = dt_from_newton_iters(10.0, 12, dt_min=0.1, dt_max=100.0)
    assert dt_hard < 10.0


def test_product_tree_avoids_upstream_fim_names() -> None:
    root = Path(__file__).resolve().parents[1] / "reservoir_backend"
    banned = (
        "SimulatorFullyImplicit",
        "FIBlackoilModel",
        "CompositionalMultiphaseFVM",
        "AppleyardChop",
    )
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            if name in text:
                hits.append(f"{path.name}:{name}")
    assert hits == []
