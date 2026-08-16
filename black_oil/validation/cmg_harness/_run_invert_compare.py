"""Run invert on the CMG rulers and print vs the last pre-MRST journal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.validation.cmg_harness.journal import Journal
from reservoir_backend.validation.cmg_harness.run_one import run_suite

BASELINE = {
    "lab_layers": {
        "id": "t017",
        "hold": 0.616,
        "forecast": 0.481,
        "p_rmse_psi": 20.18,
        "p_rmse_demean_psi": 15.01,
        "sw_rmse": 0.081,
        "J": 1.876,
    },
    "fivespot": {
        "id": "t014",
        "hold": 0.673,
        "forecast": 3.180,
        "p_rmse_psi": 421.79,
        "p_rmse_demean_psi": 30.55,
        "sw_rmse": 0.033,
        "J": 1.709,
    },
    "fault": {
        "id": "t015",
        "hold": 0.699,
        "forecast": 2.885,
        "p_rmse_psi": 233.42,
        "p_rmse_demean_psi": 81.02,
        "sw_rmse": 0.057,
        "J": 1.907,
    },
    "channel": {
        "id": "t016",
        "hold": 0.796,
        "forecast": 3.021,
        "p_rmse_psi": 334.51,
        "p_rmse_demean_psi": 78.55,
        "sw_rmse": 0.062,
        "J": 2.058,
    },
}

KEYS = ("hold", "forecast", "p_rmse_psi", "p_rmse_demean_psi", "sw_rmse", "J")


def main() -> None:
    journal = Journal()
    report = run_suite(
        ["fivespot", "fault", "channel"],
        invert=True,
        fast=False,
        journal=journal,
    )
    out = Path(__file__).with_name("invert_compare_mrst.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", out)
    print()
    print(f"{'case':<12} {'metric':<18} {'before':>10} {'after':>10} {'delta':>10}")
    print("-" * 64)
    for row in report["cases"]:
        case = row["case"]
        sc = row.get("score") or {}
        base = BASELINE.get(case) or {}
        print(f"{case:<12} {'probe':<18} {'':>10} {str(row.get('probe')):>10}")
        for key in KEYS:
            old = base.get(key)
            new = sc.get(key)
            if old is None or new is None:
                print(f"{case:<12} {key:<18} {old!s:>10} {new!s:>10}")
                continue
            try:
                delta = float(new) - float(old)
                print(f"{case:<12} {key:<18} {float(old):10.3f} {float(new):10.3f} {delta:10.3f}")
            except (TypeError, ValueError):
                print(f"{case:<12} {key:<18} {old!s:>10} {new!s:>10}")
        th = row.get("theta") or []
        if len(th) >= 2:
            klo = float(np.exp(th[0])) / 9.869233e-16
            print(f"{case:<12} {'k_lo_md':<18} {'':>10} {klo:10.1f}")
            print(f"{case:<12} {'contrast':<18} {'':>10} {float(np.exp(th[1])):10.2f}")
        notes = (sc.get("notes") or [])[-4:]
        for n in notes:
            print(f"    note: {n}")
        print()
        _plot_case(case, th)


def _plot_case(case_id: str, theta: list) -> None:
    if not theta:
        return
    import matplotlib.pyplot as plt

    from reservoir_backend.validation.cmg_harness.adapter import build_twin
    from reservoir_backend.validation.cmg_harness.catalog import MD_TO_M2, get_case
    from reservoir_backend.validation.cmg_harness.score import maps_from_traj

    spec = get_case(case_id)
    twin, extra = build_twin(spec, with_observations=False)
    k_true = extra["k_true"]
    if k_true is None:
        return
    days = spec.history_days
    t_end = float(days[-1]) * 86400.0
    k_post = twin.parameterization.expand(np.asarray(theta, dtype=float))
    f_true = maps_from_traj(twin.simulate(twin.rock_from_k(k_true), t_end=t_end), days, twin.grid)
    f_post = maps_from_traj(twin.simulate(twin.rock_from_k(k_post), t_end=t_end), days, twin.grid)
    d = float(days[-1])
    cmg = extra["maps"][d]
    nz, ny, nx = twin.grid.nz, twin.grid.ny, twin.grid.nx
    k_mid = nz // 2
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    packs = [
        (ax[0, 0], np.log10(k_true.reshape(nz, ny, nx)[k_mid] / MD_TO_M2), "log10 K_CMG (md)"),
        (ax[0, 1], np.log10(k_post.reshape(nz, ny, nx)[k_mid] / MD_TO_M2), "log10 K_post (md)"),
        (ax[0, 2], cmg["p"][k_mid], f"CMG p (psi) t={d:.0f}d"),
        (ax[1, 0], f_true[d]["p"][k_mid], "F(K_CMG) p"),
        (ax[1, 1], f_post[d]["p"][k_mid], "F(K_post) p"),
        (ax[1, 2], f_post[d]["sw"][k_mid], "F(K_post) Sw"),
    ]
    for a, arr, title in packs:
        im = a.imshow(arr, origin="lower", aspect="auto")
        a.set_title(title, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    out = Path(__file__).with_name("figures")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{case_id}_cmg_vs_inv.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("figure", path)


if __name__ == "__main__":
    main()
