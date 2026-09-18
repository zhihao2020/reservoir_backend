"""Reproduce physical_3d well diagnostics without running GEM.

Connection rates come from ``well_molar_sources(..., connections=)``, the same
calculation the Newton residual uses. Do not recompute injector volume with
injectate mobility.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reservoir_backend.twin.cmg_benchmark import (
    load_twin_case,
    theta_true_from_twin,
    load_hidden_truth,
    rmse,
    parse_gem_out_summaries,
    gem_summary_at,
    cache_manifest_matches,
    forward_cache_manifest,
)
from reservoir_backend.comp.properties import flash_state
from reservoir_backend.comp.wells import well_molar_sources
from scripts.lab_v1_cmg_compare_plot import _field_at


def diagnostics(cache=None):
    case = ROOT / "examples/lab_v1/cmg_gem/physical_3d/case.yaml"
    twin = load_twin_case(case)
    theta = theta_true_from_twin(twin)
    rock = twin.rock_from_theta(theta)
    times = np.array([0.0, 8.64, 864.0])
    manifest = None
    if cache is None:
        traj = twin.simulate(parameters=theta, t_end=864.0, report_times=times)
    else:
        from types import SimpleNamespace

        with np.load(cache) as saved:
            times = saved["times_s"].copy()
            if bool(saved["truncated"]) or float(times[-1]) != 864.0:
                raise ValueError("diagnostics need an untruncated 864 s cache")
            gm = twin.physics.geomech
            manifest = forward_cache_manifest(
                case=case,
                times_s=times,
                n_cells=twin.grid.n_cells,
                shape=(twin.grid.nx, twin.grid.ny, twin.grid.nz),
                t_end=864.0,
                geomech=bool(gm is not None and getattr(gm, "enabled", False)),
            )
            if not cache_manifest_matches(saved, manifest):
                raise ValueError("cache manifest does not match current solver/case; refuse stale diagnostics")
            if "moles" not in saved.files:
                raise ValueError("cache has no moles; refuse composition reconstructed from z only")
            states = [
                SimpleNamespace(pressure=p.copy(), moles=n.copy())
                for p, n in zip(saved["pressure"], saved["moles"])
            ]
            traj = SimpleNamespace(times_s=times, states=states, reports=[])
    np.testing.assert_allclose(traj.times_s, times, rtol=0, atol=1.0e-9)
    truth = load_hidden_truth(case.parent / "export" / "hidden")
    source = ROOT / "results/lab_v1/cmg_gem_physical_3d/sanwei_co2.out"
    raw = source.read_text(errors="replace")
    summaries = parse_gem_out_summaries(raw)
    gem_rep = gem_summary_at(summaries, 864.0)
    connections_gem = [] if gem_rep is None else gem_rep["connections"]
    controls = {(c.port_name, c.kind): c for c in twin.experiment.controls}
    result = {
        "source": str(source.relative_to(ROOT)),
        "cache": None if cache is None else str(cache),
        "cache_manifest": manifest,
        "note": "Connection rows use well_molar_sources. GEM WI is not printed; ours is geometric. Not an M2a PASS.",
        "gem_864": gem_rep,
        "snapshots": [],
    }
    for t, st in zip(times[1:], traj.states[1:]):
        props = flash_state(twin.physics.fluid, st.pressure, st.moles)
        pg = _field_at(truth.times_s, truth.pressure, t)
        recs: list[dict] = []
        q, rates, bhp = well_molar_sources(
            twin.grid, rock, twin.ports, controls, st.pressure, props, twin.physics.fluid, t,
            connections=recs,
        )
        entry = {
            "t_s": float(t),
            "rmse_p_pa": rmse(st.pressure, pg),
            "gem_map_min_max_pa": [float(pg.min()), float(pg.max())],
            "ours_min_max_pa": [float(st.pressure.min()), float(st.pressure.max())],
            "port_rates_mol_s": {k: float(v) for k, v in rates.items() if ":" not in k},
            "port_bhp_pa": {k: float(v) for k, v in bhp.items()},
            "wells": [],
        }
        for port in twin.ports:
            rows = [r for r in recs if r["well"] == port.name]
            gem_rows = [r for r in connections_gem if r["well"] == port.name]
            table = []
            for row in rows:
                gemrow = next((g for g in gem_rows if g["ijk"] == row["gem_ijk"]), None)
                table.append(
                    {
                        **row,
                        "gem_bhp_minus_block_pa": None if gemrow is None else gemrow["bhp_minus_block_pa"],
                        "gem_reservoir_m3_day": None if gemrow is None else gemrow["reservoir_m3_day"],
                    }
                )
            n_fwd = sum(1 for r in rows if r["forward"])
            n_back = sum(1 for r in rows if r["backflow"])
            entry["wells"].append(
                {
                    "well": port.name,
                    "connections": len(rows),
                    "n_forward": n_fwd,
                    "n_backflow": n_back,
                    "bhp_pa": bhp[port.name],
                    "source_mol_s": float(q[np.asarray(port.cell_ids)].sum()),
                    "connection_table": table,
                }
            )
        result["snapshots"].append(entry)
    result["accepted_steps"] = len(traj.reports) if getattr(traj, "reports", None) else None
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    result = diagnostics(args.cache)
    dest = ROOT / "results/lab_v1/cmg_gem_physical_3d_compare/well_diagnostics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"t_snapshots": [s["t_s"] for s in result["snapshots"]], "cache": result["cache"]}, indent=2))
