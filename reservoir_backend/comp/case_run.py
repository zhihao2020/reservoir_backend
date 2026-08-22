"""Thin EXAMPLE case runner for the standalone comp kernel.

Loads a YAML under ``reservoir_backend/comp/cases/`` and runs the
already-tested 1 HZ inj + 4 HZ prod two-cycle schedule. Not FIM, not
DigitalTwin, not the product CLI. Fluids/K are EXAMPLE, not Jiyang GEM.

    python -m reservoir_backend.comp.case_run
    python -m reservoir_backend.comp.case_run reservoir_backend/comp/cases/hz_1inj4prod_two_cycle.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from reservoir_backend.comp.cycle import SECONDS_PER_DAY, run_hz_1inj4prod_cycles
from reservoir_backend.comp.step import accumulate_system
from reservoir_backend.comp.streak import example_two_region_k
from reservoir_backend.comp.well import example_co2_rich_stream, example_hz_1inj4prod_layout, example_hz_1inj4prod_wells
from reservoir_backend.eos.load import load_eos_mixture_yaml, resolve_fluid_yaml
from reservoir_backend.grid.cartesian import CartesianGrid

DEFAULT_CASE = Path(__file__).resolve().parent / "cases" / "hz_1inj4prod_two_cycle.yaml"


def load_case_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("EXAMPLE case YAML must be a mapping")
    return data


def _hist_ends(hists: list[list[float]] | None) -> tuple[float, float]:
    if not hists or not hists[0]:
        raise ValueError("cycle is missing a residual history")
    h = hists[0]
    return float(h[0]), float(h[-1])


def metrics_from_multi(name: str, marker: str, multi) -> dict[str, Any]:
    """Stdout/JSON payload: per-cycle ||R||, nsteps, underflow, Δn."""
    cycles: list[dict[str, Any]] = []
    all_dts: list[float] = []
    for rec in multi.cycles:
        led = rec.ledger
        dts = led.inject.dt_used + led.soak.dt_used + led.produce.dt_used
        all_dts.extend(dts)
        r_inj = _hist_ends(led.inject_residual_hists)
        r_prod = _hist_ends(led.produce_residual_hists)
        cycles.append(
            {
                "inject_R": [r_inj[0], r_inj[1]],
                "produce_R": [r_prod[0], r_prod[1]],
                "accepted_steps": int(led.accepted_steps),
                "underflow": bool(led.underflow),
                "delta_n": [float(x) for x in np.asarray(rec.delta_n, dtype=float)],
            }
        )
    return {
        "name": name,
        "marker": marker,
        "n_cycles": len(multi.cycles),
        "accepted_steps": int(sum(c["accepted_steps"] for c in cycles)),
        "underflow": bool(multi.underflow),
        "min_dt_s": float(min(all_dts)) if all_dts else None,
        "cycles": cycles,
        "note": "EXAMPLE fluids/K, not a Jiyang GEM card",
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        f"EXAMPLE case: {metrics['name']}",
        "fluids/K are EXAMPLE, not a Jiyang GEM card",
    ]
    for i, cyc in enumerate(metrics["cycles"], 1):
        ri0, ri1 = cyc["inject_R"]
        rp0, rp1 = cyc["produce_R"]
        lines.append(f"cycle {i} inject ||R|| {ri0:.6e} -> {ri1:.6e}")
        lines.append(f"cycle {i} produce ||R|| {rp0:.6e} -> {rp1:.6e}")
    lines.append(f"accepted nsteps {metrics['accepted_steps']}")
    lines.append(f"underflow {metrics['underflow']}")
    return "\n".join(lines)


def run_example_case(path: str | Path | None = None) -> dict[str, Any]:
    """Run the YAML case with the already-tested HZ 1+4 two-cycle physics."""
    cfg = load_case_yaml(DEFAULT_CASE if path is None else path)
    if str(cfg.get("marker", "")).upper() != "EXAMPLE":
        raise ValueError("case marker must be EXAMPLE")
    if str(cfg.get("pattern", "")) != "hz_1inj4prod":
        raise ValueError("this runner only executes pattern hz_1inj4prod")
    gcfg = cfg["grid"]
    fcfg = cfg["fluid"]
    rcfg = cfg["rock"]
    wcfg = cfg["wells"]
    scfg = cfg["schedule"]
    size = tuple(float(x) for x in gcfg["size_m"])
    grid = CartesianGrid.uniform(size, float(gcfg["spacing_m"]))
    grid, inj_cells, laterals, streak = example_hz_1inj4prod_layout(
        grid, n_perf=int(gcfg["n_perf"]), streak=str(gcfg["streak"])
    )
    k = example_two_region_k(
        grid,
        streak,
        k_matrix=float(rcfg["k_matrix_m2"]),
        k_streak=float(rcfg["k_streak_m2"]),
    )
    if not fcfg.get("eos_yaml"):
        raise ValueError(
            "case fluid.eos_yaml is required; refusing to invent GEM/Jiyang criticals"
        )
    mix = load_eos_mixture_yaml(resolve_fluid_yaml(fcfg["eos_yaml"])).subset(list(fcfg["components"]))
    p = np.full(grid.n_cells, float(wcfg["p_init_pa"]))
    z = np.tile(np.asarray(fcfg["z"], dtype=float), (grid.n_cells, 1))
    vp = float(rcfg["porosity"]) * grid.cell_volumes()
    fields = accumulate_system(z, float(fcfg["T"]), p, mix, vp)
    inj, prod = example_hz_1inj4prod_wells(
        grid,
        inj_cells,
        laterals,
        k,
        mix,
        inject_rate=float(wcfg["inject_rate_mol_s"]),
        produce_bhp=float(wcfg["produce_bhp_pa"]),
        z_stream=example_co2_rich_stream(mix),
    )
    produce_days = float(scfg["produce_seconds"]) / SECONDS_PER_DAY
    _fields, multi = run_hz_1inj4prod_cycles(
        fields,
        float(fcfg["T"]),
        p,
        mix,
        grid,
        k,
        inj,
        prod,
        vp,
        n_cycles=int(cfg["n_cycles"]),
        inject_days=float(scfg["inject_days"]),
        soak_days=float(scfg["soak_days"]),
        produce_days=produce_days,
        dt_init_days=float(scfg["dt_init_days"]),
        dt_max_days=float(scfg["dt_max_days"]),
    )
    return metrics_from_multi(str(cfg["name"]), str(cfg["marker"]), multi)


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run an EXAMPLE standalone comp case (not FIM).")
    parser.add_argument(
        "case",
        nargs="?",
        default=str(DEFAULT_CASE),
        help="path to EXAMPLE YAML (default: hz_1inj4prod_two_cycle)",
    )
    parser.add_argument("--json", dest="json_path", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)
    metrics = run_example_case(args.case)
    out = stdout or sys.stdout
    print(format_metrics(metrics), file=out)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
