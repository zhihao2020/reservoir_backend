"""Run the physical_3d GEM deck if a local GEM 2024 binary exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    fill_hidden_rock,
    find_gem_exe,
    gem_kdir_down,
    load_twin_case,
    parse_gem_out_maps,
    run_gem,
    write_grid_csv,
    write_hidden_truth,
)

P3D = ROOT / "examples" / "lab_v1" / "cmg_gem" / "physical_3d"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=P3D / "sanwei_co2.dat")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_gem_physical_3d")
    p.add_argument("--case", type=Path, default=P3D / "case.yaml")
    p.add_argument("--timeout", type=float, default=14400.0)
    args = p.parse_args(argv)
    exe = find_gem_exe()
    rec = run_gem(args.deck, args.work, exe=exe, timeout_s=float(args.timeout))
    Path(args.work).mkdir(parents=True, exist_ok=True)
    (Path(args.work) / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in {"stdout_tail", "stderr_tail"}}, indent=2), flush=True)
    if rec.get("blocked"):
        return 2
    if not rec.get("ok"):
        return 1
    outs = rec.get("out_files") or []
    if outs:
        twin = load_twin_case(args.case)
        truth = parse_gem_out_maps(
            outs[0],
            nx=twin.grid.nx,
            ny=twin.grid.ny,
            nz=twin.grid.nz,
            kdir_down=gem_kdir_down(args.case),
        )
        rock = twin.rock_from_theta(np.zeros(twin.parameterization.n_params))
        truth = fill_hidden_rock(
            truth,
            porosity=float(rock.porosity.mean()),
            permeability_m2=float(rock.permeability.mean()),
        )
        hidden = Path(args.work) / "hidden"
        write_hidden_truth(hidden, truth)
        write_grid_csv(twin, hidden / "grid.csv")
        rec["hidden"] = str(hidden)
        rec["n_cells"] = truth.n_cells
        rec["t_end_s"] = float(truth.times_s[-1])
        (Path(args.work) / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({"hidden": rec["hidden"], "t_end_s": rec["t_end_s"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
