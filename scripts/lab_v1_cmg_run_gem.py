"""Run the alignment GEM deck if a local GEM 2024 binary exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    find_gem_exe,
    init_flash_report,
    parse_gem_out_maps,
    run_gem,
    write_grid_csv,
    write_hidden_truth,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "lab_v1_dev.dat")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_gem_run")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args(argv)
    exe = find_gem_exe()
    rec = run_gem(args.deck, args.work, exe=exe, timeout_s=float(args.timeout))
    (Path(args.work)).mkdir(parents=True, exist_ok=True)
    (Path(args.work) / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in {"stdout_tail", "stderr_tail"}}, indent=2), flush=True)
    if rec.get("blocked"):
        return 2
    if not rec.get("ok"):
        return 1
    outs = rec.get("out_files") or []
    if outs:
        twin = load_lab_v1(dev=True)
        truth = parse_gem_out_maps(outs[0], nx=twin.grid.nx, ny=twin.grid.ny, nz=twin.grid.nz)
        hidden = Path(args.work) / "hidden"
        write_hidden_truth(hidden, truth)
        write_grid_csv(twin, hidden / "grid.csv")
        rec["hidden"] = str(hidden)
        rec["n_cells"] = truth.n_cells
        rec["t_end_s"] = float(truth.times_s[-1])
        flash = init_flash_report(outs[0])
        rec["init_flash"] = flash
        (Path(args.work) / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        (Path(args.work) / "init_flash.json").write_text(json.dumps(flash, indent=2), encoding="utf-8")
        print(json.dumps({"hidden": rec["hidden"], "t_end_s": rec["t_end_s"], "init_flash": flash}, indent=2), flush=True)
        if not flash["pass"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
