"""Clone MXSPR006 into the 300 mm concecpt IMEX case (skill clone + patch)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from probes import PROBE_DIAMETER_M, all_sensors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEMPLATE = ROOT / "black_oil" / "validation" / "cmg_channel_3d" / "mxspr006_channel.dat"
DAT = HERE / "mxspr006_concept.dat"
TRUTH = HERE / "truth_concept.json"
PATCH_GRID = HERE / "patch_grid.json"
PATCH_WELLS = HERE / "patch_wells.json"
PATCH_SCHEDULE = HERE / "patch_schedule.json"
CMG_SCRIPTS = Path(r"C:\Users\xuzhihao\.codex\skills\cmg-suite\scripts")

NX = NY = NZ = 15
BOX_M = 0.30
DX_M = BOX_M / NX
M_TO_FT = 1.0 / 0.3048
FT_TO_M = 0.3048
MD_TO_M2 = 9.869233e-16
DX_FT = DX_M * M_TO_FT
K_MD = 100.0
KZ_MD = 10.0
PHI = 0.28
RW_FT = 0.00984
STW_STBDAY = 0.05
STO_STBDAY = 0.05
INJ_I, INJ_J = 1, 8
PROD_I, PROD_J = 15, 8
TIMES_DAY = [0.1, 0.5, 1.0, 2.0, 5.0]
TIME_FMT = {0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 5.0: "5.0"}
DTMAX = 0.05
DTWELL = 0.05
DTOP_FT = 10.0
PRES_PSI = 3000.0
SWI = 0.20


def _fmt_plane(value: float, n: int) -> str:
    vals = [f"{value:.6g}"] * n
    lines = ["   " + "  ".join(vals[i : i + 10]) for i in range(0, n, 10)]
    return "\n".join(lines)


def _fmt_all(value: float) -> str:
    return _fmt_plane(value, NX * NY * NZ)


def _perf(well: int, i: int, j: int) -> str:
    lines = [f"   *PERF *GEO {well}", "   ** if      jf     kf     ff"]
    for k in range(1, NZ + 1):
        lines.append(f"       {i}      {j}      {k}     1.0")
    return "\n".join(lines)


def _grid_block() -> str:
    return f"""   *GRID *CART {NX} {NY} {NZ}
   *KDIR *DOWN
   *DI *CON
   {DX_FT:.8f}
   *DJ *CON
   {DX_FT:.8f}
   *DK *CON
   {DX_FT:.8f}

   ** Flat lid. Mountain z only places probes; not DTOP relief.
   *DTOP
{_fmt_plane(DTOP_FT, NX * NY)}

   *POR *CON
    {PHI:.3f}

   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{_fmt_all(K_MD)}
   *PERMJ *ALL
{_fmt_all(K_MD)}
   *PERMK *ALL
{_fmt_all(KZ_MD)}

"""


def write_truth(sensors: list[dict]) -> dict:
    n_p = sum(1 for s in sensors if s["kind"] == "pressure")
    n_s = sum(1 for s in sensors if s["kind"] == "saturation")
    truth = {
        "description": "300 mm single-layer concecpt IMEX case; one xyz one kind",
        "source_template": str(TEMPLATE.relative_to(ROOT)).replace("\\", "/"),
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "di_ft": DX_FT,
            "dj_ft": DX_FT,
            "dk_ft": DX_FT,
            "dx_m": DX_M,
            "kdir": "DOWN",
            "k1": "top",
            "size_m": [BOX_M, BOX_M, BOX_M],
            "dtop_ft": DTOP_FT,
        },
        "rock": {
            "type": "homogeneous",
            "k_md": K_MD,
            "kz_md": KZ_MD,
            "k_m2": K_MD * MD_TO_M2,
            "kz_m2": KZ_MD * MD_TO_M2,
            "phi": PHI,
        },
        "controls": {
            "inj": "rate",
            "prod": "pressure",
            "stw_stbday": STW_STBDAY,
            "sto_stbday": STO_STBDAY,
            "pres_psi": PRES_PSI,
            "swi": SWI,
            "rw_ft": RW_FT,
            "probe_diameter_m": PROBE_DIAMETER_M,
        },
        "wells": {
            "INJ": {
                "i": INJ_I,
                "j": INJ_J,
                "k_cmg": list(range(1, NZ + 1)),
                "control": "rate",
            },
            "PROD": {
                "i": PROD_I,
                "j": PROD_J,
                "k_cmg": list(range(1, NZ + 1)),
                "control": "pressure",
            },
        },
        "times_day": TIMES_DAY,
        "history_end_day": 2.0,
        "sensors": sensors,
        "sensor_counts": {"pressure": n_p, "saturation": n_s, "total": len(sensors)},
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def write_patch_specs() -> None:
    grid_spec = {
        "operations": [
            {
                "type": "block_replace",
                "parameter_name": "grid_cart_15",
                "reason": "Replace VARI channel grid with 15^3 0.30 m CART, uniform 100 md",
                "start": "   *GRID *VARI 7 7 5",
                "end": "   *MODEL *BLACKOIL_SEAWATER",
                "replacement": _grid_block(),
            }
        ]
    }
    wells_spec = {
        "operations": [
            {
                "type": "replace",
                "parameter_name": "title1",
                "old": "'Undulating Channel Twin'",
                "new": "'Concepcpt 300mm single-layer lab'",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "title2",
                "old": "'7x7x5 DTOP mountain ridge channel'",
                "new": "'15x15x15 flat-DTOP homogeneous (lab 0.30 m)'",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "stw",
                "old": "*OPERATE   *MAX       *STW    5000.0",
                "new": f"*OPERATE   *MAX       *STW    {STW_STBDAY}",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "sto",
                "old": "*OPERATE   *MAX       *STO    2500",
                "new": f"*OPERATE   *MAX       *STO    {STO_STBDAY}",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "geometry",
                "old": "*GEOMETRY  *K  0.25   0.34    1.0     0.0",
                "new": f"*GEOMETRY  *K  {RW_FT:.5f}   0.34    1.0     0.0",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "dtwell",
                "old": "*DTWELL      1.0",
                "new": f"*DTWELL      {DTWELL}",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "perf_inj",
                "old": (
                    "   *PERF *GEO 1\n"
                    "   ** if      jf     kf     ff\n"
                    "       1      1      2     1.0\n"
                    "       1      1      3     1.0\n"
                    "       1      1      4     1.0\n"
                ),
                "new": _perf(1, INJ_I, INJ_J) + "\n",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "perf_prod",
                "old": (
                    "   *PERF *GEO 2\n"
                    "   ** if      jf     kf     ff\n"
                    "       7      7      2     1.0\n"
                    "       7      7      3     1.0\n"
                    "       7      7      4     1.0\n"
                ),
                "new": _perf(2, PROD_I, PROD_J) + "\n",
                "count": 1,
            },
        ]
    }
    time_block = "\n".join(f"   *TIME      {TIME_FMT[t]}" for t in TIMES_DAY) + "\n\n"
    schedule_spec = {
        "operations": [
            {
                "type": "replace",
                "parameter_name": "wprn_grid",
                "old": "*WPRN   *GRID 5",
                "new": "*WPRN   *GRID 1",
                "count": 1,
            },
            {
                "type": "replace",
                "parameter_name": "dtmax",
                "old": "*DTMAX  62.",
                "new": f"*DTMAX  {DTMAX}",
                "count": 1,
            },
            {
                "type": "block_replace",
                "parameter_name": "times",
                "reason": "Lab window 0.1-5 day instead of multi-year DATEs",
                "start": "   *TIME      1.0",
                "end": "   *STOP",
                "replacement": time_block,
            },
        ]
    }
    PATCH_GRID.write_text(json.dumps(grid_spec, indent=2), encoding="utf-8")
    PATCH_WELLS.write_text(json.dumps(wells_spec, indent=2), encoding="utf-8")
    PATCH_SCHEDULE.write_text(json.dumps(schedule_spec, indent=2), encoding="utf-8")


def _run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"empty stdout: {cmd}")
    return json.loads(out)


def apply_clone_and_patches() -> dict:
    py = sys.executable
    clone = _run_json(
        [
            py,
            str(CMG_SCRIPTS / "clone_case.py"),
            "--source",
            str(TEMPLATE),
            "--dest",
            str(DAT),
            "--execute",
            "--overwrite",
            "--pretty",
        ]
    )
    patches = []
    for spec in (PATCH_GRID, PATCH_WELLS, PATCH_SCHEDULE):
        patches.append(
            _run_json(
                [
                    py,
                    str(CMG_SCRIPTS / "patch_dat.py"),
                    "--file",
                    str(DAT),
                    "--spec",
                    str(spec),
                    "--execute",
                    "--pretty",
                ]
            )
        )
    return {"clone": clone, "patches": patches}


def build(*, execute: bool) -> dict:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    sensors = all_sensors()
    truth = write_truth(sensors)
    write_patch_specs()
    result: dict = {
        "dat": str(DAT),
        "truth": str(TRUTH),
        "n_sensors": len(sensors),
        "sensor_counts": truth["sensor_counts"],
        "patches": [str(PATCH_GRID), str(PATCH_WELLS), str(PATCH_SCHEDULE)],
        "mode": "execute" if execute else "specs_only",
    }
    if execute:
        result["apply"] = apply_clone_and_patches()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build concecpt IMEX case via clone/patch specs.")
    parser.add_argument("--execute", action="store_true", help="Clone MXSPR006 and apply the three patches.")
    args = parser.parse_args()
    print(json.dumps(build(execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
