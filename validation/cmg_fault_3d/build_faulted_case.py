"""Build IMEX waterflood case with structural + sealing fault.

Base: cloned mxspr006 (seawater black-oil). Fault patterns from official
mxgeo002 (*FAULT throw) and hrw/wwm (*TRANSI seal).

Geometry:
  9x9x4 CART (FAULT throw is supported on CART; DTOP undulation is not)
  Structural throw along a mid-field fault trace
  Sealing TRANSI=0 on most of the fault plane + a leaky window for channel path
  High-k channel offset across the fault (juxtaposition / dogleg)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_fault.dat"
TRUTH = HERE / "truth_fault.json"


def build_geometry(nx: int = 9, ny: int = 9, nz: int = 4):
    di = dj = 300.0
    dk = np.array([45.0, 35.0, 25.0, 20.0])  # k=1 bottom
    # Fault plane between i=5 and i=6 (TRANSI on i=5 faces)
    fault_i = 5
    # Structural FAULT throw traces (mxgeo002 style): throw, then i-range j-range lines
    # Use throw 25 ft, trace roughly N-S with a step
    fault_throw_ft = 25.0
    fault_trace = [
        # (i1, i2, j1, j2) CMG 1-based inclusive ranges for *FAULT cards after throw
        (2, 2, 1, 3),
        (3, 3, 1, 5),
        (4, 4, 1, 7),
        (5, 5, 1, 9),
        (6, 9, 1, 9),
    ]

    # Sealing: full plane closed except leaky window near j=7:9
    seal_js = list(range(1, 7))  # sealed j=1..6
    window_js = list(range(7, 10))  # leaky j=7..9
    seal_mult = 0.0
    window_mult = 0.05

    # High-k channel dogleg: left block corridor j~5, right block offset j~8
    channel: list[list[int]] = []
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            if i <= fault_i:
                on = abs(j - 5) <= 1
            else:
                on = abs(j - 8) <= 1
            # connect through fault window at j=7,8 near plane
            if i in (fault_i, fault_i + 1) and j in (7, 8):
                on = True
            if not on:
                continue
            # k stack: thicker pay mid-layers
            for k in (2, 3, 4):
                channel.append([i, j, k])

    seen: set[tuple[int, int, int]] = set()
    uniq = []
    for b in channel:
        key = (b[0], b[1], b[2])
        if key not in seen:
            seen.add(key)
            uniq.append(b)

    inj = {"i": 1, "j": 5, "k_perfs": [2, 3, 4]}
    prod = {"i": 9, "j": 8, "k_perfs": [2, 3, 4]}
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "di": di,
        "dj": dj,
        "dk": dk,
        "fault_i": fault_i,
        "fault_throw_ft": fault_throw_ft,
        "fault_trace": fault_trace,
        "seal_js": seal_js,
        "window_js": window_js,
        "seal_mult": seal_mult,
        "window_mult": window_mult,
        "channel": uniq,
        "inj": inj,
        "prod": prod,
    }


def _mod_cells(blocks: list[list[int]], value: float) -> str:
    return "\n".join(f"      {i}:{i} {j}:{j} {k}:{k} = {value}" for i, j, k in blocks)


def _fault_block(throw: float, trace: list[tuple[int, int, int, int]]) -> str:
    # First line includes throw; continuation lines are i-range j-range
    lines = ["   ** Structural fault throw (ft) and plan-view trace (mxgeo002 style)"]
    first = True
    for i1, i2, j1, j2 in trace:
        i_txt = f"{i1}" if i1 == i2 else f"{i1}:{i2}"
        j_txt = f"{j1}" if j1 == j2 else f"{j1}:{j2}"
        if first:
            lines.append(f"   *FAULT   {throw:.1f}    {i_txt}   {j_txt}")
            first = False
        else:
            lines.append(f"                  {i_txt}   {j_txt}")
    return "\n".join(lines)


def _transi_block(g: dict) -> str:
    fi = g["fault_i"]
    nz = g["nz"]
    lines = [
        "   ** Hydraulic fault: seal most of plane, leaky window for channel path",
        "   *TRANSI *CON 1.0",
        "   *MOD",
    ]
    # seal
    j1, j2 = min(g["seal_js"]), max(g["seal_js"])
    lines.append(f"      {fi}:{fi} {j1}:{j2} 1:{nz} = {g['seal_mult']}")
    # window
    j1, j2 = min(g["window_js"]), max(g["window_js"])
    lines.append(f"      {fi}:{fi} {j1}:{j2} 1:{nz} = {g['window_mult']}")
    return "\n".join(lines)


def apply_to_dat(text: str, g: dict) -> str:
    nx, ny, nz = g["nx"], g["ny"], g["nz"]
    di, dj, dk = g["di"], g["dj"], g["dk"]
    channel = g["channel"]
    mod_h = _mod_cells(channel, 2000.0)
    mod_v = _mod_cells(channel, 150.0)

    grid_block = f"""   *GRID *CART {nx} {ny} {nz}
   *KDIR *DOWN
   *DI *CON
   {di}
   *DJ *CON
   {dj}
   *DK *KVAR
     {"  ".join(f"{v:.1f}" for v in dk)}

   *DEPTH 1 1 1 2000.0

   *POR *CON
    0.300

   *PRPOR    14.7
   *CPOR     3.0E-6

   ** Background + offset high-k channel (dogleg across fault)
   *PERMI *CON
      40.0
   *MOD
{mod_h}
   *PERMJ *CON
      40.0
   *MOD
{mod_h}
   *PERMK *CON
      15.0
   *MOD
{mod_v}

{_fault_block(g["fault_throw_ft"], g["fault_trace"])}

{_transi_block(g)}
"""

    start = text.find("*GRID *CART")
    if start < 0:
        start = text.find("*GRID *VARI")
    if start < 0:
        raise ValueError("GRID not found")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("*MODEL *BLACKOIL")
    if end < 0:
        raise ValueError("*MODEL not found")
    banner = text.rfind("********************************************************************************", 0, end)
    if banner > start:
        end = banner
    new_text = text[:start] + grid_block + "\n\n   " + text[end:]

    new_text = new_text.replace("'Seawater Test Problem'", "'Faulted Channel Twin'")
    new_text = new_text.replace(
        "'3d Seawater Test Case in a 7x7x3 grid configuration.'",
        "'9x9x4 structural+sealing fault with offset channel'",
    )

    # Reporting
    new_text = new_text.replace("*WPRN   *GRID 1", "*WPRN   *GRID 5\n   *OUTPRN *GRID *SO *SG *SW *PRES")

    # Wells: rename positions
    inj = g["inj"]
    prod = g["prod"]

    def make_perf(wellnum: int, i: int, j: int, ks: list[int]) -> str:
        lines = [f"   *PERF *GEO {wellnum}", "   ** if      jf     kf     ff"]
        for k in ks:
            lines.append(f"       {i}      {j}      {k}     1.0")
        return "\n".join(lines)

    new_text = re.sub(
        r"   \*PERF \*GEO 1\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(1, inj["i"], inj["j"], inj["k_perfs"]) + "\n\n",
        new_text,
        count=1,
    )
    new_text = re.sub(
        r"   \*PERF \*GEO\s*2\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(2, prod["i"], prod["j"], prod["k_perfs"]) + "\n\n",
        new_text,
        count=1,
    )

    # Short schedule (same pattern as channel case)
    new_text = re.sub(
        r"   \*DATE 1988 01 01\n   \*DATE 1989 01 01\n   \*DATE 1990 01 01\n   \*DATE 1991 01 01\n\n   \*INCOMPWL 1\n     0\.0 0\.90\n\n   \*DATE 1992 01 01\n   \*DATE 1993 01 01\n   \*DATE 1994 01 01\n   \*DATE 1995 01 01\n\n   \*SCLRMV-WELL 2\n    0\.75\n\n   \*SCLTBL-WELL 2\n    2  1\n\n   \*DATE 1996 01 01\n   \*DATE 1997 01 01\n   \*DATE 1998 01 01\n\n   \*STOP",
        "   *DATE 1988 01 01\n   *DATE 1988 07 01\n   *DATE 1989 01 01\n   *DATE 1989 07 01\n   *DATE 1990 01 01\n\n   *STOP",
        new_text,
        count=1,
    )
    return new_text


def write_truth(g: dict) -> dict:
    truth = {
        "description": "Structural throw + sealing fault with offset high-k channel",
        "grid": {
            "nx": g["nx"],
            "ny": g["ny"],
            "nz": g["nz"],
            "di_ft": g["di"],
            "dj_ft": g["dj"],
            "dk_ft": g["dk"].tolist(),
            "kdir": "DOWN",
            "k1": "bottom",
            "grid_type": "CART",
        },
        "fault": {
            "type": "structural_throw_plus_sealing_transmissibility",
            "plane": f"I-face between i={g['fault_i']} and i={g['fault_i']+1}",
            "throw_ft": g["fault_throw_ft"],
            "trace_ij_ranges": [
                {"i1": a, "i2": b, "j1": c, "j2": d} for a, b, c, d in g["fault_trace"]
            ],
            "seal_j": g["seal_js"],
            "window_j": g["window_js"],
            "seal_transi_mult": g["seal_mult"],
            "window_transi_mult": g["window_mult"],
            "reference_sample": "IMEX TPL geo/mxgeo002.dat (*FAULT) + hrw TRANSI seal pattern",
        },
        "units": "FIELD (md, ft, psi)",
        "wells": {
            "INJ": {
                "name": "SEAWATER INJECTOR",
                "i": g["inj"]["i"],
                "j": g["inj"]["j"],
                "k_perfs": g["inj"]["k_perfs"],
            },
            "PROD": {
                "name": "PRODUCER",
                "i": g["prod"]["i"],
                "j": g["prod"]["j"],
                "k_perfs": g["prod"]["k_perfs"],
            },
        },
        "background_perm_md": {"kx": 40.0, "ky": 40.0, "kz": 15.0},
        "channel_perm_md": {"kx": 2000.0, "ky": 2000.0, "kz": 150.0},
        "channel_blocks_ijk": g["channel"],
        "fault_seal_cells_note": (
            "Seal is face-based TRANSI, not cell nulls. Channel doglegs through j-window."
        ),
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main() -> int:
    if not DAT.exists():
        raise SystemExit(f"missing base DAT: {DAT} (clone mxspr006 first)")
    g = build_geometry()
    text = DAT.read_text(encoding="latin-1")
    # If already transformed, re-clone is safer; allow rebuild from current content
    # if MODEL missing or grid already patched, still replace GRID..MODEL section
    new_text = apply_to_dat(text, g)
    DAT.write_text(new_text, encoding="latin-1")
    truth = write_truth(g)
    print(
        json.dumps(
            {
                "dat": str(DAT),
                "channel_blocks": len(g["channel"]),
                "fault_throw_ft": g["fault_throw_ft"],
                "fault_i": g["fault_i"],
                "seal_j": g["seal_js"],
                "window_j": g["window_js"],
                "inj": g["inj"],
                "prod": g["prod"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
