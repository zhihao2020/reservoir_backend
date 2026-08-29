"""Regenerate undulating DTOP mountain-ridge CMG case + truth mask.

Run from repo root or this directory:
  python validation/black_oil/cmg_channel_3d/build_undulating_case.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_channel.dat"
TRUTH = HERE / "truth_channel.json"


def build_structure(nx: int = 7, ny: int = 7, nz: int = 5):
    di = dj = 315.0
    dk = np.array([40.0, 35.0, 30.0, 25.0, 20.0])  # k=1 bottom ... k=nz top
    dtop = np.zeros((ny, nx))
    channel: list[list[int]] = []
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            ridge = float(np.exp(-0.5 * ((i - j) / 1.15) ** 2))
            t = ((i - 1) + (j - 1)) / 12.0
            mountain = float(np.exp(-(((t - 0.5) / 0.30) ** 2)))
            wave = 22.0 * np.sin(2 * np.pi * (i - 1) / 6.5) * np.cos(np.pi * (j - 1) / 6.5)
            dtop[j - 1, i - 1] = 2080.0 - 95.0 * ridge * (0.35 + 0.65 * mountain) + wave
            if abs(i - j) <= 1:
                elev = ridge * (0.35 + 0.65 * mountain)
                if elev > 0.55:
                    ks = [3, 4, 5]
                elif elev > 0.25:
                    ks = [2, 3, 4]
                else:
                    ks = [1, 2, 3]
                for k in ks:
                    channel.append([i, j, k])
    # unique
    seen: set[tuple[int, int, int]] = set()
    uniq = []
    for b in channel:
        key = (b[0], b[1], b[2])
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return di, dj, dk, dtop, uniq


def mod_lines(blocks: list[list[int]], value: float) -> str:
    return "\n".join(f"      {i}:{i} {j}:{j} {k}:{k} = {value}" for i, j, k in blocks)


def apply_to_dat(text: str, nx: int, ny: int, nz: int, di: float, dj: float, dk: np.ndarray, dtop: np.ndarray, channel: list[list[int]]) -> str:
    dtop_lines = []
    for j in range(ny):
        vals = "  ".join(f"{dtop[j, i]:7.1f}" for i in range(nx))
        dtop_lines.append("  " + vals)
    dtop_block = "\n".join(dtop_lines)
    mod_h = mod_lines(channel, 2000.0)
    mod_v = mod_lines(channel, 200.0)
    grid_block = f"""   *GRID *VARI {nx} {ny} {nz}
   *KDIR *DOWN
   *DI *CON
   {di}
   *DJ *CON
   {dj}
   *DK *KVAR
     {"  ".join(f"{v:.1f}" for v in dk)}

   ** Undulating structure: DTOP mountain ridge along injector-producer diagonal
   *DTOP
{dtop_block}

   *POR *CON
    0.300

   *PRPOR    14.7
   *CPOR     3.0E-6

   ** Background low-k + undulating high-k mountain/channel body
   *PERMI *CON
      50.0
   *MOD
{mod_h}
   *PERMJ *CON
      50.0
   *MOD
{mod_h}
   *PERMK *CON
      20.0
   *MOD
{mod_v}
"""
    start = text.find("*GRID *CART")
    if start < 0:
        start = text.find("*GRID *VARI")
    if start < 0:
        raise ValueError("GRID keyword not found")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("*MODEL *BLACKOIL")
    if end < 0:
        raise ValueError("*MODEL not found")
    # include preceding star banner if present
    banner = text.rfind("********************************************************************************", 0, end)
    if banner > start:
        end = banner

    new_text = text[:start] + grid_block + "\n\n   " + text[end:]
    new_text = new_text.replace("'Channel Twin Sensor Inverse'", "'Undulating Channel Twin'")
    new_text = new_text.replace("'7x7x3 high-k diagonal channel'", "'7x7x5 DTOP mountain ridge channel'")
    new_text = new_text.replace("'Channel Twin for Sensor Inverse Validation'", "'Undulating Channel Twin'")
    new_text = new_text.replace(
        "'7x7x3 seawater inject with diagonal high-k channel (patched)'",
        "'7x7x5 DTOP mountain ridge channel'",
    )

    inj_ks = sorted({k for i, j, k in channel if i == 1 and j == 1}) or [2, 3]
    prod_ks = sorted({k for i, j, k in channel if i == 7 and j == 7}) or [3, 4]

    def make_perf(wellnum: int, i: int, j: int, ks: list[int]) -> str:
        lines = [f"   *PERF *GEO {wellnum}", "   ** if      jf     kf     ff"]
        for k in ks:
            lines.append(f"       {i}      {j}      {k}     1.0")
        return "\n".join(lines)

    new_text = re.sub(
        r"   \*PERF \*GEO 1\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(1, 1, 1, inj_ks) + "\n\n",
        new_text,
        count=1,
    )
    new_text = re.sub(
        r"   \*PERF \*GEO\s*2\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(2, 7, 7, prod_ks) + "\n\n",
        new_text,
        count=1,
    )
    return new_text, inj_ks, prod_ks


def main() -> int:
    nx = ny = 7
    nz = 5
    di, dj, dk, dtop, channel = build_structure(nx, ny, nz)
    text = DAT.read_text(encoding="latin-1")
    new_text, inj_ks, prod_ks = apply_to_dat(text, nx, ny, nz, di, dj, dk, dtop, channel)
    DAT.write_text(new_text, encoding="latin-1")
    truth = {
        "description": "Undulating DTOP mountain ridge + high-k channel for mxspr006_channel.dat",
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "di_ft": di,
            "dj_ft": dj,
            "dk_ft": dk.tolist(),
            "kdir": "DOWN",
            "k1": "bottom",
            "grid_type": "VARI",
        },
        "structure": {
            "type": "DTOP_undulating_mountain_ridge",
            "dtop_ft": dtop.tolist(),
            "dtop_min_ft": float(dtop.min()),
            "dtop_max_ft": float(dtop.max()),
            "relief_ft": float(dtop.max() - dtop.min()),
            "note": "Smaller DTOP = structural crest. Ridge along I~J with sinusoidal waves. Requires *GRID *VARI.",
        },
        "units": "FIELD (md, ft, psi)",
        "wells": {
            "INJ": {"name": "SEAWATER INJECTOR", "i": 1, "j": 1, "k_perfs": inj_ks},
            "PROD": {"name": "PRODUCER", "i": 7, "j": 7, "k_perfs": prod_ks},
        },
        "background_perm_md": {"kx": 50.0, "ky": 50.0, "kz": 20.0},
        "channel_perm_md": {"kx": 2000.0, "ky": 2000.0, "kz": 200.0},
        "channel_blocks_ijk": channel,
        "note": "High-k body follows diagonal mountain ridge; vertical k-stack rises toward structural crest.",
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dat": str(DAT),
                "channel_blocks": len(channel),
                "relief_ft": truth["structure"]["relief_ft"],
                "dtop_min": truth["structure"]["dtop_min_ft"],
                "dtop_max": truth["structure"]["dtop_max_ft"],
                "inj_ks": inj_ks,
                "prod_ks": prod_ks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
