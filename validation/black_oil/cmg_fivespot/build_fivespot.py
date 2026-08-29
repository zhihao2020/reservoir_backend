"""Five-spot IMEX ruler: 4 corner injectors + center producer, strip + noise."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "cmg_channel_3d" / "mxspr006_channel.dat"
DAT = HERE / "mxspr006_fivespot.dat"
TRUTH = HERE / "truth_fivespot.json"

NX = NY = 15
NZ = 6
STB_TO_M3S = 0.158987 / 86400.0


def build_fields(seed: int = 11):
    rng = np.random.default_rng(seed)
    di = dj = 147.0
    dk = np.array([32.0, 28.0, 24.0, 22.0, 20.0, 18.0], dtype=float)
    noise = rng.normal(0.0, 0.35, size=(NZ, NY, NX))
    kx = 60.0 * np.exp(noise)
    strip = np.zeros((NZ, NY, NX), dtype=bool)
    for j in range(NY):
        for i in range(NX):
            # gentle high-k belt through the producer row (j~8)
            wiggle = 1.2 * np.sin(2 * np.pi * i / 11.0)
            if abs((j + 1) - (8 + wiggle)) <= 1.6:
                for k in (1, 2, 3, 4):
                    strip[k, j, i] = True
                    kx[k, j, i] = float(np.clip(900.0 * (1.0 + 0.15 * rng.normal()), 500.0, 1400.0))
    ky = kx.copy()
    kz = np.where(strip, 0.12 * kx, 0.40 * kx)
    return di, dj, dk, kx, ky, kz, strip


def _fmt_block(vals: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(vals, dtype=float).ravel(order="C")
    lines, row = [], []
    for v in flat:
        row.append(f"{v:10.4g}")
        if len(row) >= per_line:
            lines.append("     " + "".join(row))
            row = []
    if row:
        lines.append("     " + "".join(row))
    return "\n".join(lines)


def main() -> int:
    di, dj, dk, kx, ky, kz, strip = build_fields()
    text = SRC.read_text(encoding="latin-1")
    grid_block = f"""   *GRID *CART {NX} {NY} {NZ}
   *KDIR *DOWN
   *DI *CON
   {di}
   *DJ *CON
   {dj}
   *DK *KVAR
     {"  ".join(f"{v:.1f}" for v in dk)}

   *POR *CON
    0.300
   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{_fmt_block(kx)}
   *PERMJ *ALL
{_fmt_block(ky)}
   *PERMK *ALL
{_fmt_block(kz)}
"""
    start = text.find("*GRID *CART")
    if start < 0:
        start = text.find("*GRID *VARI")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("*MODEL *BLACKOIL")
    banner = text.rfind("********************************************************************************", 0, end)
    if banner > start:
        end = banner
    text = text[:start] + grid_block + "\n\n   " + text[end:]
    text = text.replace("'Undulating Channel Twin'", "'Five-spot strip twin'")
    text = text.replace("'7x7x5 DTOP mountain ridge channel'", "'15x15x6 five-spot high-k strip'")

    ks = [2, 3, 4, 5]
    corners = [(1, 1), (NX, 1), (1, NY), (NX, NY)]
    ci = cj = (NX + 1) // 2
    well_block = [
        "   *DTWELL      1.0",
        "   *WELL 1    'INJ1'",
        "   *WELL 2    'PROD'",
        "   *WELL 3    'INJ2'",
        "   *WELL 4    'INJ3'",
        "   *WELL 5    'INJ4'",
        "",
    ]
    # well 1 = first corner inj
    well_block += [
        "   *INJECTOR  1",
        "   *INCOMP    *WATER   0.0 1.0",
        "   *OPERATE   *MAX       *STW    1250.0",
        "   *OPERATE   *MAX       *BHP    9000.0 *CONT *REPEAT",
        "   *GEOMETRY  *K  0.25   0.34    1.0     0.0",
        "   *PERF *GEO 1",
        "   ** if      jf     kf     ff",
    ]
    i, j = corners[0]
    for k in ks:
        well_block.append(f"       {i}      {j}      {k}     1.0")
    well_block += [
        "",
        "   *PRODUCER  2",
        "   *OPERATE   *MAX       *STO    2500",
        "   *OPERATE   *MIN       *BHP    1500.0",
        "   *PERF *GEO 2",
        "   ** if      jf     kf     ff",
    ]
    for k in ks:
        well_block.append(f"       {ci}      {cj}      {k}     1.0")
    for n, (i, j) in enumerate(corners[1:], start=3):
        well_block += [
            "",
            f"   *INJECTOR  {n}",
            "   *INCOMP    *WATER   0.0 1.0",
            "   *OPERATE   *MAX       *STW    1250.0",
            "   *OPERATE   *MAX       *BHP    9000.0 *CONT *REPEAT",
            "   *GEOMETRY  *K  0.25   0.34    1.0     0.0",
            f"   *PERF *GEO {n}",
            "   ** if      jf     kf     ff",
        ]
        for k in ks:
            well_block.append(f"       {i}      {j}      {k}     1.0")
    well_block += ["", "   *SCLTBL-WELL 2", "    1  1", "", ""]
    well_txt = "\n".join(well_block) + "\n"
    text2, nsub = re.subn(
        r"   \*DTWELL.*?\n   \*SCLTBL-WELL 2\n    1  1\n+",
        well_txt,
        text,
        count=1,
        flags=re.S,
    )
    if nsub != 1:
        raise RuntimeError("failed to replace well block")
    DAT.write_text(text2, encoding="latin-1")

    blocks = []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                if strip[k, j, i]:
                    blocks.append([i + 1, j + 1, k + 1])
    q_inj = 1250.0 * STB_TO_M3S
    q_prod = -2500.0 * STB_TO_M3S
    wells = {
        "INJ1": {"name": "INJ1", "i": 1, "j": 1, "k_perfs": ks, "role": "injector", "rate_m3s": q_inj},
        "INJ2": {"name": "INJ2", "i": NX, "j": 1, "k_perfs": ks, "role": "injector", "rate_m3s": q_inj},
        "INJ3": {"name": "INJ3", "i": 1, "j": NY, "k_perfs": ks, "role": "injector", "rate_m3s": q_inj},
        "INJ4": {"name": "INJ4", "i": NX, "j": NY, "k_perfs": ks, "role": "injector", "rate_m3s": q_inj},
        "PROD": {"name": "PROD", "i": ci, "j": cj, "k_perfs": ks, "role": "producer", "rate_m3s": q_prod},
    }
    k_mat = kx[~strip]
    k_ch = kx[strip]
    truth = {
        "description": "Five-spot waterflood, high-k strip + noisy matrix",
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "di_ft": di,
            "dj_ft": dj,
            "dk_ft": dk.tolist(),
            "kdir": "DOWN",
            "k1": "bottom",
            "grid_type": "CART",
        },
        "units": "FIELD (md, ft, psi)",
        "wells": wells,
        "background_perm_md": {
            "kx": float(np.exp(np.mean(np.log(np.clip(k_mat, 1e-9, None))))),
            "cv": float(np.std(k_mat) / max(np.mean(k_mat), 1e-9)),
        },
        "channel_perm_md": {"kx": float(np.mean(k_ch))},
        "channel_blocks_ijk": blocks,
        "n_channel_blocks": int(np.sum(strip)),
        "n_cells": NX * NY * NZ,
        "note": "Validation ruler only. Different well pattern from 1-inj 1-prod channel.",
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(json.dumps({"dat": str(DAT), "n_cells": NX * NY * NZ, "strip": int(np.sum(strip))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
