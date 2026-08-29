"""Build a finer undulating-channel IMEX ruler (clone + patch mxspr006_channel).

21x21x8 VARI grid, noisy matrix, irregular channel — validation only.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "cmg_channel_3d" / "mxspr006_channel.dat"
DAT = HERE / "mxspr006_channel_fine.dat"
TRUTH = HERE / "truth_channel_fine.json"

NX = NY = 21
NZ = 8


def build_fields(seed: int = 7):
    rng = np.random.default_rng(seed)
    # keep similar areal extent as 7*315 ft
    di = dj = 105.0
    dk = np.array([28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0], dtype=float)
    dtop = np.zeros((NY, NX), dtype=float)
    k_x = np.zeros((NZ, NY, NX), dtype=float)
    channel = np.zeros((NZ, NY, NX), dtype=bool)

    # lognormal matrix ~ geo-mean 50 md
    noise = rng.normal(0.0, 0.40, size=(NZ, NY, NX))
    k_mat = 50.0 * np.exp(noise)
    k_x[:] = k_mat

    for j in range(NY):
        for i in range(NX):
            ii, jj = i + 1, j + 1
            ridge = float(np.exp(-0.5 * ((i - j) / 2.4) ** 2))
            t = (i + j) / float(NX + NY - 2)
            mountain = float(np.exp(-(((t - 0.5) / 0.32) ** 2)))
            wave = 18.0 * np.sin(2 * np.pi * i / 14.0) * np.cos(np.pi * j / 16.0)
            dtop[j, i] = 2080.0 - 90.0 * ridge * (0.35 + 0.65 * mountain) + wave

            # irregular width: 2–3 cells + pinch/swell
            half = 1.15 + 0.55 * np.sin(2 * np.pi * t * 1.7 + 0.4)
            off = abs(i - j) + 0.35 * np.sin(2 * np.pi * t * 2.3)
            if off <= half:
                elev = ridge * (0.35 + 0.65 * mountain)
                if elev > 0.55:
                    ks = range(3, 8)  # 1-based later
                elif elev > 0.28:
                    ks = range(2, 7)
                else:
                    ks = range(1, 6)
                for k1 in ks:
                    kk = k1 - 1
                    if 0 <= kk < NZ:
                        channel[kk, j, i] = True
                        jitter = 1.0 + 0.18 * float(rng.normal())
                        k_x[kk, j, i] = float(np.clip(2000.0 * jitter, 1200.0, 2800.0))

    k_y = k_x.copy()
    k_z = np.where(channel, 0.10 * k_x, 0.40 * k_x)
    return di, dj, dk, dtop, k_x, k_y, k_z, channel


def _fmt_block(vals: np.ndarray, per_line: int = 8) -> str:
    flat = np.asarray(vals, dtype=float).ravel(order="C")
    # IMEX *ALL typically I fastest, then J, then K. Our arrays are (k,j,i)
    # so we iterate k, j, i which is C-order of (nz,ny,nx).
    lines = []
    row: list[str] = []
    for v in flat:
        row.append(f"{v:10.4g}")
        if len(row) >= per_line:
            lines.append("     " + "".join(row))
            row = []
    if row:
        lines.append("     " + "".join(row))
    return "\n".join(lines)


def _dtop_block(dtop: np.ndarray) -> str:
    lines = []
    for j in range(dtop.shape[0]):
        vals = "  ".join(f"{dtop[j, i]:7.1f}" for i in range(dtop.shape[1]))
        lines.append("  " + vals)
    return "\n".join(lines)


def apply_to_dat(
    text: str,
    *,
    di: float,
    dj: float,
    dk: np.ndarray,
    dtop: np.ndarray,
    k_x: np.ndarray,
    k_y: np.ndarray,
    k_z: np.ndarray,
    inj_i: int,
    inj_j: int,
    prod_i: int,
    prod_j: int,
    inj_ks: list[int],
    prod_ks: list[int],
) -> str:
    grid_block = f"""   *GRID *VARI {NX} {NY} {NZ}
   *KDIR *DOWN
   *DI *CON
   {di}
   *DJ *CON
   {dj}
   *DK *KVAR
     {"  ".join(f"{v:.1f}" for v in dk)}

   ** Fine undulating ridge + noisy matrix (validation ruler)
   *DTOP
{_dtop_block(dtop)}

   *POR *CON
    0.300

   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{_fmt_block(k_x)}
   *PERMJ *ALL
{_fmt_block(k_y)}
   *PERMK *ALL
{_fmt_block(k_z)}
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
    banner = text.rfind("********************************************************************************", 0, end)
    if banner > start:
        end = banner
    new_text = text[:start] + grid_block + "\n\n   " + text[end:]
    new_text = new_text.replace("'Undulating Channel Twin'", "'Fine Undulating Channel Twin'")
    new_text = new_text.replace("'7x7x5 DTOP mountain ridge channel'", "'21x21x8 fine undulating channel ruler'")

    def make_perf(wellnum: int, i: int, j: int, ks: list[int]) -> str:
        lines = [f"   *PERF *GEO {wellnum}", "   ** if      jf     kf     ff"]
        for k in ks:
            lines.append(f"       {i}      {j}      {k}     1.0")
        return "\n".join(lines)

    new_text = re.sub(
        r"   \*PERF \*GEO 1\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(1, inj_i, inj_j, inj_ks) + "\n\n",
        new_text,
        count=1,
    )
    new_text = re.sub(
        r"   \*PERF \*GEO\s*2\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        make_perf(2, prod_i, prod_j, prod_ks) + "\n\n",
        new_text,
        count=1,
    )
    return new_text


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    di, dj, dk, dtop, kx, ky, kz, channel = build_fields()
    text = SRC.read_text(encoding="latin-1")
    inj_i = inj_j = 1
    prod_i = prod_j = NX
    inj_ks = [k + 1 for k in range(NZ) if channel[k, 0, 0]] or [3, 4, 5]
    prod_ks = [k + 1 for k in range(NZ) if channel[k, NY - 1, NX - 1]] or [4, 5, 6]
    dat = apply_to_dat(
        text,
        di=di,
        dj=dj,
        dk=dk,
        dtop=dtop,
        k_x=kx,
        k_y=ky,
        k_z=kz,
        inj_i=inj_i,
        inj_j=inj_j,
        prod_i=prod_i,
        prod_j=prod_j,
        inj_ks=inj_ks,
        prod_ks=prod_ks,
    )
    DAT.write_text(dat, encoding="latin-1")

    ch_n = int(np.sum(channel))
    k_mat = kx[~channel]
    k_ch = kx[channel]
    blocks = []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                if channel[k, j, i]:
                    blocks.append([i + 1, j + 1, k + 1])
    truth = {
        "description": "Fine undulating channel IMEX ruler (noisy matrix, irregular width)",
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "di_ft": di,
            "dj_ft": dj,
            "dk_ft": dk.tolist(),
            "kdir": "DOWN",
            "k1": "bottom",
            "grid_type": "VARI",
        },
        "structure": {
            "type": "DTOP_undulating_mountain_ridge",
            "dtop_min_ft": float(dtop.min()),
            "dtop_max_ft": float(dtop.max()),
            "relief_ft": float(dtop.max() - dtop.min()),
        },
        "units": "FIELD (md, ft, psi)",
        "wells": {
            "INJ": {"name": "SEAWATER INJECTOR", "i": inj_i, "j": inj_j, "k_perfs": inj_ks},
            "PROD": {"name": "PRODUCER", "i": prod_i, "j": prod_j, "k_perfs": prod_ks},
        },
        "background_perm_md": {
            "kx": float(np.exp(np.mean(np.log(k_mat)))),
            "ky": float(np.exp(np.mean(np.log(k_mat)))),
            "kz": float(np.exp(np.mean(np.log(0.40 * k_mat)))),
            "cv": float(np.std(k_mat) / np.mean(k_mat)),
        },
        "channel_perm_md": {
            "kx": float(np.mean(k_ch)),
            "ky": float(np.mean(k_ch)),
            "kz": float(np.mean(0.10 * k_ch)),
        },
        "channel_blocks_ijk": blocks,
        "n_channel_blocks": ch_n,
        "n_cells": int(NX * NY * NZ),
        "note": "Validation ruler only. Product inversion never reads this k field.",
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dat": str(DAT),
                "n_cells": NX * NY * NZ,
                "channel_blocks": ch_n,
                "matrix_cv": truth["background_perm_md"]["cv"],
                "k_mat_geo": truth["background_perm_md"]["kx"],
                "k_ch_mean": truth["channel_perm_md"]["kx"],
                "relief_ft": truth["structure"]["relief_ft"],
                "inj_ks": inj_ks,
                "prod_ks": prod_ks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
