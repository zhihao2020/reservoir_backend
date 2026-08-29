"""Clone mxspr006_channel.dat into a two-layer lab-analog IMEX case.

Does not modify the original channel/fault samples. Geometry is a small
Cartesian box in FIELD units so IMEX PVT stays well-posed. Truth is two
horizontal layers (identifiable by RegionParameterization).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEMPLATE = ROOT / "black_oil" / "validation" / "cmg_channel_3d" / "mxspr006_channel.dat"
DAT = HERE / "lab_layers.dat"
TRUTH = HERE / "truth_lab_layers.json"

# Analog box: 12 x 8 x 6, 24 x 16 x 9 ft  (~7.3 x 4.9 x 2.7 m)
NX, NY, NZ = 12, 8, 6
DI_FT, DJ_FT, DK_FT = 2.0, 2.0, 1.5
K_LO_MD, K_HI_MD = 50.0, 500.0
PHI = 0.30
# KDIR DOWN: K=1 is top. Top three layers = high k.
N_TOP_HIGH = 3
# Voidage-style *pressure* pair so a Δp remains after t=0.
# Rates are observations. This is experimental design, not "make F = CMG".
STW_STBDAY = 1.0e6
INJ_BHP_PSI = 3200.0
PROD_BHP_PSI = 2800.0
PRES_PSI = 3000.0
SWI = 0.20
FT_TO_M = 0.3048
MD_TO_M2 = 9.869233e-16


def _replace_grid(text: str) -> str:
    start = text.find("*GRID *VARI")
    if start < 0:
        start = text.find("*GRID *CART")
    if start < 0:
        raise ValueError("GRID keyword not found")
    start = text.rfind("\n", 0, start) + 1
    end = text.find("*MODEL *BLACKOIL")
    if end < 0:
        raise ValueError("*MODEL not found")
    banner = text.rfind("********************************************************************************", 0, end)
    if banner > start:
        end = banner

    perm = np.full((NZ, NY, NX), K_LO_MD, dtype=float)
    perm[:N_TOP_HIGH, :, :] = K_HI_MD  # Plane K=1 is top
    kz = perm * 0.1

    def fmt(arr: np.ndarray) -> str:
        vals = [f"{arr[k, j, i]:.6g}" for k in range(NZ) for j in range(NY) for i in range(NX)]
        lines = ["   " + "  ".join(vals[n : n + 10]) for n in range(0, len(vals), 10)]
        return "\n".join(lines)

    block = f"""   *GRID *CART {NX} {NY} {NZ}
   *KDIR *DOWN
   *DI *CON
   {DI_FT:.6f}
   *DJ *CON
   {DJ_FT:.6f}
   *DK *CON
   {DK_FT:.6f}

   *POR *CON
    {PHI:.3f}

   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{fmt(perm)}
   *PERMJ *ALL
{fmt(perm)}
   *PERMK *ALL
{fmt(kz)}

"""
    return text[:start] + block + "\n   " + text[end:]


def _patch_io_and_wells(text: str) -> str:
    text = text.replace("'Undulating Channel Twin'", "'Lab two-layer invert ruler'")
    text = re.sub(r"\*WPRN\s+\*GRID\s+\d+", "*WPRN   *GRID 1", text, count=1)
    text = re.sub(r"\*DTMAX\s+[0-9.]+", "*DTMAX  0.25", text, count=1)
    text = re.sub(r"\*GEOMETRY\s+\*K\s+[0-9.]+", "*GEOMETRY  *K  0.20", text, count=1)
    text = re.sub(
        r"\*OPERATE\s+\*MAX\s+\*STW\s+[0-9.]+",
        f"*OPERATE   *MAX       *STW    {STW_STBDAY:.1f}",
        text,
        count=1,
    )
    text = re.sub(
        r"\*OPERATE\s+\*MAX\s+\*BHP\s+[0-9.]+[^\n]*",
        f"*OPERATE   *MAX       *BHP    {INJ_BHP_PSI:.1f}",
        text,
        count=1,
    )
    text = re.sub(
        r"\*OPERATE\s+\*MAX\s+\*STO\s+[0-9.]+",
        "*OPERATE   *MAX       *STO    1.0E6",
        text,
        count=1,
    )
    text = re.sub(
        r"\*OPERATE\s+\*MIN\s+\*BHP\s+[0-9.]+",
        f"*OPERATE   *MIN       *BHP    {PROD_BHP_PSI:.1f}",
        text,
        count=1,
    )
    text = re.sub(r"\*DTWELL\s+[0-9.]+", "*DTWELL      0.01", text, count=1)

    inj_j = NY // 2
    prod_j = NY // 2
    inj_k = list(range(N_TOP_HIGH, NZ + 1))  # complete into both layers
    prod_k = list(range(1, NZ + 1))

    def perf(well: int, i: int, j: int, ks: list[int]) -> str:
        lines = [f"   *PERF *GEO {well}", "   ** if      jf     kf     ff"]
        for k in ks:
            lines.append(f"       {i}      {j}      {k}     1.0")
        return "\n".join(lines)

    text = re.sub(
        r"   \*PERF \*GEO 1\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        perf(1, 1, inj_j, inj_k) + "\n\n",
        text,
        count=1,
    )
    text = re.sub(
        r"   \*PERF \*GEO\s*2\n(?:   \*\*[^\n]*\n)?(?:\s+\d+\s+\d+\s+\d+\s+[\d.]+\n?)+",
        perf(2, NX, prod_j, prod_k) + "\n\n",
        text,
        count=1,
    )
    text = re.sub(
        r"   \*TIME\s+1\.0\s*\n\s*\*DATE[^\n]*\n(?:\s*\*DATE[^\n]*\n)+",
        "   *TIME      0.25\n   *TIME      0.50\n   *TIME      1.00\n"
        "   *TIME      2.00\n   *TIME      4.00\n   *TIME      8.00\n",
        text,
        count=1,
    )
    return text


def build() -> dict[str, object]:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    text = TEMPLATE.read_text(encoding="latin-1")
    text = _replace_grid(text)
    text = _patch_io_and_wells(text)
    DAT.write_text(text, encoding="latin-1")

    region = np.zeros((NZ, NY, NX), dtype=np.int64)
    region[:N_TOP_HIGH, :, :] = 1  # 1 = high-k top (CMG K=1)
    truth = {
        "description": "Two-layer lab-analog IMEX ruler for DigitalTwin ES-MDA",
        "source_template": str(TEMPLATE.relative_to(ROOT)).replace("\\", "/"),
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "di_ft": DI_FT,
            "dj_ft": DJ_FT,
            "dk_ft": DK_FT,
            "kdir": "DOWN",
            "k1": "top",
            "size_m": [NX * DI_FT * FT_TO_M, NY * DJ_FT * FT_TO_M, NZ * DK_FT * FT_TO_M],
        },
        "layers": {
            "top_high_k_md": K_HI_MD,
            "bottom_low_k_md": K_LO_MD,
            "n_top_high": N_TOP_HIGH,
            "k_hi_m2": K_HI_MD * MD_TO_M2,
            "k_lo_m2": K_LO_MD * MD_TO_M2,
        },
        "controls": {
            "inj_bhp_psi": INJ_BHP_PSI,
            "prod_bhp_psi": PROD_BHP_PSI,
            "pres_psi": PRES_PSI,
            "swi": SWI,
            "phi": PHI,
        },
        "wells": {
            "INJ": {"i": 1, "j": NY // 2, "k_cmg": list(range(N_TOP_HIGH, NZ + 1)), "control": "pressure"},
            "PROD": {"i": NX, "j": NY // 2, "k_cmg": list(range(1, NZ + 1)), "control": "pressure"},
        },
        "times_day": [0.25, 0.50, 1.00, 2.00, 4.00, 8.00],
        "history_end_day": 4.00,
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    np.save(HERE / "region_id_cmg_k1_top.npy", region)
    return {"dat": str(DAT), "nx": NX, "ny": NY, "nz": NZ, "k_hi_md": K_HI_MD, "k_lo_md": K_LO_MD}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
