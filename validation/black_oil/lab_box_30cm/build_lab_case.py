"""Clone mxspr006 into a 30 cm lab-box IMEX case with draped high-k (flat DTOP).

Default grid is 30^3 (inversion-check). Pass --n 50 for CMG truth.
Does not modify the field-scale channel/fault cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.core.units import MD_TO_M2  # noqa: E402
from reservoir_backend.pipeline.lab_horizon import LabBoxSpec, sample_lab_box, well_xyz  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = ROOT / "black_oil" / "validation" / "cmg_channel_3d" / "mxspr006_channel.dat"
DAT = HERE / "lab_box_30cm.dat"
TRUTH = HERE / "truth_lab_box.json"
M_TO_FT = 1.0 / 0.3048


def _m2_to_md(k: float) -> float:
    return float(k) / MD_TO_M2


def _fmt_all(arr: np.ndarray) -> str:
    """IMEX *ALL: I fastest, then J, then K (k=1 bottom)."""
    nz, ny, nx = arr.shape
    vals: list[str] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vals.append(f"{arr[k, j, i]:.6g}")
    lines = []
    for n in range(0, len(vals), 10):
        lines.append("   " + "  ".join(vals[n : n + 10]))
    return "\n".join(lines)


def _transi_mod(fault: np.ndarray, window_mult: float = 0.05) -> str:
    """Seal +I faces of fault cells (1-based IJK)."""
    nz, ny, nx = fault.shape
    lines: list[str] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                if fault[k, j, i]:
                    lines.append(f"      {i + 1}:{i + 1} {j + 1}:{j + 1} {k + 1}:{k + 1} = 0.0")
                elif fault[k, j, i + 1]:
                    lines.append(
                        f"      {i + 1}:{i + 1} {j + 1}:{j + 1} {k + 1}:{k + 1} = {window_mult}"
                    )
    return "\n".join(lines)


def _well_ks(high: np.ndarray, i0: int, j0: int) -> list[int]:
    ks = [k + 1 for k in range(high.shape[0]) if high[k, j0, i0]]
    if ks:
        return ks
    mid = high.shape[0] // 2 + 1
    return [max(1, mid - 1), mid, min(high.shape[0], mid + 1)]


def _cell_ij(x: float, y: float, n: int, spec: LabBoxSpec) -> tuple[int, int]:
    i = int(np.clip(np.floor(x / spec.lx * n), 0, n - 1))
    j = int(np.clip(np.floor(y / spec.ly * n), 0, n - 1))
    return i, j


def apply_to_dat(
    text: str,
    *,
    n: int,
    painted: dict[str, object],
    include_fault: bool,
    inj_ijk: tuple[int, int, list[int]],
    prod_ijk: tuple[int, int, list[int]],
) -> tuple[str, list[int], list[int], float, float, float]:
    spec = painted["spec"]
    assert isinstance(spec, LabBoxSpec)
    dx_ft = float(painted["dx"]) * M_TO_FT
    k_bg = _m2_to_md(float(painted["k_background"]))
    k_hi = _m2_to_md(float(painted["k_high"]))
    k_arr = np.asarray(painted["k"], dtype=float) / MD_TO_M2
    kz = np.clip(k_arr * 0.1, 0.001, None)
    fault = np.asarray(painted["fault_mask"])
    inj_i, inj_j, inj_ks = inj_ijk
    prod_i, prod_j, prod_ks = prod_ijk
    transi = _transi_mod(fault) if include_fault else ""
    transi_block = ""
    if transi:
        transi_block = f"\n   *TRANSI *CON\n      1.0\n   *MOD\n{transi}\n"

    grid_block = f"""   *GRID *CART {n} {n} {n}
   *KDIR *DOWN
   *DI *CON
   {dx_ft:.8f}
   *DJ *CON
   {dx_ft:.8f}
   *DK *CON
   {dx_ft:.8f}

   ** Flat lid (lab box cover). Mountain is PERM drape only; do not use DTOP relief.
   *DTOP *CON
    10.0

   *POR *CON
    {float(np.mean(np.asarray(painted["phi"]))):.3f}

   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{_fmt_all(k_arr)}
   *PERMJ *ALL
{_fmt_all(k_arr)}
   *PERMK *ALL
{_fmt_all(kz)}
{transi_block}"""

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
    new_text = new_text.replace("'Undulating Channel Twin'", "'Lab 30cm mountain-drape box'")
    new_text = new_text.replace(
        "'7x7x5 DTOP mountain ridge channel'",
        f"'{n}x{n}x{n} flat-DTOP draped high-k (lab 0.30 m)'",
    )

    rad = min(0.25, 0.30 * dx_ft)
    new_text = re.sub(
        r"\*GEOMETRY\s+\*K\s+[0-9.]+",
        f"*GEOMETRY  *K  {rad:.5f}",
        new_text,
        count=1,
    )
    new_text = re.sub(
        r"\*OPERATE\s+\*MAX\s+\*STW\s+[0-9.]+",
        "*OPERATE   *MAX       *STW    0.05",
        new_text,
        count=1,
    )
    new_text = re.sub(
        r"\*OPERATE\s+\*MAX\s+\*STO\s+[0-9.]+",
        "*OPERATE   *MAX       *STO    0.05",
        new_text,
        count=1,
    )
    new_text = re.sub(r"\*DTWELL\s+[0-9.]+", "*DTWELL      0.05", new_text, count=1)

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
    new_text = re.sub(
        r"   \*TIME\s+1\.0\s*\n\s*\*DATE[^\n]*\n(?:\s*\*DATE[^\n]*\n)+",
        "   *TIME      0.1\n   *TIME      0.5\n   *TIME      1.0\n   *TIME      2.0\n   *TIME      5.0\n",
        new_text,
        count=1,
    )
    return new_text, inj_ks, prod_ks, k_bg, k_hi, dx_ft


def build(n: int, *, include_fault: bool) -> dict[str, object]:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    spec = LabBoxSpec()
    painted = sample_lab_box(n, n, n, spec, include_fault=include_fault)
    inj, prod = well_xyz(spec)
    high = np.asarray(painted["highk_mask"])
    ii, jj = _cell_ij(inj[0], inj[1], n, spec)
    pi, pj = _cell_ij(prod[0], prod[1], n, spec)
    inj_ks = _well_ks(high, ii, jj)
    prod_ks = _well_ks(high, pi, pj)
    text = TEMPLATE.read_text(encoding="latin-1")
    new_text, inj_ks, prod_ks, k_bg, k_hi, dx_ft = apply_to_dat(
        text,
        n=n,
        painted=painted,
        include_fault=include_fault,
        inj_ijk=(ii + 1, jj + 1, inj_ks),
        prod_ijk=(pi + 1, pj + 1, prod_ks),
    )
    DAT.write_text(new_text, encoding="latin-1")
    zh = np.asarray(painted["z_horizon"], dtype=float)
    blocks = [[int(i + 1), int(j + 1), int(k + 1)] for k, j, i in zip(*np.where(high), strict=True)]
    truth = {
        "description": "Lab 30 cm box, mold removed, draped high-k on z_horizon; flat DTOP",
        "source_template": str(TEMPLATE.relative_to(ROOT)).replace("\\", "/"),
        "grid": {
            "nx": n,
            "ny": n,
            "nz": n,
            "di_ft": dx_ft,
            "dj_ft": dx_ft,
            "dk_ft": [dx_ft] * n,
            "kdir": "DOWN",
            "k1": "bottom",
            "grid_type": "CART",
            "box_m": [spec.lx, spec.ly, spec.lz],
        },
        "structure": {
            "type": "draped_highk_on_z_horizon",
            "dtop": "CON 10 ft (flat lid)",
            "relief_m": float(zh.max() - zh.min()),
            "z_horizon_min_m": float(zh.min()),
            "z_horizon_max_m": float(zh.max()),
        },
        "units": "FIELD (md, ft, psi) — geometry converted from 0.30 m SI box",
        "wells": {
            "INJ": {
                "x_m": inj[0],
                "y_m": inj[1],
                "z_m": inj[2],
                "i": ii + 1,
                "j": jj + 1,
                "k_perfs": inj_ks,
            },
            "PROD": {
                "x_m": prod[0],
                "y_m": prod[1],
                "z_m": prod[2],
                "i": pi + 1,
                "j": pj + 1,
                "k_perfs": prod_ks,
            },
        },
        "background_perm_md": {"kx": k_bg, "ky": k_bg, "kz": k_bg * 0.1},
        "channel_perm_md": {"kx": k_hi, "ky": k_hi, "kz": k_hi * 0.1},
        "include_fault": include_fault,
        "channel_blocks_ijk": blocks,
        "note": "Same z_horizon as reservoir_backend.pipeline.lab_horizon; not a PVT-scaled lab fluid.",
    }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return {
        "dat": str(DAT),
        "n": n,
        "highk_blocks": len(blocks),
        "relief_m": truth["structure"]["relief_m"],
        "include_fault": include_fault,
        "inj_ks": inj_ks,
        "prod_ks": prod_ks,
        "di_ft": dx_ft,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30, help="cells per axis (30 inversion, 50 CMG truth)")
    p.add_argument("--no-fault", action="store_true", help="omit TRANSI baffle")
    args = p.parse_args(argv)
    if args.n < 8:
        raise SystemExit("--n must be >= 8")
    info = build(args.n, include_fault=not args.no_fault)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
