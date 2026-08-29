"""Build five IMEX shale-oil *analog* rulers (depletion + HW + frac strips).

Clone official mxspr006 black-oil PVT; remove water injector. Not GEM, not adsorption.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(r"D:\Tool\CMG\IMEX\2024.20\TPL\spr\mxspr006.dat")
IMEX_EXE = Path(r"D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe")

NX, NY, NZ = 21, 31, 5
J_WELL = 16  # 1-based
K_WELL = 3
DI = DJ = 50.0
DK = np.array([25.0, 20.0, 18.0, 16.0, 14.0])
K_MAT = 0.001
K_SRV = 0.4
K_FRAC = 8000.0


CASES = ("S1", "S2", "S3", "S4", "S5")
CASE_DIR = {
    "S1": "cmg_s1_hw5frac",
    "S2": "cmg_s2_hw9frac",
    "S3": "cmg_s3_twohw",
    "S4": "cmg_s4_parent_child",
    "S5": "cmg_s5_shutin",
}


def frac_i_list(case: str) -> list[int]:
    if case == "S2":
        return [3, 5, 7, 9, 11, 13, 15, 17, 19]
    return [4, 8, 11, 14, 18]


def wells_for(case: str) -> list[dict]:
    hw1 = {
        "name": "HW1",
        "j": J_WELL,
        "k": K_WELL,
        "i0": 3,
        "i1": 19,
        "role": "producer",
        "open_from_day": 0.0,
    }
    hw2 = {
        "name": "HW2",
        "j": 8,
        "k": K_WELL,
        "i0": 3,
        "i1": 19,
        "role": "producer",
        "open_from_day": 365.0 if case == "S4" else 0.0,
    }
    if case in ("S3", "S4"):
        return [hw1, hw2]
    return [hw1]


def build_k(case: str, seed: int = 3):
    rng = np.random.default_rng(seed)
    kx = K_MAT * np.exp(rng.normal(0.0, 0.15, size=(NZ, NY, NX)))
    frac = np.zeros((NZ, NY, NX), dtype=bool)
    srv = np.zeros((NZ, NY, NX), dtype=bool)
    half = 8  # frac half-length in J cells
    for i_f in frac_i_list(case):
        ii = i_f - 1
        for j in range(max(0, J_WELL - 1 - half), min(NY, J_WELL + half)):
            for k in (1, 2, 3):  # 0-based k=2,3,4
                frac[k, j, ii] = True
                kx[k, j, ii] = K_FRAC
        for di in (-1, 1):
            ia = ii + di
            if 0 <= ia < NX:
                for j in range(max(0, J_WELL - 1 - half), min(NY, J_WELL + half)):
                    for k in (1, 2, 3):
                        if not frac[k, j, ia]:
                            srv[k, j, ia] = True
                            kx[k, j, ia] = K_SRV
    if case in ("S3", "S4"):
        j2 = 8
        for i_f in frac_i_list("S1"):
            ii = i_f - 1
            for j in range(max(0, j2 - 1 - 6), min(NY, j2 + 6)):
                for k in (1, 2, 3):
                    frac[k, j, ii] = True
                    kx[k, j, ii] = K_FRAC
    ky = kx.copy()
    kz = np.where(frac, 0.2 * kx, 0.1 * kx)
    return kx, ky, kz, frac, srv


def _fmt(vals: np.ndarray) -> str:
    flat = np.asarray(vals, dtype=float).ravel(order="C")
    lines, row = [], []
    for v in flat:
        row.append(f"{v:12.5g}")
        if len(row) >= 6:
            lines.append("     " + "".join(row))
            row = []
    if row:
        lines.append("     " + "".join(row))
    return "\n".join(lines)


def _grid_block(kx, ky, kz) -> str:
    return f"""   *GRID *CART {NX} {NY} {NZ}
   *KDIR *DOWN
   *DI *CON
   {DI}
   *DJ *CON
   {DJ}
   *DK *KVAR
     {"  ".join(f"{v:.1f}" for v in DK)}
   *DEPTH 1 1 1 8500.

   *POR *CON
    0.080
   *PRPOR    14.7
   *CPOR     3.0E-6

   *PERMI *ALL
{_fmt(kx)}
   *PERMJ *ALL
{_fmt(ky)}
   *PERMK *ALL
{_fmt(kz)}
"""


def _well_block(case: str) -> str:
    wells = wells_for(case)
    lines = ["   *DTWELL      1.0"]
    for n, w in enumerate(wells, start=1):
        lines.append(f"   *WELL {n}    '{w['name']}'")
    lines.append("")
    for n, w in enumerate(wells, start=1):
        lines += [
            f"   *PRODUCER  {n}",
            "   *OPERATE   *MAX       *STO    800",
            "   *OPERATE   *MIN       *BHP    1500.0",
            "   *GEOMETRY  *J  0.25   0.34    1.0     0.0",
            f"   *PERF *GEO {n}",
            "   ** if      jf     kf     ff",
        ]
        for i in range(w["i0"], w["i1"] + 1):
            lines.append(f"       {i}      {w['j']}      {w['k']}     1.0")
        if w["open_from_day"] > 0:
            lines.append(f"   *SHUTIN  {n}")
        lines.append("")
    lines += ["   *SCLTBL-WELL 1", "    1  1", ""]
    # dates: ~2 years
    lines += [
        "   *TIME      1.0",
        "   *DATE 1988 01 01",
        "   *DATE 1988 04 01",
        "   *DATE 1988 07 01",
    ]
    if case == "S5":
        lines += [
            "   *SHUTIN  1",
            "   *DATE 1988 10 01",
            "   *OPEN    1",
            "   *DATE 1989 01 01",
        ]
    elif case == "S4":
        lines += [
            "   *DATE 1988 11 01",
            "   *OPEN    2",
            "   *DATE 1989 01 01",
        ]
    else:
        lines += [
            "   *DATE 1988 10 01",
            "   *DATE 1989 01 01",
        ]
    lines += [
        "   *DATE 1989 07 01",
        "   *DATE 1990 01 01",
        "",
        "   *STOP",
        "",
    ]
    return "\n".join(lines)


def apply_template(src: str, case: str, kx, ky, kz) -> str:
    text = src
    # Do not search bare "*GRID": that hits "*WPRN *GRID" and swallows OUTPRN.
    text = re.sub(r"\*WPRN\s+\*GRID\s+\d+", "*WPRN   *GRID 5", text, count=1)
    text = text.replace(
        "*OUTPRN *GRID    *EXCEPT  *OILPOT  *DATUMPRES",
        "*OUTPRN *GRID *SO *SG *SW *PRES",
    )
    mgrid = re.search(r"(?m)^\s*\*GRID\s+\*CART\b", text)
    if not mgrid:
        raise RuntimeError("no *GRID *CART")
    g0 = text.rfind("\n", 0, mgrid.start()) + 1
    g1 = text.find("*MODEL")
    if g1 < 0:
        raise RuntimeError("no *MODEL")
    banner = text.rfind("********************************************************************************", 0, g1)
    if banner > g0:
        g1 = banner
    text = text[:g0] + _grid_block(kx, ky, kz) + "\n\n   " + text[g1:]
    titles = {
        "S1": ("Shale-oil analog S1 HW 5-frac depletion", "single HW five planar frac strips"),
        "S2": ("Shale-oil analog S2 HW 9-frac dense", "single HW nine tighter frac strips"),
        "S3": ("Shale-oil analog S3 two HW interference", "two parallel HWs depletion"),
        "S4": ("Shale-oil analog S4 parent-child", "HW1 first year then HW2"),
        "S5": ("Shale-oil analog S5 shut-in", "HW produce-shutin-reopen"),
    }
    t1, t2 = titles[case]
    # replace first quoted titles if present
    text = re.sub(r"'[^']{8,80}'", f"'{t1}'", text, count=1)
    w0 = text.find("*DTWELL")
    if w0 < 0:
        raise RuntimeError("no *DTWELL")
    text = text[:w0] + _well_block(case)
    return text


def write_case(case: str) -> dict:
    out_dir = HERE.parent / CASE_DIR[case]
    out_dir.mkdir(parents=True, exist_ok=True)
    kx, ky, kz, frac, srv = build_k(case)
    src = SRC.read_text(encoding="latin-1")
    dat = apply_template(src, case, kx, ky, kz)
    dat_path = out_dir / f"mxshale_{case.lower()}.dat"
    dat_path.write_text(dat, encoding="latin-1")
    frac_blocks, srv_blocks = [], []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                ijk = [i + 1, j + 1, k + 1]
                if frac[k, j, i]:
                    frac_blocks.append(ijk)
                elif srv[k, j, i]:
                    srv_blocks.append(ijk)
    k_mat = kx[~(frac | srv)]
    truth = {
        "scenario": case,
        "description": {
            "S1": "Single horizontal producer, 5 hydraulic-frac strips, depletion.",
            "S2": "Same pad as S1 with 9 tighter frac stages.",
            "S3": "Two parallel HWs on from t=0 (interference).",
            "S4": "Parent HW1 from t=0; child HW2 opens after ~1 year.",
            "S5": "Same as S1 with mid-life shut-in then reopen.",
        }[case],
        "analog_note": (
            "IMEX single-porosity black-oil analog of shale-oil depletion. "
            "Not GEM compositional, not adsorption, not dual-porosity."
        ),
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "di_ft": DI,
            "dj_ft": DJ,
            "dk_ft": DK.tolist(),
            "grid_type": "CART",
        },
        "units": "FIELD (md, ft, psi)",
        "matrix_perm_md": {
            "kx_geo": float(np.exp(np.mean(np.log(np.clip(k_mat, 1e-12, None))))),
        },
        "frac_perm_md": K_FRAC,
        "srv_perm_md": K_SRV,
        "frac_i_planes": frac_i_list(case if case != "S3" else "S1"),
        "wells": wells_for(case),
        "channel_blocks_ijk": frac_blocks,
        "srv_blocks_ijk": srv_blocks,
        "n_frac_blocks": int(np.sum(frac)),
        "n_srv_blocks": int(np.sum(srv)),
        "n_cells": NX * NY * NZ,
        "cloned_from": str(SRC),
    }
    # keep probe-study compatible alias
    truth["high_k_blocks_ijk"] = frac_blocks
    (out_dir / f"truth_{case.lower()}.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8"
    )
    readme = out_dir / "README.md"
    readme.write_text(
        f"# {case} {CASE_DIR[case]}\n\n{truth['description']}\n\n"
        f"{truth['analog_note']}\n\n"
        f"```bash\npython validation/shale_oil/cmg_shale_suite/run_imex.py --case {case}\n```\n",
        encoding="utf-8",
    )
    (out_dir / ".gitignore").write_text("*.out\n*.sr3\n*.rstr.sr3\n*.log\n", encoding="utf-8")
    return {
        "case": case,
        "dir": str(out_dir),
        "dat": str(dat_path),
        "n_frac": truth["n_frac_blocks"],
        "k_mat_geo": truth["matrix_perm_md"]["kx_geo"],
        "n_wells": len(wells_for(case)),
    }


def main() -> int:
    if not SRC.is_file():
        raise SystemExit(f"missing official sample {SRC}")
    reports = [write_case(c) for c in CASES]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
