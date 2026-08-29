"""Clone GEM TPL gmspr003 and patch Jiyang 1-inj 4-prod CO2 huff-n-puff.

Fluid is the published EXAMPLE card (CO2 + C1 + nC10 from OPM 1D_COMP).
Not a Jiyang field GEM card. Do not invent Tc/Pc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # validation/jiyang/cmg_co2_hnp → repo root
CLONE_SRC = Path(r"D:\Tool\CMG\GEM\2024.20\TPL\spr\gmspr003.dat")
CARD = ROOT / "examples" / "compositional" / "fixtures" / "comp_c1c10co2.yaml"
REGIONS = ROOT / "examples" / "jiyang" / "jiyang_frac_regions.npy"
OUT_DAT = HERE / "jiyang_co2_hnp.dat"

NX, NY, NZ = 21, 21, 5
SIZE_M = (1260.0, 1260.0, 40.0)
PHI = 0.06
P_INIT_KPA = 50000.0
P_PROD_KPA = 47000.0
P_SOAK_KPA = 50000.0
TRES_C = 76.85  # 350 K, same as product EXAMPLE
Q_INJ_M3S = 5.0e-5
Q_INJ_M3D = Q_INJ_M3S * 86400.0  # GEM SI *STG is m3/day
SW_INIT = 0.25

# Horizontal laterals locked to examples/jiyang/jiyang_hnp.yaml (1-based IJK).
WELLS = {
    "INJ": (11, 3),
    "P1": (3, 3),
    "P2": (7, 3),
    "P3": (15, 3),
    "P4": (19, 3),
}
I_LAT = range(4, 19)  # I = 4..18


def _load_card() -> dict:
    data = yaml.safe_load(CARD.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{CARD} must be a mapping")
    names = [str(x) for x in data["names"]]
    if names != ["CO2", "C1", "nC10"]:
        raise ValueError(f"unexpected EXAMPLE names {names}")
    return data


def _relperm_tables(src: Path) -> tuple[str, str]:
    text = src.read_text(encoding="latin-1")
    i_swt = text.find("*SWT")
    i_sgt = text.find("*SGT")
    i_end = text.find("**-------------------------------------INITIAL")
    if min(i_swt, i_sgt, i_end) < 0 or not (i_swt < i_sgt < i_end):
        raise ValueError(f"cannot extract *SWT/*SGT from {src}")
    return text[i_swt:i_sgt].rstrip() + "\n", text[i_sgt:i_end].rstrip() + "\n"


def _perm_md() -> np.ndarray:
    """Region geometry from the frac map; md levels are GEM-stable, not npy SI."""
    regions = np.asarray(np.load(REGIONS), dtype=int).ravel()
    if regions.size != NX * NY * NZ:
        raise ValueError(f"{REGIONS} size {regions.size} != {NX * NY * NZ}")
    # npy SI spans 1e-18..8e-12 m2 (~0.001..8000 md); that PI blows the first Newton.
    # Invert still sees the same 2-region shape. Contrast 100x.
    return np.where(regions == 1, 5.0, 0.05).astype(float)


def _ijk_block(perm_md: np.ndarray) -> str:
    lines = ["*PERMI *IJK"]
    n = 0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                val = float(perm_md[n])
                lines.append(f"  {i + 1:2d} {j + 1:2d} {k + 1:2d}  {val:.6g}")
                n += 1
    lines.append("*PERMJ *EQUALSI")
    lines.append("*PERMK *EQUALSI * 0.10")
    return "\n".join(lines) + "\n"


def _perf(j: int, k: int) -> str:
    rows = [f"            {i} {j} {k}  1.0" for i in I_LAT]
    return "\n".join(rows)


def _bin_lower(kij: list) -> str:
    # GEM *BIN is lower triangle without the diagonal: (2,1), (3,1) (3,2), ...
    a = np.asarray(kij, dtype=float)
    rows = []
    for i in range(1, a.shape[0]):
        rows.append("  " + "  ".join(f"{float(a[i, j]):.5f}" for j in range(i)))
    return "\n".join(rows)


def build_deck() -> Path:
    if not CLONE_SRC.is_file():
        raise FileNotFoundError(f"GEM clone source missing: {CLONE_SRC}")
    card = _load_card()
    swt, sgt = _relperm_tables(CLONE_SRC)
    perm = _perm_md()
    regions = np.asarray(np.load(REGIONS), dtype=int).ravel()
    if regions.size != perm.size:
        raise ValueError("region_map and perm size mismatch")
    names = ["CO2", "C1", "NC10"]  # GEM-safe alias of nC10
    pc_atm = np.asarray(card["pc_bar"], dtype=float)  # bar ≈ atm for this card
    tc = np.asarray(card["tc_k"], dtype=float)
    ac = np.asarray(card["omega"], dtype=float)
    mw = np.asarray(card["mw_g_mol"], dtype=float)
    # Reid/Prausnitz molar volumes (m3/kmol) for the named species, not Jiyang.
    vcrit = np.array([0.0940, 0.0990, 0.6240])
    pchor = np.array([78.0, 77.0, 431.0])
    di = SIZE_M[0] / NX
    dj = SIZE_M[1] / NY
    dk = SIZE_M[2] / NZ

    inj_comp = "1.0  0.0  0.0"
    wells = [
        ("1", "INJ", WELLS["INJ"], True),
        ("2", "P1", WELLS["P1"], False),
        ("3", "P2", WELLS["P2"], False),
        ("4", "P3", WELLS["P3"], False),
        ("5", "P4", WELLS["P4"], False),
    ]
    well_decl = "\n".join(f"      *WELL {num} '{name}'" for num, name, _, _ in wells)
    perf_block = []
    for num, _name, (j, k), _inj in wells:
        perf_block.append(f"      *PERF *GEO  {num}")
        perf_block.append(_perf(j, k))
    producer_ops_prod = "\n".join(
        f"      *PRODUCER  {num}\n            *OPERATE  *MIN  *BHP  {P_PROD_KPA:.1f}"
        for num, _n, _jk, inj in wells
        if not inj
    )
    producer_ops_soak = "\n".join(
        f"      *PRODUCER  {num}\n            *OPERATE  *MIN  *BHP  {P_SOAK_KPA:.1f}"
        for num, _n, _jk, inj in wells
        if not inj
    )

    dat = f"""**--------------------------------------------------------------------**
** FILE :  jiyang_co2_hnp.dat
** CLONE:  {CLONE_SRC}
**         GEM TPL gmspr003.dat (CO2 pattern flood). Relperm tables copied.
** FLUID:  EXAMPLE PR CO2+C1+nC10 from examples/compositional/fixtures/comp_c1c10co2.yaml
**         (OPM opm-tests compositional/1D_COMP.DATA). NOT a Jiyang GEM card.
** GRID :  CART {NX}x{NY}x{NZ}  size_m={list(SIZE_M)}  ports from jiyang_hnp.yaml
** WELLS:  1 injector + 4 horizontal producers (I=4..18, k=3)
** K    :  jiyang_frac_regions.npy shape; matrix 0.05 md, SRV 5 md (100x)
** SCHED:  90 d depletion, then 1 cycle inj 1 mo / soak 1 mo / prod 10 mo
** GATE :  well rates / BHP only. Not field Dice. Product does not call GEM.
**--------------------------------------------------------------------**

*RESULTS *SIMULATOR *GEM
*FILENAMES *OUTPUT *SRFOUT *RESTARTOUT *INDEXOUT *MAINRESULTSOUT
*TITLE1 'Jiyang 1-inj 4-prod CO2 HnP'
*TITLE2 'EXAMPLE C1-nC10-CO2 - not Jiyang PVT'
*INUNIT *SI

*WRST 0
*WPRN *WELL *TIME
*WPRN *GRID *TIME
*WSRF *WELL 1
*WSRF *GRID 1
*OUTPRN *WELL *BRIEF
*OUTPRN *GRID *PRES *SO *SG *SW *Z 'CO2' *Z 'C1' *Z 'NC10'
*OUTSRF *WELL *RECO *RECG *TOIP *TGIP
*OUTSRF *GRID *SW *SO *SG *Z 'CO2'

**-------------------------------------RESERVOIR & GRID DATA------------
*GRID *CART {NX} {NY} {NZ}
*DI *CON {di:.6f}
*DJ *CON {dj:.6f}
*DK *CON {dk:.6f}
*DEPTH 1 1 1  2500.0
*POR *CON {PHI}
{_ijk_block(perm)}
**-------------------------------------FLUID PROPERTY DATA--------------
** PCRIT in atm (GEM convention even with *INUNIT *SI). Values = pc_bar
** from the published EXAMPLE YAML (CO2 73.773 bar, C1 45.992, nC10 21.03).
*MODEL *PR
*NC 3 3
*COMPNAME
'{names[0]}' '{names[1]}' '{names[2]}'
*HCFLAG  0  0  0
*PCRIT   {pc_atm[0]:.5f}  {pc_atm[1]:.5f}  {pc_atm[2]:.5f}
*VCRIT   {vcrit[0]:.5f}  {vcrit[1]:.5f}  {vcrit[2]:.5f}
*TCRIT   {tc[0]:.3f}  {tc[1]:.3f}  {tc[2]:.3f}
*AC      {ac[0]:.5f}  {ac[1]:.5f}  {ac[2]:.5f}
*MW      {mw[0]:.2f}  {mw[1]:.2f}  {mw[2]:.2f}
*PVC3 1.20
*VSHIFT  0.0  0.0  0.0
*VISCOR *HZYT
*MIXVC 1.0
*VISVC   {vcrit[0]:.5f}  {vcrit[1]:.5f}  {vcrit[2]:.5f}
*VISCOEFF  0.1023  0.023364  0.058533  -0.040758  0.0093324
*OMEGA   0.45723553  0.45723553  0.45723553
*OMEGB   0.077796074  0.077796074  0.077796074
*PCHOR   {pchor[0]:.1f}  {pchor[1]:.1f}  {pchor[2]:.1f}
*BIN
{_bin_lower(card["kij"])}
*TRES  {TRES_C:.3f}
*PHASEID *CRIT
**-------------------------------------ROCK FLUID DATA------------------
*ROCKFLUID
*RPT
{swt}{sgt}**-------------------------------------INITIAL RESERVOIR CONDITION------
*INITIAL
*VERTICAL *OFF
*PRES *CON {P_INIT_KPA:.1f}
*SW   *CON {SW_INIT:.5f}
*ZGLOBALC 'CO2' CON  0.01
*ZGLOBALC 'C1'  CON  0.54
*ZGLOBALC 'NC10' CON  0.45
**-------------------------------------NUMERICAL METHOD-----------------
*NUMERICAL
*NORM *PRESS  100.0
*NORM *SATUR  0.2
*NORM *GMOLAR 0.2
*DTMIN  1.0e-5
*DTMAX  10.0
**-------------------------------------WELL DATA------------------------
*RUN
*DATE 2000 1 1
      *DTWELL 0.01
{well_decl}
      *INJECTOR  1
            *INCOMP    *SOLVENT    {inj_comp}
            *OPERATE   *MAX  *STG  {Q_INJ_M3D:.6f}
      *SHUTIN  1
{producer_ops_prod}
      *GEOMETRY *I  0.10  0.37  1.0  0.0
{chr(10).join(perf_block)}

** 90-day depletion ends. Open injector; producers to soak BHP.
*DATE 2000 4 1
      *OPEN  1
      *INJECTOR  1
            *OPERATE   *MAX  *STG  {Q_INJ_M3D:.6f}
{producer_ops_soak}

** Soak: shut injector, keep producers at soak BHP.
*DATE 2000 5 1
      *SHUTIN  1
{producer_ops_soak}

** Produce 10 months.
*DATE 2000 6 1
      *SHUTIN  1
{producer_ops_prod}

*DATE 2001 4 1
      *STOP
"""
    OUT_DAT.write_text(dat.replace("\n", "\r\n"), encoding="latin-1")
    return OUT_DAT


def self_check(path: Path | None = None) -> None:
    text = (path or OUT_DAT).read_text(encoding="latin-1")
    upper = text.upper()
    if "NOT A JIYANG" not in upper:
        raise AssertionError("deck must declare it is not a Jiyang field card")
    for token in (
        "*GRID *CART 21 21 5",
        "*NC 3 3",
        "*WELL 1 'INJ'",
        "*WELL 5 'P4'",
        "73.773",
        "304.128",
        "GMSPR003",
        "*DATE 2000 4 1",
        "*DATE 2001 4 1",
    ):
        if token.upper() not in upper:
            raise AssertionError(f"missing {token}")
    if text.upper().count("*WELL") < 5:
        raise AssertionError("expected 5 wells")


if __name__ == "__main__":
    out = build_deck()
    self_check(out)
    print(f"wrote {out} bytes={out.stat().st_size}")
