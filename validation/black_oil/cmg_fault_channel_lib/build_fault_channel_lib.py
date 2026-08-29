"""CART fault+channel liberation ruler. Clone of fault_channel_gas.dat.

Initial Sg=0 (So=0.80, Sw=0.20). PROD BHP 1800 < pb 2500 so free gas
appears only by live-oil flash.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC_DIR = ROOT / "black_oil" / "validation" / "cmg_fault_channel_gas"
SRC = SRC_DIR / "fault_channel_gas.dat"
DAT = HERE / "fault_channel_lib.dat"
TRUTH = HERE / "truth_fault_channel_lib.json"

SO = 0.80
SW = 0.20
SG = 0.00
PROD_BHP = 1800.0


def build() -> dict:
    if not SRC.is_file():
        raise FileNotFoundError(f"need {SRC}")
    text = SRC.read_text(encoding="latin-1")
    text = text.replace(
        "'Fault+channel free-gas invert ruler'",
        "'Fault+channel live-oil liberation ruler'",
    )
    old_sat = "   *SO   *CON     0.70\n   *SW   *CON     0.20"
    new_sat = f"   *SO   *CON     {SO:.2f}\n   *SW   *CON     {SW:.2f}"
    if old_sat not in text:
        raise ValueError("SO/SW init block not found")
    text = text.replace(old_sat, new_sat, 1)
    old_bhp = "   *OPERATE   *MIN       *BHP    2800.0"
    new_bhp = f"   *OPERATE   *MIN       *BHP    {PROD_BHP:.1f}"
    if old_bhp not in text:
        raise ValueError("PROD BHP operate card not found")
    text = text.replace(old_bhp, new_bhp, 1)
    DAT.write_text(text, encoding="latin-1")

    for name in ("region_id.npy", "face_mult_x.npy", "k_true.npy"):
        src = SRC_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"need {src}")
        shutil.copy2(src, HERE / name)

    base = json.loads((SRC_DIR / "truth_fault_channel_gas.json").read_text(encoding="utf-8"))
    base["description"] = (
        "CART fault+channel live-oil liberation: Sg=0 initially, PROD BHP 1800 < pb 2500"
    )
    base["controls"]["swi"] = SW
    base["controls"]["sgi"] = SG
    base["controls"]["soi"] = SO
    base["controls"]["prod_bhp_psi"] = PROD_BHP
    base["controls"]["pb_psi"] = 2500.0
    TRUTH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"wrote {DAT.name}  Sw={SW} So={SO} Sg={SG}  PROD BHP={PROD_BHP}")
    return base


if __name__ == "__main__":
    build()
