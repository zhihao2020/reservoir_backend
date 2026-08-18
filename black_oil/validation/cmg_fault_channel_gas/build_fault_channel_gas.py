"""CART fault+channel with explicit free gas. Clone of fault_channel.dat."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC_DIR = ROOT / "black_oil" / "validation" / "cmg_fault_channel"
SRC = SRC_DIR / "fault_channel.dat"
DAT = HERE / "fault_channel_gas.dat"
TRUTH = HERE / "truth_fault_channel_gas.json"

SG = 0.10
SO = 0.70
SW = 0.20


def build() -> dict:
    if not SRC.is_file():
        raise FileNotFoundError(f"need {SRC}; run build_fault_channel.py first")
    text = SRC.read_text(encoding="latin-1")
    text = text.replace("'Fault+channel invert ruler'", "'Fault+channel free-gas invert ruler'")
    # IMEX *BLACKOIL_SEAWATER: Sg = 1 - So - Sw. There is no *SG init keyword.
    old = "   *SO   *CON     0.8\n   *SW   *CON     0.2"
    new = f"   *SO   *CON     {SO:.2f}\n   *SW   *CON     {SW:.2f}"
    if old not in text:
        raise ValueError("SO/SW init block not found")
    text = text.replace(old, new, 1)
    DAT.write_text(text, encoding="latin-1")

    for name in ("region_id.npy", "face_mult_x.npy", "k_true.npy"):
        src = SRC_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"need {src}")
        shutil.copy2(src, HERE / name)

    base = json.loads((SRC_DIR / "truth_fault_channel.json").read_text(encoding="utf-8"))
    base["description"] = "CART fault+channel with explicit free gas Sg=0.10 (P>Pb, no liberation)"
    base["controls"]["swi"] = SW
    base["controls"]["sgi"] = SG
    base["controls"]["soi"] = SO
    base["controls"]["pb_psi"] = 2500.0
    TRUTH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"wrote {DAT.name}  Sw={SW} So={SO} Sg={SG}")
    return base


if __name__ == "__main__":
    build()
