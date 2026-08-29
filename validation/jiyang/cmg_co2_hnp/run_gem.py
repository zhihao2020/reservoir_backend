"""Run the Jiyang-pattern GEM CO2 huff-n-puff ruler if a local license exists."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAT = HERE / "jiyang_co2_hnp.dat"
GEM_EXE = Path(r"D:\Tool\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe")


def main() -> int:
    if not DAT.is_file():
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from build_deck import build_deck, self_check

        build_deck()
        self_check()
    if not GEM_EXE.is_file():
        print(f"GEM not found: {GEM_EXE}", file=sys.stderr)
        return 2
    cmd = [str(GEM_EXE), "-f", DAT.name]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(HERE))
    out = HERE / "jiyang_co2_hnp.out"
    text = out.read_text(encoding="latin-1", errors="replace") if out.is_file() else ""
    ok = "End of simulation" in text and "FATAL ERROR" not in text.upper()
    fatal = "FATAL ERROR" in text.upper()
    print(f"returncode={proc.returncode} out={out.is_file()} ok={ok} fatal={fatal}")
    if fatal:
        for line in text.splitlines():
            if "ERROR" in line.upper() or "FATAL" in line.upper():
                print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
