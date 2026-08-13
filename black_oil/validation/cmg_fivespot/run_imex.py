"""Run IMEX on the five-spot ruler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_fivespot.dat"
IMEX_EXE = Path(r"D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe")


def main() -> int:
    if not IMEX_EXE.is_file():
        print(f"IMEX not found: {IMEX_EXE}", file=sys.stderr)
        return 2
    cmd = [str(IMEX_EXE), "-f", DAT.name]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(HERE))
    out = HERE / "mxspr006_fivespot.out"
    text = out.read_text(encoding="latin-1", errors="replace") if out.is_file() else ""
    ok = "Normal Termination" in text
    print(f"returncode={proc.returncode} out={out.is_file()} normal={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
