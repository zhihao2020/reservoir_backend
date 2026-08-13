"""Run IMEX on the fine channel ruler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAT = HERE / "mxspr006_channel_fine.dat"
IMEX_EXE = Path(r"D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe")


def main() -> int:
    if not IMEX_EXE.is_file():
        print(f"IMEX not found: {IMEX_EXE}", file=sys.stderr)
        return 2
    if not DAT.is_file():
        print(f"missing {DAT}", file=sys.stderr)
        return 2
    cmd = [str(IMEX_EXE), "-f", DAT.name]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(HERE))
    out = HERE / "mxspr006_channel_fine.out"
    log = HERE / "mxspr006_channel_fine.log"
    text = ""
    for p in (out, log):
        if p.is_file():
            text += p.read_text(encoding="latin-1", errors="replace")
    ok = "Normal Termination" in text or proc.returncode == 0
    print(f"returncode={proc.returncode} out_exists={out.is_file()} normal={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
