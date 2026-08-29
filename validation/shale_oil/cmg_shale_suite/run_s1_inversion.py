"""Offline S1 shale inversion vs IMEX PRES (ruler only; no CMG at invert time).

Usage (repo root):
  python validation/shale_oil/cmg_shale_suite/run_s1_inversion.py
"""

from __future__ import annotations

from run_suite_inversion import main

if __name__ == "__main__":
    raise SystemExit(main(["--case", "S1"]))
