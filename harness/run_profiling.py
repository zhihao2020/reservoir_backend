"""Harness wrapper for profiling the full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_full_pipeline import run_profiling


def main() -> None:
    """Run profiling from the harness entry point."""
    summary = run_profiling()
    print(f"profiling success={summary['success']}")


if __name__ == "__main__":
    main()
