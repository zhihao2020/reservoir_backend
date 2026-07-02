"""Script wrapper for `python -m reservoir_backend.cli.run_case`."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reservoir_backend.cli.run_case import main


if __name__ == "__main__":
    raise SystemExit(main())
