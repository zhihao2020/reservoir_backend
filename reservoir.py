"""Run the backend from a source checkout or from Linux ``src.so``.

``python -m src`` works on the source tree only. The compiled module has no
code object, so the joint-debug package uses this launcher instead:

    python3 reservoir.py --version
    python3 reservoir.py examples/small/case.yaml -o results/small
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
