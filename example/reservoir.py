"""Launch the backend from this folder, whether ``src`` is source or ``src.so``.

A Nuitka module has no code object, so ``python -m src`` does not work on the
compiled library. This launcher does.

    python reservoir.py case.yaml --tcp-port 9000 --ip 127.0.0.1 \\
        --lab-port 9001 --field-port 9002
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
