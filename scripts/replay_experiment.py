"""Non-realtime experiment replay from experiments/EXP00N (no UDP socket)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.runtime.replay import replay_experiment as replay


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    report = replay(args.experiment, output=args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
