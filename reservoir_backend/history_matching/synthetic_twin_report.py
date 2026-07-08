"""Runner for IND-004 synthetic twin history-matching prototype."""

from __future__ import annotations

import json

from reservoir_backend.history_matching.synthetic_twin import run_synthetic_twin_history_matching


def main() -> None:
    summary = run_synthetic_twin_history_matching()
    print(json.dumps({"success": summary["success"], "rmse_after": summary["rmse_after"]}, sort_keys=True))


if __name__ == "__main__":
    main()
