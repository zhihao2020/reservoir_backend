"""Runner for IND-003 well schedule model report."""

from __future__ import annotations

import json

from reservoir_backend.schedule.well_schedule import run_well_schedule_report


def main() -> None:
    summary = run_well_schedule_report()
    print(json.dumps({"success": summary["success"], "report": summary["report_json_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
