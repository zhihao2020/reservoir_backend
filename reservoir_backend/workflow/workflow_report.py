"""Runner for the Industrialization Goal workflow reports."""

from __future__ import annotations

import json

from reservoir_backend.workflow.industrial_case import run_industrial_case_workflow


def run_workflow_report(output_dir: str = "accuracy_reports") -> dict[str, object]:
    """Run the currently available industrial workflow report."""
    return run_industrial_case_workflow(output_dir=output_dir)


def main() -> None:
    summary = run_workflow_report()
    print(json.dumps({"success": summary["success"], "report": summary["engineering_report_json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
