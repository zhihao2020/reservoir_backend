"""Report runner for the lightweight IMPES sequential loop."""

from __future__ import annotations

import json
from pathlib import Path

from reservoir_backend.simulation.impes import create_synthetic_waterflood_case, run_impes_simulation


def run_impes_report(output_dir: str | Path = "accuracy_reports") -> dict[str, object]:
    """Run the synthetic IMPES case and write JSON/Markdown summaries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = run_impes_simulation(create_synthetic_waterflood_case())
    summary = result.summary
    summary["report_json_path"] = str(output_path / "impes_loop_summary.json")
    summary["report_markdown_path"] = str(output_path / "impes_loop_summary.md")
    summary["source_task"] = "F3-04"
    json_path = output_path / "impes_loop_summary.json"
    md_path = output_path / "impes_loop_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# IMPES Sequential Loop Summary",
        "",
        f"- success: {summary['success']}",
        f"- case_id: {summary['case_id']}",
        f"- num_steps: {summary['num_steps']}",
        f"- grid_shape: {summary['grid_shape']}",
        f"- max_cfl: {summary['max_cfl']}",
        f"- max_mass_balance_error: {summary['max_mass_balance_error']}",
        f"- final_water_cut: {summary['final_water_cut']}",
        f"- breakthrough_time: {summary['breakthrough_time']}",
        "",
        "## Production Curve",
        "",
        "| step | time | total_liquid_rate | water_rate | oil_rate | water_cut |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in summary["production_curve"]:
        lines.append(
            "| {step} | {time:.6g} | {total_liquid_rate:.6g} | {water_rate:.6g} | {oil_rate:.6g} | {water_cut:.6g} |".format(
                **entry
            )
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- No fully implicit simulator is implemented.",
            "- No black-oil model or PVT behavior is implemented.",
            "- No complex well-control model is implemented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_impes_report()
    print(json.dumps({"success": summary["success"], "report": summary["report_json_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
