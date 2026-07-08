"""Multi-well schedule model v0.

The schedule layer validates industrial case schedule metadata and control
interfaces. It does not implement Peaceman well indices, complex wellbore
networks, black-oil well controls, or pressure-solver rewrites.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from reservoir_backend.project.project_registry import json_safe


WELL_TYPES = {"injector", "producer"}
CONTROL_TYPES = {"rate", "bhp"}
WELL_STATUSES = {"open", "shut"}


@dataclass(frozen=True)
class WellControlStep:
    """One well-control instruction at a schedule time."""

    well_id: str
    time: float
    well_type: str
    control_type: str
    target: float
    unit: str
    status: str = "open"
    report_step: bool = False
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WellControlStep":
        normalized = validate_control_step(data)
        return cls(**normalized)


@dataclass(frozen=True)
class WellSchedule:
    """Validated multi-well schedule."""

    schedule_id: str
    steps: list[WellControlStep]
    report_interval: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "schedule_id": self.schedule_id,
                "steps": [step.to_dict() for step in self.steps],
                "report_interval": self.report_interval,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WellSchedule":
        normalized = validate_well_schedule(data)
        return cls(
            schedule_id=normalized["schedule_id"],
            steps=[WellControlStep.from_dict(item) for item in normalized["steps"]],
            report_interval=normalized["report_interval"],
            metadata=normalized["metadata"],
        )


def validate_control_step(step: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one well-control step."""
    data = dict(step)
    required = ("well_id", "time", "well_type", "control_type", "target", "unit", "status")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"schedule step missing required keys: {missing}")
    well_id = str(data["well_id"]).strip()
    if not well_id:
        raise ValueError("well_id must be non-empty")
    well_type = str(data["well_type"]).strip().lower()
    control_type = str(data["control_type"]).strip().lower()
    status = str(data.get("status", "open")).strip().lower()
    if well_type not in WELL_TYPES:
        raise ValueError(f"unsupported well_type: {well_type}")
    if control_type not in CONTROL_TYPES:
        raise ValueError(f"unsupported control_type: {control_type}")
    if status not in WELL_STATUSES:
        raise ValueError(f"unsupported well status: {status}")
    time_value = float(data["time"])
    target = float(data["target"])
    if time_value < 0.0:
        raise ValueError("schedule time must be nonnegative")
    if control_type == "rate" and target < 0.0:
        raise ValueError("rate control target must be nonnegative")
    if control_type == "bhp" and target <= 0.0:
        raise ValueError("BHP control target must be positive")
    constraints = dict(data.get("constraints", {}))
    _validate_constraints(constraints)
    return {
        "well_id": well_id,
        "time": time_value,
        "well_type": well_type,
        "control_type": control_type,
        "target": target,
        "unit": str(data["unit"]),
        "status": status,
        "report_step": bool(data.get("report_step", False)),
        "constraints": constraints,
        "metadata": dict(data.get("metadata", {})),
    }


def validate_well_schedule(schedule: WellSchedule | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a schedule object or mapping."""
    data = schedule.to_dict() if isinstance(schedule, WellSchedule) else dict(schedule)
    if not str(data.get("schedule_id", "")).strip():
        raise ValueError("schedule_id must be non-empty")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("schedule steps must be a non-empty list")
    steps = [validate_control_step(step) for step in raw_steps]
    _validate_time_ordering(steps)
    report_interval = data.get("report_interval")
    if report_interval is not None and float(report_interval) <= 0.0:
        raise ValueError("report_interval must be positive")
    return {
        "schedule_id": str(data["schedule_id"]),
        "steps": steps,
        "report_interval": None if report_interval is None else float(report_interval),
        "metadata": dict(data.get("metadata", {})),
    }


def generate_report_steps(schedule: WellSchedule | Mapping[str, Any]) -> list[float]:
    """Generate sorted report-step times from explicit flags and interval."""
    data = validate_well_schedule(schedule)
    times = {float(step["time"]) for step in data["steps"] if step["report_step"]}
    if data["report_interval"] is not None:
        start = min(float(step["time"]) for step in data["steps"])
        end = max(float(step["time"]) for step in data["steps"])
        current = start
        while current <= end + 1.0e-12:
            times.add(round(current, 12))
            current += float(data["report_interval"])
    if not times:
        times = {float(step["time"]) for step in data["steps"]}
    return sorted(times)


def build_schedule_summary(schedule: WellSchedule | Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable schedule summary report."""
    data = validate_well_schedule(schedule)
    steps = data["steps"]
    injector_count = len({step["well_id"] for step in steps if step["well_type"] == "injector"})
    producer_count = len({step["well_id"] for step in steps if step["well_type"] == "producer"})
    open_steps = sum(1 for step in steps if step["status"] == "open")
    shut_steps = sum(1 for step in steps if step["status"] == "shut")
    summary = {
        "summary_name": "well_schedule_summary",
        "source_task": "IND-003",
        "success": True,
        "schedule_id": data["schedule_id"],
        "num_steps": len(steps),
        "num_wells": len({step["well_id"] for step in steps}),
        "num_injectors": injector_count,
        "num_producers": producer_count,
        "control_types": sorted({step["control_type"] for step in steps}),
        "statuses": sorted({step["status"] for step in steps}),
        "open_steps": open_steps,
        "shut_steps": shut_steps,
        "report_steps": generate_report_steps(data),
        "production_constraints": _collect_constraints(steps),
        "warnings": [],
        "limitations": [
            "Schedule v0 validates controls and metadata only.",
            "No full Peaceman industrial well model.",
            "No complex wellbore network.",
            "No black-oil well control.",
            "No pressure solver rewrite.",
        ],
    }
    return json_safe(summary)


def write_schedule_summary_report(summary: Mapping[str, Any], output_dir: str | Path = "accuracy_reports") -> dict[str, str]:
    """Write the schedule summary JSON and Markdown reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "well_schedule_model_summary.json"
    md_path = root / "well_schedule_model_summary.md"
    json_path.write_text(json.dumps(json_safe(dict(summary)), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def create_demo_schedule() -> WellSchedule:
    """Return a deterministic multi-well schedule fixture."""
    return WellSchedule(
        schedule_id="well_schedule_v0_demo",
        report_interval=1.0,
        steps=[
            WellControlStep("I1", 0.0, "injector", "rate", 100.0, "m3/day", "open", True),
            WellControlStep("P1", 0.0, "producer", "bhp", 9.0, "MPa", "open", True, {"min_oil_rate": 1.0}),
            WellControlStep("I1", 2.0, "injector", "rate", 80.0, "m3/day", "open", True),
            WellControlStep("P1", 3.0, "producer", "rate", 50.0, "m3/day", "shut", True),
        ],
        metadata={"source_task": "IND-003"},
    )


def run_well_schedule_report(output_dir: str | Path = "accuracy_reports") -> dict[str, Any]:
    """Generate the default schedule summary report."""
    summary = build_schedule_summary(create_demo_schedule())
    paths = write_schedule_summary_report(summary, output_dir)
    summary["report_json_path"] = paths["json"]
    summary["report_markdown_path"] = paths["markdown"]
    return summary


def _validate_time_ordering(steps: list[dict[str, Any]]) -> None:
    last_by_well: dict[str, float] = {}
    for step in steps:
        well_id = step["well_id"]
        time_value = float(step["time"])
        if well_id in last_by_well and time_value < last_by_well[well_id]:
            raise ValueError(f"schedule time must be nondecreasing for well {well_id}")
        last_by_well[well_id] = time_value


def _validate_constraints(constraints: Mapping[str, Any]) -> None:
    for key, value in constraints.items():
        if key in {"min_oil_rate", "max_water_cut", "max_gas_rate"} and float(value) < 0.0:
            raise ValueError(f"constraint {key} must be nonnegative")
        if key == "max_water_cut" and float(value) > 1.0:
            raise ValueError("max_water_cut must be <= 1")


def _collect_constraints(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        if step["constraints"]:
            result.setdefault(step["well_id"], []).append(dict(step["constraints"]))
    return result


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Well Schedule Model Summary",
        "",
        "## Implemented Scope",
        "",
        f"- success: {summary['success']}",
        f"- schedule_id: {summary['schedule_id']}",
        f"- num_wells: {summary['num_wells']}",
        f"- control_types: {', '.join(summary['control_types'])}",
        f"- report_steps: {summary['report_steps']}",
        "",
        "## Test Results",
        "",
        "- See `tests/test_well_schedule_model.py` and `pytest -q`.",
        "",
        "## Known Limitations",
        "",
    ]
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(
        [
            "- No full Peaceman industrial well model.",
            "- No complex wellbore network.",
            "- No black-oil well control.",
        ]
    )
    lines.extend(["", "## Next Steps", "", "- Connect schedule v0 to industrial workflow config in a later task."])
    return "\n".join(lines) + "\n"
