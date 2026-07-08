from __future__ import annotations

import json
from pathlib import Path

import pytest

from reservoir_backend.schedule.well_schedule import (
    WellControlStep,
    WellSchedule,
    build_schedule_summary,
    create_demo_schedule,
    generate_report_steps,
    run_well_schedule_report,
    validate_control_step,
    validate_well_schedule,
    write_schedule_summary_report,
)


def test_multi_well_schedule() -> None:
    summary = build_schedule_summary(create_demo_schedule())
    assert summary["num_wells"] == 2
    assert summary["num_steps"] == 4


def test_injector_schedule() -> None:
    summary = build_schedule_summary(create_demo_schedule())
    assert summary["num_injectors"] == 1


def test_producer_schedule() -> None:
    summary = build_schedule_summary(create_demo_schedule())
    assert summary["num_producers"] == 1


def test_rate_control_validation() -> None:
    step = validate_control_step({"well_id": "I1", "time": 0.0, "well_type": "injector", "control_type": "rate", "target": 1.0, "unit": "m3/day", "status": "open"})
    assert step["control_type"] == "rate"


def test_bhp_control_interface_validation() -> None:
    step = validate_control_step({"well_id": "P1", "time": 0.0, "well_type": "producer", "control_type": "bhp", "target": 10.0, "unit": "MPa", "status": "open"})
    assert step["control_type"] == "bhp"


def test_report_step_generation() -> None:
    steps = generate_report_steps(create_demo_schedule())
    assert steps == [0.0, 1.0, 2.0, 3.0]


def test_well_status_open_shut() -> None:
    summary = build_schedule_summary(create_demo_schedule())
    assert summary["statuses"] == ["open", "shut"]
    assert summary["shut_steps"] == 1


def test_invalid_schedule_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_well_schedule({"schedule_id": "bad", "steps": []})


def test_invalid_well_type_rejected() -> None:
    with pytest.raises(ValueError, match="well_type"):
        validate_control_step({"well_id": "X", "time": 0.0, "well_type": "observer", "control_type": "rate", "target": 1.0, "unit": "m3/day", "status": "open"})


def test_invalid_control_type_rejected() -> None:
    with pytest.raises(ValueError, match="control_type"):
        validate_control_step({"well_id": "X", "time": 0.0, "well_type": "injector", "control_type": "foo", "target": 1.0, "unit": "m3/day", "status": "open"})


def test_negative_rate_rejected() -> None:
    with pytest.raises(ValueError, match="rate"):
        validate_control_step({"well_id": "I1", "time": 0.0, "well_type": "injector", "control_type": "rate", "target": -1.0, "unit": "m3/day", "status": "open"})


def test_nonpositive_bhp_rejected() -> None:
    with pytest.raises(ValueError, match="BHP"):
        validate_control_step({"well_id": "P1", "time": 0.0, "well_type": "producer", "control_type": "bhp", "target": 0.0, "unit": "MPa", "status": "open"})


def test_time_ordering_rejected() -> None:
    data = {
        "schedule_id": "bad_time",
        "steps": [
            {"well_id": "I1", "time": 1.0, "well_type": "injector", "control_type": "rate", "target": 1.0, "unit": "m3/day", "status": "open"},
            {"well_id": "I1", "time": 0.0, "well_type": "injector", "control_type": "rate", "target": 1.0, "unit": "m3/day", "status": "open"},
        ],
    }
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_well_schedule(data)


def test_schedule_summary_report(tmp_path: Path) -> None:
    summary = build_schedule_summary(create_demo_schedule())
    paths = write_schedule_summary_report(summary, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert "Well Schedule Model Summary" in Path(paths["markdown"]).read_text(encoding="utf-8")


def test_schedule_summary_json_serializable() -> None:
    json.dumps(build_schedule_summary(create_demo_schedule()))


def test_production_constraints_documented() -> None:
    summary = build_schedule_summary(create_demo_schedule())
    assert "P1" in summary["production_constraints"]
    assert summary["production_constraints"]["P1"][0]["min_oil_rate"] == 1.0


def test_dataclass_roundtrip() -> None:
    schedule = create_demo_schedule()
    restored = WellSchedule.from_dict(schedule.to_dict())
    assert restored.schedule_id == schedule.schedule_id
    assert isinstance(restored.steps[0], WellControlStep)


def test_runner_generates_report(tmp_path: Path) -> None:
    summary = run_well_schedule_report(tmp_path)
    assert summary["success"] is True
    assert Path(summary["report_json_path"]).exists()


def test_no_black_oil_claim() -> None:
    text = "\n".join(build_schedule_summary(create_demo_schedule())["limitations"])
    assert "No black-oil well control." in text
    assert "No full Peaceman industrial well model." in text
