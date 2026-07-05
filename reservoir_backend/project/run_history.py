"""Run history tracking for project / case management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from reservoir_backend.project.project_registry import json_safe, utc_timestamp


VALID_RUN_STATUSES = {"queued", "running", "completed", "failed", "validated", "cancelled"}


@dataclass(frozen=True)
class RunRecord:
    """One run record for a case."""

    run_id: str
    case_id: str
    started_at: str = field(default_factory=utc_timestamp)
    finished_at: str | None = None
    status: str = "queued"
    report_paths: list[str] = field(default_factory=list)
    result_manifest_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        normalized = validate_run_record(data)
        return cls(
            run_id=normalized["run_id"],
            case_id=normalized["case_id"],
            started_at=normalized["started_at"],
            finished_at=normalized["finished_at"],
            status=normalized["status"],
            report_paths=normalized["report_paths"],
            result_manifest_paths=normalized["result_manifest_paths"],
            metrics=normalized["metrics"],
            warnings=normalized["warnings"],
        )


def validate_run_record(run: RunRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a run record."""
    data = run.to_dict() if isinstance(run, RunRecord) else dict(run)
    required = (
        "run_id",
        "case_id",
        "started_at",
        "finished_at",
        "status",
        "report_paths",
        "result_manifest_paths",
        "metrics",
        "warnings",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"run record missing required keys: {missing}")
    for key in ("run_id", "case_id", "started_at", "status"):
        if not isinstance(data[key], str):
            raise ValueError(f"run record field {key!r} must be a string")
        if not data[key]:
            raise ValueError(f"{key} must be non-empty")
    if data["finished_at"] is not None and not isinstance(data["finished_at"], str):
        raise ValueError("finished_at must be a string or None")
    if data["status"] not in VALID_RUN_STATUSES:
        raise ValueError(f"unsupported run status: {data['status']}")
    for key in ("report_paths", "result_manifest_paths", "warnings"):
        if not isinstance(data[key], list) or any(not isinstance(item, str) for item in data[key]):
            raise ValueError(f"run record field {key!r} must be a list of strings")
    if not isinstance(data["metrics"], dict):
        raise ValueError("run metrics must be a dict")
    return json_safe(data)


class RunHistory:
    """Append-only in-memory run history."""

    def __init__(self, runs: Iterable[RunRecord | Mapping[str, Any]] | None = None):
        self._runs: dict[str, dict[str, Any]] = {}
        for run in runs or []:
            self.append(run)

    def append(self, run: RunRecord | Mapping[str, Any]) -> dict[str, Any]:
        data = validate_run_record(run)
        run_id = data["run_id"]
        if run_id in self._runs:
            raise ValueError(f"duplicate run_id: {run_id}")
        self._runs[run_id] = data
        return data

    def list(self, *, case_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        values = list(self._runs.values())
        if case_id is not None:
            values = [item for item in values if item["case_id"] == case_id]
        if status is not None:
            values = [item for item in values if item["status"] == status]
        return values

    def find(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def validate_report_paths(self, root: str = ".") -> dict[str, Any]:
        from pathlib import Path

        root_path = Path(root)
        missing: list[str] = []
        existing: list[str] = []
        for run in self.list():
            for path in [*run["report_paths"], *run["result_manifest_paths"]]:
                candidate = Path(path)
                resolved = candidate if candidate.is_absolute() else root_path / candidate
                if resolved.exists():
                    existing.append(path)
                else:
                    missing.append(path)
        return {
            "success": not missing,
            "num_runs": len(self._runs),
            "num_existing_paths": len(existing),
            "num_missing_paths": len(missing),
            "existing_paths": existing,
            "missing_paths": missing,
            "warnings": [f"missing run report path: {path}" for path in missing],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"num_runs": len(self._runs), "runs": self.list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunHistory":
        return cls(data.get("runs", []))
