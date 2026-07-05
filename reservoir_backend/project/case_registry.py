"""Case metadata registry for project-managed reservoir runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from reservoir_backend.project.project_registry import json_safe


VALID_CASE_STATUSES = {
    "draft",
    "ready",
    "running",
    "completed",
    "failed",
    "validated",
    "archived",
}


@dataclass(frozen=True)
class CaseMetadata:
    """Case-level metadata linking inputs, outputs, modules, and status."""

    case_id: str
    project_id: str
    case_name: str
    input_paths: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    module_tags: list[str] = field(default_factory=list)
    status: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseMetadata":
        normalized = validate_case_metadata(data)
        return cls(
            case_id=normalized["case_id"],
            project_id=normalized["project_id"],
            case_name=normalized["case_name"],
            input_paths=normalized["input_paths"],
            output_paths=normalized["output_paths"],
            module_tags=normalized["module_tags"],
            status=normalized["status"],
            metadata=normalized["metadata"],
        )


def validate_case_metadata(case: CaseMetadata | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize case metadata."""
    data = case.to_dict() if isinstance(case, CaseMetadata) else dict(case)
    required = (
        "case_id",
        "project_id",
        "case_name",
        "input_paths",
        "output_paths",
        "module_tags",
        "status",
        "metadata",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"case metadata missing required keys: {missing}")
    for key in ("case_id", "project_id", "case_name", "status"):
        if not isinstance(data[key], str):
            raise ValueError(f"case metadata field {key!r} must be a string")
        if key != "status" and not data[key]:
            raise ValueError(f"{key} must be non-empty")
    if data["status"] not in VALID_CASE_STATUSES:
        raise ValueError(f"unsupported case status: {data['status']}")
    for key in ("input_paths", "output_paths", "module_tags"):
        if not isinstance(data[key], list) or any(not isinstance(item, str) for item in data[key]):
            raise ValueError(f"case metadata field {key!r} must be a list of strings")
    if not isinstance(data["metadata"], dict):
        raise ValueError("case metadata must be a dict")
    return json_safe(data)


class CaseRegistry:
    """In-memory registry for cases."""

    def __init__(self, cases: Iterable[CaseMetadata | Mapping[str, Any]] | None = None):
        self._cases: dict[str, dict[str, Any]] = {}
        for case in cases or []:
            self.add(case)

    def add(self, case: CaseMetadata | Mapping[str, Any]) -> dict[str, Any]:
        data = validate_case_metadata(case)
        case_id = data["case_id"]
        if case_id in self._cases:
            raise ValueError(f"duplicate case_id: {case_id}")
        self._cases[case_id] = data
        return data

    def list(self, *, project_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        values = list(self._cases.values())
        if project_id is not None:
            values = [item for item in values if item["project_id"] == project_id]
        if status is not None:
            values = [item for item in values if item["status"] == status]
        return values

    def find(self, case_id: str) -> dict[str, Any] | None:
        return self._cases.get(case_id)

    def update_status(self, case_id: str, status: str) -> dict[str, Any]:
        if status not in VALID_CASE_STATUSES:
            raise ValueError(f"unsupported case status: {status}")
        if case_id not in self._cases:
            raise KeyError(case_id)
        self._cases[case_id] = {**self._cases[case_id], "status": status}
        return self._cases[case_id]

    def validate_paths(self, root: str = ".") -> dict[str, Any]:
        from pathlib import Path

        root_path = Path(root)
        missing: list[str] = []
        existing: list[str] = []
        for case in self.list():
            for path in [*case["input_paths"], *case["output_paths"]]:
                candidate = Path(path)
                resolved = candidate if candidate.is_absolute() else root_path / candidate
                if resolved.exists():
                    existing.append(path)
                else:
                    missing.append(path)
        return {
            "success": not missing,
            "num_cases": len(self._cases),
            "num_existing_paths": len(existing),
            "num_missing_paths": len(missing),
            "existing_paths": existing,
            "missing_paths": missing,
            "warnings": [f"missing case path: {path}" for path in missing],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"num_cases": len(self._cases), "cases": self.list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseRegistry":
        return cls(data.get("cases", []))
