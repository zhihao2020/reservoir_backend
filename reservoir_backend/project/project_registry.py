"""Project metadata registry.

This module is deliberately file/dict based. It is not a database service and
does not orchestrate solver execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_timestamp() -> str:
    """Return a UTC timestamp string for metadata records."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common Python values into JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ProjectMetadata:
    """Project-level metadata for grouping cases and runs."""

    project_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=utc_timestamp)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectMetadata":
        normalized = validate_project_metadata(data)
        return cls(
            project_id=normalized["project_id"],
            name=normalized["name"],
            description=normalized["description"],
            created_at=normalized["created_at"],
            metadata=normalized["metadata"],
        )


def validate_project_metadata(project: ProjectMetadata | Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize project metadata."""
    data = project.to_dict() if isinstance(project, ProjectMetadata) else dict(project)
    required = ("project_id", "name", "description", "created_at", "metadata")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"project metadata missing required keys: {missing}")
    for key in ("project_id", "name", "description", "created_at"):
        if not isinstance(data[key], str):
            raise ValueError(f"project metadata field {key!r} must be a string")
    if not data["project_id"]:
        raise ValueError("project_id must be non-empty")
    if not data["name"]:
        raise ValueError("project name must be non-empty")
    if not isinstance(data["metadata"], dict):
        raise ValueError("project metadata must be a dict")
    return json_safe(data)


class ProjectRegistry:
    """In-memory registry for project metadata."""

    def __init__(self, projects: Iterable[ProjectMetadata | Mapping[str, Any]] | None = None):
        self._projects: dict[str, dict[str, Any]] = {}
        for project in projects or []:
            self.add(project)

    def add(self, project: ProjectMetadata | Mapping[str, Any]) -> dict[str, Any]:
        data = validate_project_metadata(project)
        project_id = data["project_id"]
        if project_id in self._projects:
            raise ValueError(f"duplicate project_id: {project_id}")
        self._projects[project_id] = data
        return data

    def list(self) -> list[dict[str, Any]]:
        return list(self._projects.values())

    def find(self, project_id: str) -> dict[str, Any] | None:
        return self._projects.get(project_id)

    def to_dict(self) -> dict[str, Any]:
        return {"num_projects": len(self._projects), "projects": self.list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectRegistry":
        return cls(data.get("projects", []))
