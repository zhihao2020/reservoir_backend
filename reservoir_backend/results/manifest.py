from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_MANIFEST_KEYS = (
    "result_id",
    "case_id",
    "run_id",
    "module",
    "result_type",
    "field_name",
    "shape",
    "dtype",
    "unit",
    "path",
    "format",
    "created_at",
    "source_task",
    "source_report",
    "metadata",
    "warnings",
    "limitations",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return value


@dataclass(frozen=True)
class ResultManifest:
    result_id: str
    case_id: str
    run_id: str
    module: str
    result_type: str
    field_name: str
    shape: list[int] | tuple[int, ...]
    dtype: str
    unit: str
    path: str
    format: str
    created_at: str = field(default_factory=utc_timestamp)
    source_task: str = ""
    source_report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["shape"] = [int(item) for item in data["shape"]]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResultManifest":
        validate_result_manifest(data)
        return cls(
            result_id=str(data["result_id"]),
            case_id=str(data["case_id"]),
            run_id=str(data["run_id"]),
            module=str(data["module"]),
            result_type=str(data["result_type"]),
            field_name=str(data["field_name"]),
            shape=[int(item) for item in data["shape"]],
            dtype=str(data["dtype"]),
            unit=str(data["unit"]),
            path=str(data["path"]),
            format=str(data["format"]),
            created_at=str(data["created_at"]),
            source_task=str(data["source_task"]),
            source_report=str(data["source_report"]),
            metadata=dict(data["metadata"]),
            warnings=list(data["warnings"]),
            limitations=list(data["limitations"]),
        )


def validate_result_manifest(manifest: ResultManifest | Mapping[str, Any]) -> dict[str, Any]:
    data = manifest.to_dict() if isinstance(manifest, ResultManifest) else dict(manifest)
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in data]
    if missing:
        raise ValueError(f"result manifest missing required keys: {missing}")

    string_keys = (
        "result_id",
        "case_id",
        "run_id",
        "module",
        "result_type",
        "field_name",
        "dtype",
        "unit",
        "path",
        "format",
        "created_at",
        "source_task",
        "source_report",
    )
    for key in string_keys:
        if not isinstance(data[key], str):
            raise ValueError(f"result manifest field {key!r} must be a string")
    if not data["result_id"]:
        raise ValueError("result manifest result_id must be non-empty")
    if not isinstance(data["shape"], (list, tuple)):
        raise ValueError("result manifest shape must be a list or tuple")
    if any((not isinstance(item, int)) or item < 0 for item in data["shape"]):
        raise ValueError("result manifest shape entries must be non-negative integers")
    if not isinstance(data["metadata"], dict):
        raise ValueError("result manifest metadata must be a dict")
    if not isinstance(data["warnings"], list):
        raise ValueError("result manifest warnings must be a list")
    if not isinstance(data["limitations"], list):
        raise ValueError("result manifest limitations must be a list")
    return _json_safe(data)
