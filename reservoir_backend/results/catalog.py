from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import ResultManifest, validate_result_manifest


class ResultCatalog:
    """In-memory catalog for result manifests.

    The catalog is deliberately lightweight. It registers paths and metadata but
    never creates or mutates numerical result files.
    """

    def __init__(self, manifests: Iterable[ResultManifest | Mapping[str, Any]] | None = None):
        self._items: dict[str, dict[str, Any]] = {}
        for manifest in manifests or []:
            self.add(manifest)

    def add(self, manifest: ResultManifest | Mapping[str, Any]) -> dict[str, Any]:
        data = validate_result_manifest(manifest)
        result_id = data["result_id"]
        if result_id in self._items:
            raise ValueError(f"duplicate result_id: {result_id}")
        self._items[result_id] = data
        return data

    def list(self) -> list[dict[str, Any]]:
        return list(self._items.values())

    def find(
        self,
        *,
        result_id: str | None = None,
        result_type: str | None = None,
        module: str | None = None,
        field_name: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = self.list()
        if result_id is not None:
            matches = [item for item in matches if item["result_id"] == result_id]
        if result_type is not None:
            matches = [item for item in matches if item["result_type"] == result_type]
        if module is not None:
            matches = [item for item in matches if item["module"] == module]
        if field_name is not None:
            matches = [item for item in matches if item["field_name"] == field_name]
        return matches

    def validate_paths(self, root: str | Path = ".") -> dict[str, Any]:
        root_path = Path(root)
        missing: list[str] = []
        existing: list[str] = []
        for item in self.list():
            path = Path(item["path"])
            resolved = path if path.is_absolute() else root_path / path
            if resolved.exists():
                existing.append(item["path"])
            else:
                missing.append(item["path"])
        warnings = [f"missing result path: {path}" for path in missing]
        return {
            "success": not missing,
            "num_results": len(self._items),
            "num_existing_paths": len(existing),
            "num_missing_paths": len(missing),
            "existing_paths": existing,
            "missing_paths": missing,
            "warnings": warnings,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_results": len(self._items),
            "results": self.list(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResultCatalog":
        return cls(data.get("results", []))
