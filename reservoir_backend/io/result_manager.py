"""Result management helpers."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D


class ResultManager:
    """Manage per-case result output directories and files."""

    def __init__(self, results_root: str | Path = "results") -> None:
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.case_dir: Path | None = None
        self.case_id: str | None = None

    def create_case_dir(self, case_id: str) -> Path:
        """Create and select a case output directory."""
        if not case_id:
            raise ValueError("case_id must not be empty")
        case_dir = self.results_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        self.case_dir = case_dir
        self.case_id = case_id
        return case_dir

    def save_npy(self, name: str, array: Any) -> Path:
        """Save an array as `.npy` in the selected case directory."""
        path = self._path(name, ".npy")
        np.save(path, np.asarray(array))
        return path

    def save_field(self, name: str, field: Field3D) -> Path:
        """Save a `Field3D` values array as `.npy`."""
        return self.save_npy(name, field.values)

    def save_json(self, name: str, data: dict[str, Any]) -> Path:
        """Save JSON-serializable data."""
        path = self._path(name, ".json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(data), handle, indent=2)
        return path

    def save_csv(self, name: str, rows_or_dataframe: Any) -> Path:
        """Save rows or a pandas-like dataframe to CSV."""
        path = self._path(name, ".csv")
        if hasattr(rows_or_dataframe, "to_csv"):
            rows_or_dataframe.to_csv(path, index=False)
            return path

        rows = list(rows_or_dataframe)
        with path.open("w", newline="", encoding="utf-8") as handle:
            if not rows:
                return path
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def save_case_summary(self, summary: dict[str, Any]) -> Path:
        """Save `case_summary.json`."""
        return self.save_json("case_summary", summary)

    def list_case_outputs(self, case_id: str) -> list[Path]:
        """List output files for a case."""
        case_dir = self.results_root / case_id
        if not case_dir.exists():
            return []
        return sorted(path for path in case_dir.iterdir() if path.is_file())

    def validate_required_outputs(self, case_id: str, required_files: list[str]) -> bool:
        """Validate that all required case output files exist."""
        case_dir = self.results_root / case_id
        missing = [name for name in required_files if not (case_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing required outputs for {case_id}: {missing}")
        return True

    def _path(self, name: str, suffix: str) -> Path:
        if self.case_dir is None:
            raise RuntimeError("create_case_dir must be called before saving outputs")
        filename = name if name.endswith(suffix) else f"{name}{suffix}"
        path = self.case_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def save_field_npz(field: Field3D, path: str | Path) -> Path:
    """Save a `Field3D` and grid metadata to a compressed NPZ file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": field.name,
        "unit": field.unit,
        "grid": {
            "nx": field.grid.nx,
            "ny": field.grid.ny,
            "nz": field.grid.nz,
            "dx": field.grid.dx,
            "dy": field.grid.dy,
            "dz": field.grid.dz,
        },
        "has_confidence": field.confidence is not None,
    }
    confidence = np.array([], dtype=float) if field.confidence is None else field.confidence
    np.savez_compressed(
        output,
        values=field.values,
        confidence=confidence,
        metadata=json.dumps(metadata),
    )
    return output


def load_field_npz(path: str | Path) -> Field3D:
    """Load a `Field3D` saved by `save_field_npz`."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        grid_meta = metadata["grid"]
        grid = Grid3D(
            nx=int(grid_meta["nx"]),
            ny=int(grid_meta["ny"]),
            nz=int(grid_meta["nz"]),
            dx=float(grid_meta["dx"]),
            dy=float(grid_meta["dy"]),
            dz=float(grid_meta["dz"]),
        )
        confidence = data["confidence"] if metadata["has_confidence"] else None
        return Field3D(
            grid=grid,
            values=data["values"],
            name=metadata["name"],
            unit=metadata["unit"],
            confidence=confidence,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
