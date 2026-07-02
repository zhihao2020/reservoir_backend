"""Output writer convenience functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reservoir_backend.core.field import Field3D
from reservoir_backend.io.result_manager import ResultManager


def write_field(manager: ResultManager, name: str, field: Field3D) -> Path:
    """Write a field through a `ResultManager`."""
    return manager.save_field(name, field)


def write_report(manager: ResultManager, name: str, report: dict[str, Any]) -> Path:
    """Write a JSON report through a `ResultManager`."""
    return manager.save_json(name, report)
