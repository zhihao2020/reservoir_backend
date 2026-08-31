"""Write 3-D snapshots to NPZ. UDP only returns path + metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class FieldStore:
    def __init__(self, folder: str | Path = "results/fields") -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frame_id = 0

    def write(self, fields: dict[str, Any], *, time_s: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.frame_id += 1
        path = self.folder / f"field_{self.frame_id:06d}.npz"
        packed = {k: np.asarray(v) for k, v in fields.items() if v is not None}
        meta = dict(metadata or {})
        meta["time_s"] = float(time_s)
        meta["frame_id"] = int(self.frame_id)
        np.savez(path, **packed)
        return {
            "frame_id": int(self.frame_id),
            "path": str(path),
            "time_s": float(time_s),
            "pressure_source": str(meta.get("pressure_source", "full")),
            "saturation_source": str(meta.get("saturation_source", "last_full")),
            "saturations_held": bool(meta.get("saturations_held", False)),
            "last_full_time_s": meta.get("last_full_time_s"),
            "Cf_update_time_s": meta.get("Cf_update_time_s"),
        }
