from __future__ import annotations

import json
from pathlib import Path

from reservoir_backend.pipeline.run import main


def test_cli_sensor_case(tmp_path: Path) -> None:
    out = tmp_path / "run"
    code = main(["--config", "config/sensor_case.yaml", "--output", str(out)])
    assert code == 0
    assert (out / "mesh.csv").exists()
    assert (out / "pressure.npy").exists()
    assert (out / "saturation.npz").exists()
    assert (out / "properties.npz").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_cells"] > 0
    assert "INJ" in summary["well_cell_id"]
