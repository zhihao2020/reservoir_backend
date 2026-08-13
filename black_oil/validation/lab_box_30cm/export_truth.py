"""Export mold-function truth for 15^3 / 30^3 / 50^3 from one source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline.lab_horizon import LabBoxSpec, sample_lab_box, well_xyz  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "truth"


def _pack(painted: dict[str, object], spec: LabBoxSpec) -> dict[str, object]:
    inj, prod = well_xyz(spec)
    zh = np.asarray(painted["z_horizon"], dtype=float)
    return {
        "box_m": [spec.lx, spec.ly, spec.lz],
        "grid": {
            "nx": int(painted["nx"]),
            "ny": int(painted["ny"]),
            "nz": int(painted["nz"]),
            "dx_m": float(painted["dx"]),
            "dy_m": float(painted["dy"]),
            "dz_m": float(painted["dz"]),
        },
        "horizon": {
            "z_min_m": float(zh.min()),
            "z_max_m": float(zh.max()),
            "relief_m": float(zh.max() - zh.min()),
            "note": "Flat lid. Mountain is draped k only; no DTOP, no NULL.",
        },
        "k_m2": {
            "background": float(painted["k_background"]),
            "highk": float(painted["k_high"]),
            "fault": float(painted["k_fault"]),
        },
        "phi": float(np.mean(np.asarray(painted["phi"]))),
        "include_fault": bool(painted["include_fault"]),
        "n_highk": int(np.sum(painted["highk_mask"])),
        "n_fault": int(np.sum(painted["fault_mask"])),
        "wells": {
            "INJ": {"x": inj[0], "y": inj[1], "z": inj[2]},
            "PROD": {"x": prod[0], "y": prod[1], "z": prod[2]},
        },
    }


def export_one(n: int, *, include_fault: bool) -> Path:
    spec = LabBoxSpec()
    painted = sample_lab_box(n, n, n, spec, include_fault=include_fault)
    dest = OUT / f"n{n}"
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "k.npy", np.asarray(painted["k"]))
    np.save(dest / "phi.npy", np.asarray(painted["phi"]))
    np.save(dest / "layer_id.npy", np.asarray(painted["layer_id"]))
    np.save(dest / "highk_mask.npy", np.asarray(painted["highk_mask"]))
    np.save(dest / "fault_mask.npy", np.asarray(painted["fault_mask"]))
    np.save(dest / "z_horizon.npy", np.asarray(painted["z_horizon"]))
    meta = _pack(painted, spec)
    (dest / "truth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # packing guide: horizon elevation at each (x, y) column, metres
    zh = np.asarray(painted["z_horizon"])
    xc = np.asarray(painted["x"])
    yc = np.asarray(painted["y"])
    rows = ["x_m,y_m,z_horizon_m,highk_bottom_m,highk_top_m"]
    half = spec.highk_half_thick
    for j, y in enumerate(yc):
        for i, x in enumerate(xc):
            z0 = float(zh[j, i])
            rows.append(f"{x:.5f},{y:.5f},{z0:.5f},{z0 - half:.5f},{z0 + half:.5f}")
    (dest / "packing_horizon.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dest


def main() -> int:
    written = []
    for n, fault in ((15, False), (30, True), (50, True)):
        dest = export_one(n, include_fault=fault)
        written.append(str(dest))
    print(json.dumps({"exported": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
