"""C_f identifiability and sensor information ranking I_j = (dy/dm)^2 / sigma^2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.lab_v1 import CF_TRUE_M2, load_lab_v1, sensor_information


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true", default=True)
    p.add_argument("--product", action="store_true")
    p.add_argument("--cf-ref", type=float, default=CF_TRUE_M2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.product:
        args.dev = False
    twin = load_lab_v1(dev=bool(args.dev))
    rows = sensor_information(twin, cf_ref=float(args.cf_ref))
    dest = Path(args.out or (ROOT / "results" / "lab_v1" / "sensitivity"))
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "sensor_ranking.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sensor_id",
                "kind",
                "zone",
                "pressure_sensitivity",
                "saturation_sensitivity",
                "information",
                "rank_band",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    (dest / "report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "top": rows[:5], "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
