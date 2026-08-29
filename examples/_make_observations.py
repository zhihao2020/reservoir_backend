"""Write example observation CSVs from H(F(known structure)). Not a user entry point."""

from __future__ import annotations

import csv
from pathlib import Path

from reservoir_backend.twin.apply import attach_two_layer_demo, write_observation_csv
from reservoir_backend.io.case import load_case

ROOT = Path(__file__).resolve().parent


def write_lab_units(src: Path, dst: Path) -> None:
    with src.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "time_unit", "sensor", "kind", "value", "unit", "sigma", "holdout"])
        for r in rows:
            kind = r["kind"]
            t_min = float(r["time_s"]) / 60.0
            if kind == "pressure":
                w.writerow(
                    [
                        f"{t_min:.12g}",
                        "min",
                        r["sensor"],
                        kind,
                        f"{float(r['value']) / 1000:.8g}",
                        "kPa",
                        f"{float(r['sigma']) / 1000:.6g}",
                        r["holdout"],
                    ]
                )
            else:
                w.writerow(
                    [
                        f"{t_min:.12g}",
                        "min",
                        r["sensor"],
                        kind,
                        r["value"],
                        "",
                        r["sigma"],
                        r["holdout"],
                    ]
                )


def main() -> None:
    cases = (
        (ROOT / "two_layer", ["P_out_top", "S_mid_bot"]),
        (ROOT / "channel", ["Pmx_out", "Sch"]),
    )
    for folder, hold in cases:
        twin = load_case(folder / "case.yaml")
        attach_two_layer_demo(twin, holdout=hold)
        si = folder / "observations.csv"
        write_observation_csv(si, twin)
        write_lab_units(si, folder / "observations_kpa_min.csv")
        n = sum(1 for _ in si.read_text(encoding="utf-8").splitlines()) - 1
        print(f"{folder.name}: {n} rows")


if __name__ == "__main__":
    main()
