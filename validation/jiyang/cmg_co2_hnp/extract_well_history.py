"""Export GEM well rates/BHP into the jiyang_hnp well-history CSV schema.

Does not score field L2. PLACEHOLDER rows stay if .out is missing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "jiyang_co2_hnp.out"
CSV = ROOT / "examples" / "jiyang" / "fixtures" / "jiyang_hnp_well_history.csv"
WELLS = ("INJ", "P1", "P2", "P3", "P4")
HEADER = ["time_s", "sensor", "well", "kind", "value", "sigma", "holdout"]
Q_INJ_IF_STG = 5.0e-5  # SI m3/s, same as YAML / deck *STG


def _placeholder() -> list[dict[str, str]]:
    rows = []
    for well in WELLS:
        kind = "q_inj" if well == "INJ" else "bhp"
        rows.append(
            {
                "time_s": "0",
                "sensor": f"{well}_{kind}",
                "well": well,
                "kind": kind,
                "value": "0.0",
                "sigma": "1.0e-8",
                "holdout": "1",
            }
        )
    return rows


def parse_out(text: str) -> list[dict[str, str]]:
    """Parse GEM FIELD SUMMARY well tables (BHP kPa, oil/gas m3/day)."""
    rows: list[dict[str, str]] = []
    chunks = re.split(r"TIME:\s+([0-9.]+)\s+days", text, flags=re.I)
    for i in range(1, len(chunks) - 1, 2):
        time_s = float(chunks[i]) * 86400.0
        body = chunks[i + 1]
        tables = re.split(r"No\.\s+Name", body)
        for table in tables[1:]:
            names = [n.upper() for n in re.findall(r"\b(?:INJ|P[1-4])\b", table.split("Well Type")[0] if "Well Type" in table else table[:800])]
            names = [n for n in names if n in {"INJ", "P1", "P2", "P3", "P4"}]
            # preserve order, drop duplicates from wrapped headers
            seen: set[str] = set()
            ordered: list[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    ordered.append(n)
            names = ordered
            if not names:
                continue

            def nums(line: str) -> list[float]:
                out: list[float] = []
                for cell in line.split("+")[1:]:
                    m = re.search(r"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[Ee][-+]?\d+)?", cell)
                    if m:
                        out.append(float(m.group(0)))
                return out

            bhp: list[float] = []
            oil: list[float] = []
            gas: list[float] = []
            stg = False
            for line in table.splitlines():
                if "Bottom Hole" in line and "kPa" in line:
                    bhp = [v * 1.0e3 for v in nums(line)]
                if line.strip().startswith("Oil") and "m3/day" in line:
                    oil = [v / 86400.0 for v in nums(line)]
                if line.strip().startswith("Gas") and "m3/day" in line:
                    gas = [v * 1.0e3 / 86400.0 for v in nums(line)]
                if "Status" in line and "STG" in line:
                    stg = True
            for j, well in enumerate(names):
                if j < len(bhp) and bhp[j] > 0.0:
                    rows.append(_row(time_s, well, "bhp", bhp[j], 1.0e5))
                if well != "INJ" and j < len(oil):
                    rows.append(_row(time_s, well, "q_oil", oil[j], 1.0e-8))
                if well != "INJ" and j < len(gas):
                    rows.append(_row(time_s, well, "q_gas", gas[j], 1.0e-8))
                if well == "INJ" and stg:
                    rows.append(_row(time_s, well, "q_inj", Q_INJ_IF_STG, 1.0e-8))
    return rows


def _row(time_s: float, well: str, kind: str, value: float, sigma: float) -> dict[str, str]:
    sensor = f"{well}_bhp" if kind == "bhp" else f"{well}_{kind}"
    return {
        "time_s": f"{time_s:.6g}",
        "sensor": sensor,
        "well": well,
        "kind": kind,
        "value": f"{value:.8g}",
        "sigma": f"{sigma:.6g}",
        "holdout": "0",
    }


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    if OUT.is_file():
        rows = parse_out(OUT.read_text(encoding="latin-1", errors="replace"))
        if not rows:
            rows = _placeholder()
    else:
        rows = _placeholder()
    write_csv(rows, HERE / "well_history.csv")
    print(f"wrote {HERE / 'well_history.csv'} n={len(rows)} from_out={OUT.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
