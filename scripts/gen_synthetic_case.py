#!/usr/bin/env python3
"""Generate the synthetic example cases (small / twod / model_compare / offline).

Deterministic, no GEM needed. Produces the case.yaml + probes.csv / wells.csv /
observations.csv / series.csv that the ``python -m src`` loader reads (see
``src/core/lab_case.py``). ``offline/`` reuses the GEM-derived shailoil CSVs from
``examples/online/`` with a smaller grid; the others use an analytic CO2-flood
toy field so the pipeline runs in seconds.

Usage:
    python scripts/gen_synthetic_case.py            # (re)generate all cases
    python scripts/gen_synthetic_case.py --out examples
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

P0 = 19.0e6  # producer-side pressure, Pa
P1 = 19.3e6  # injector-side pressure, Pa
INJ_RATE = 8.33333333e-8  # CO2 injection, m3/s
OIL_RATE = -3.0e-8  # producer oil rate, m3/s (negative = production)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _probe_csv(path: Path, probes: list[tuple[str, float, float, float]]) -> None:
    _write_csv(path, ["id", "x_m", "y_m", "z_m"], [
        {"id": pid, "x_m": x, "y_m": y, "z_m": z} for pid, x, y, z in probes
    ])


def _well_csv(path: Path, wells: list[dict]) -> None:
    _write_csv(path, ["id", "kind", "x_m", "y_m", "z_m", "x2_m", "y2_m", "z2_m"], wells)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _pressure_at(pt: tuple[float, float, float], inj: tuple[float, float, float], t_idx: int, n_t: int) -> float:
    # Linear ramp between producer-side and injector-side pressure by distance.
    d = _dist(pt, inj)
    frac = max(0.0, min(1.0, 1.0 - d / 0.3))
    ramp = 0.5 + 0.5 * (t_idx / max(1, n_t - 1))
    return P0 + (P1 - P0) * frac * ramp


def _sg_at(pt: tuple[float, float, float], inj: tuple[float, float, float], t_idx: int, n_t: int) -> float:
    # CO2 front advances from the injector over the sequence of time steps.
    d = _dist(pt, inj)
    front = 0.05 + 0.25 * (t_idx / max(1, n_t - 1))
    if d >= front:
        return 0.0
    return round(max(0.0, 0.4 * (1.0 - d / front)), 4)


def _observations_csv(path: Path, probes: list[tuple[str, float, float, float]], inj: tuple[float, float, float], times: list[float]) -> None:
    rows: list[dict] = []
    n_t = len(times)
    for t_idx, t in enumerate(times):
        for pid, x, y, z in probes:
            pt = (x, y, z)
            sg = _sg_at(pt, inj, t_idx, n_t)
            so = round(1.0 - sg, 4)
            sw = 0.0
            p = round(_pressure_at(pt, inj, t_idx, n_t))
            rows += [
                {"time_s": t, "probe": pid, "quantity": "pressure", "value": p, "unit": "Pa"},
                {"time_s": t, "probe": pid, "quantity": "sw", "value": sw, "unit": ""},
                {"time_s": t, "probe": pid, "quantity": "so", "value": so, "unit": ""},
                {"time_s": t, "probe": pid, "quantity": "sg", "value": sg, "unit": ""},
            ]
    _write_csv(path, ["time_s", "probe", "quantity", "value", "unit"], rows)


def _series_csv(path: Path, inj_id: str, prod_ids: list[str], times: list[float], n_prod: int) -> None:
    rows: list[dict] = []
    for t_idx, t in enumerate(times):
        # Injector: constant CO2 injection carried as gas.
        rows.append({
            "time_s": t, "well": inj_id, "pw_pa": P1,
            "q_m3s": INJ_RATE, "qw_m3s": 0.0, "qo_m3s": 0.0, "qg_m3s": INJ_RATE,
        })
        # Producers: oil production, with a growing gas share as the front arrives.
        gas_frac = 0.2 * (t_idx / max(1, len(times) - 1))
        for pid in prod_ids:
            qg = OIL_RATE * gas_frac
            qo = OIL_RATE * (1.0 - gas_frac)
            rows.append({
                "time_s": t, "well": pid, "pw_pa": P0,
                "q_m3s": round(qo + qg, 12), "qw_m3s": 0.0,
                "qo_m3s": round(qo, 12), "qg_m3s": round(qg, 12),
            })
    _write_csv(path, ["time_s", "well", "pw_pa", "q_m3s", "qw_m3s", "qo_m3s", "qg_m3s"], rows)


def _yaml(name: str, nx: int, ny: int, nz: int, model: str) -> str:
    return (
        "geometry:\n"
        "  origin_m: [0.0, 0.0, 0.0]\n"
        "  extent_m: [0.30, 0.30, 0.30]\n"
        f"  nx: {nx}\n"
        f"  ny: {ny}\n"
        f"  nz: {nz}\n"
        "rock:\n"
        "  phi0: 0.05\n"
        "  k0_md: 0.03\n"
        "probes:\n"
        "  file: probes.csv\n"
        "  observations: observations.csv\n"
        "wells:\n"
        "  file: wells.csv\n"
        "  series: series.csv\n"
        "interpolation:\n"
        "  method: auto\n"
        "  rock_method: kriging\n"
        "  power: 2.0\n"
        "inversion:\n"
        f"  model: {model}\n"
        "black_oil:\n"
        "  mu_w_pa_s: 5.0e-4\n"
        "  mu_o_pa_s: 2.0e-3\n"
        "  mu_g_pa_s: 2.0e-5\n"
        "  Swc: 0.0\n"
        "  Sor: 0.0\n"
        "  Sgc: 0.0\n"
        "  ct_1pa: 1.2e-9\n"
        "well:\n"
        "  rw: 0.003\n"
        "  skin: 0.0\n"
        "  kv_kh: 0.2\n"
        "similarity:\n"
        "  field_length_m: 300.0\n"
        "  field_width_m: 300.0\n"
        "  field_height_m: 30.0\n"
        "  field_flow_path_m: 150.0\n"
        "  model_flow_path_m: 0.15\n"
        "  volume_basis: reservoir\n"
        "experiment:\n"
        f"  name: {name}\n"
    )


def _case(name: str, nx: int, ny: int, nz: int, model: str,
          probes: list[tuple[str, float, float, float]],
          wells: list[dict],
          inj_pt: tuple[float, float, float],
          times: list[float]) -> None:
    out = EXAMPLES / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.yaml").write_text(_yaml(name, nx, ny, nz, model), encoding="utf-8")
    _probe_csv(out / "probes.csv", probes)
    _well_csv(out / "wells.csv", wells)
    _observations_csv(out / "observations.csv", probes, inj_pt, times)
    inj_id = next(w["id"] for w in wells if w["kind"] == "injector")
    prod_ids = [w["id"] for w in wells if w["kind"] == "producer"]
    _series_csv(out / "series.csv", inj_id, prod_ids, times, len(prod_ids))


def gen_small() -> None:
    probes = [
        ("P_corner_1", 0.05, 0.05, 0.15),
        ("P_corner_2", 0.25, 0.05, 0.15),
        ("P_corner_3", 0.05, 0.25, 0.15),
        ("P_corner_4", 0.25, 0.25, 0.15),
    ]
    wells = [
        {"id": "INJ", "kind": "injector", "x_m": 0.15, "y_m": 0.15, "z_m": 0.29, "x2_m": 0.15, "y2_m": 0.15, "z2_m": 0.09},
        {"id": "PROD", "kind": "producer", "x_m": 0.25, "y_m": 0.25, "z_m": 0.15, "x2_m": "", "y2_m": "", "z2_m": ""},
    ]
    _case("small", 8, 8, 8, "black_oil_3phase", probes, wells, (0.15, 0.15, 0.15),
          [2592000.0, 5184000.0, 7776000.0])


def gen_twod() -> None:
    probes = [
        ("P_1", 0.05, 0.05, 0.15),
        ("P_2", 0.15, 0.05, 0.15),
        ("P_3", 0.25, 0.05, 0.15),
        ("P_4", 0.05, 0.15, 0.15),
        ("P_5", 0.25, 0.15, 0.15),
        ("P_6", 0.15, 0.25, 0.15),
    ]
    wells = [
        {"id": "INJ", "kind": "injector", "x_m": 0.15, "y_m": 0.15, "z_m": 0.15, "x2_m": "", "y2_m": "", "z2_m": ""},
        {"id": "PROD1", "kind": "producer", "x_m": 0.05, "y_m": 0.05, "z_m": 0.15, "x2_m": "", "y2_m": "", "z2_m": ""},
        {"id": "PROD2", "kind": "producer", "x_m": 0.25, "y_m": 0.25, "z_m": 0.15, "x2_m": "", "y2_m": "", "z2_m": ""},
    ]
    _case("twod", 30, 30, 1, "black_oil_3phase", probes, wells, (0.15, 0.15, 0.15),
          [2592000.0, 5184000.0, 7776000.0, 10368000.0])


def gen_model_compare() -> None:
    probes = [
        ("P_1", 0.05, 0.15, 0.15),
        ("P_2", 0.15, 0.05, 0.15),
        ("P_3", 0.25, 0.15, 0.15),
        ("P_4", 0.15, 0.25, 0.15),
        ("P_5", 0.15, 0.15, 0.15),
    ]
    wells = [
        {"id": "INJ", "kind": "injector", "x_m": 0.15, "y_m": 0.15, "z_m": 0.29, "x2_m": 0.15, "y2_m": 0.15, "z2_m": 0.09},
        {"id": "PROD", "kind": "producer", "x_m": 0.25, "y_m": 0.25, "z_m": 0.15, "x2_m": "", "y2_m": "", "z2_m": ""},
    ]
    out = EXAMPLES / "model_compare"
    out.mkdir(parents=True, exist_ok=True)
    _probe_csv(out / "probes.csv", probes)
    _well_csv(out / "wells.csv", wells)
    _observations_csv(out / "observations.csv", probes, (0.15, 0.15, 0.15),
                      [2592000.0, 5184000.0, 7776000.0])
    _series_csv(out / "series.csv", "INJ", ["PROD"], [2592000.0, 5184000.0, 7776000.0], 1)
    (out / "total_mobility.yaml").write_text(_yaml("model_compare_total_mobility", 10, 10, 10, "total_mobility"), encoding="utf-8")
    (out / "three_phase.yaml").write_text(_yaml("model_compare_three_phase", 10, 10, 10, "black_oil_3phase"), encoding="utf-8")


def gen_offline() -> None:
    out = EXAMPLES / "offline"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("probes.csv", "wells.csv", "observations.csv", "series.csv"):
        src = EXAMPLES / "online" / name
        if src.is_file():
            shutil.copyfile(src, out / name)
    (out / "case.yaml").write_text(_yaml("offline_20cubed_shailoil", 20, 20, 20, "black_oil_3phase"), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXAMPLES, help="examples root")
    args = parser.parse_args(argv)
    gen_small()
    gen_twod()
    gen_model_compare()
    gen_offline()
    print(f"generated cases under {args.out}: small, twod, model_compare, offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
