"""Drive shailoil.dat (or reuse an equivalent GEM .out) into a lab case.

Writes ``results/shailoil_cmg/case/``:
  case.yaml, probes.csv, observations.csv, wells.csv, series.csv, cmg_truth.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

DAY_S = 86400.0
KPA_TO_PA = 1.0e3
MD_TO_M2 = 9.869233e-16
YEAR_DAYS = 360.0
MONTH_DAYS = tuple(float(30 * n) for n in range(1, 13))
NX = NY = NZ = 15
DX = 0.02

_TIME_DAYS = re.compile(r"Time\s*=\s*([0-9.Ee+\-]+)", re.I)
_ALLVAL = re.compile(r"All values are\s+([0-9.Ee+\-]+)", re.I)
_PLANE = re.compile(r"Plane\s+K\s*=\s*(\d+)", re.I)
_IHEAD = re.compile(r"^\s*I\s*=\s*(.+)$", re.I)
_JROW = re.compile(r"J=\s*(\d+)\s+(.*)")
_WELL_SUM = re.compile(
    r"Well Summary at Reservoir Conditions at\s+([0-9.Ee+\-]+)\s+days", re.I
)
_WELL_HEAD = re.compile(
    r"^\s*\d+\s+(\S+)\s+(BHF|BHP)\s+(\d+),(\d+),(\d+)\s+([0-9.Ee+\-]+)\s+"
    r"([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)"
)
_WELL_TOTAL = re.compile(
    r"^\s*Total\s+([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)\s+([0-9.Ee+\-]+)"
)
_PRES_SPEC = re.compile(r"^\s*\*?PRES\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.I)
_WELL_NAME = re.compile(r"^\s*\*?WELL\s+'([^']+)'\s*$", re.I)
_INJ = re.compile(r"^\s*\*?INJECTOR\s+'([^']+)'\s*$", re.I)
_PROD = re.compile(r"^\s*\*?PRODUCER\s+'([^']+)'\s*$", re.I)
_PERF = re.compile(r"^\s*\*?PERF\s+GEO\s+'([^']+)'\s*$", re.I)
_PERF_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+")


def find_gem_exe() -> Path | None:
    env = Path(os.environ.get("CMG_GEM_EXE", ""))
    if env.is_file():
        return env
    for cand in (
        Path(r"D:\Tool\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe"),
        Path(r"D:\Tool\CMG2021\GEM\2021.10\Win_x64\EXE\gm202110.exe"),
    ):
        if cand.is_file():
            return cand
    return None


def gem_to_our_k(k_1based: int, nz: int = NZ) -> int:
    return int(nz) - int(k_1based)


def cell_center(i1: int, j1: int, k1: int) -> tuple[float, float, float]:
    i0, j0, k0 = int(i1) - 1, int(j1) - 1, gem_to_our_k(int(k1))
    return ((i0 + 0.5) * DX, (j0 + 0.5) * DX, (k0 + 0.5) * DX)


def cell_index(i1: int, j1: int, k1: int) -> int:
    i0, j0, k0 = int(i1) - 1, int(j1) - 1, gem_to_our_k(int(k1))
    return k0 * NY * NX + j0 * NX + i0


def _flatten_planes(
    planes: dict[int, dict[int, list[float]]],
    nx: int,
    ny: int,
    nz: int,
) -> NDArray[np.float64]:
    field = np.full(nx * ny * nz, np.nan, dtype=float)
    for k, rows in planes.items():
        k0 = gem_to_our_k(int(k), nz)
        if not 0 <= k0 < nz:
            continue
        for j, vals in rows.items():
            for i, v in enumerate(vals):
                if i >= nx or not np.isfinite(v):
                    continue
                cell = k0 * ny * nx + (int(j) - 1) * nx + i
                if 0 <= cell < field.size:
                    field[cell] = float(v)
    return field


def parse_gem_out_maps(
    out_path: str | Path,
    *,
    nx: int = NX,
    ny: int = NY,
    nz: int = NZ,
    t_max_days: float = YEAR_DAYS,
) -> dict[str, Any]:
    """Parse GEM ASCII grid maps up to ``t_max_days`` (pressure in Pa)."""
    lines = Path(out_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    t_days = 0.0
    maps: dict[tuple[str, str], NDArray[np.float64]] = {}
    kind = None
    current: str | None = None
    planes: dict[int, dict[int, list[float]]] = {}
    kplane = 1
    i_ids: list[int] = []
    snapshots: list[tuple[float, dict[tuple[str, str], NDArray[np.float64]]]] = []

    def flush() -> None:
        nonlocal planes, current, kind
        cont = current or ("bulk" if kind else None)
        if kind and cont and planes:
            maps[(kind, cont)] = _flatten_planes(planes, nx, ny, nz)
        planes = {}

    def stash() -> None:
        flush()
        if maps.get(("pressure", "bulk")) is None and maps.get(("pressure", "fracture")) is None:
            if maps.get(("pressure", "matrix")) is None:
                return
        snapshots.append((float(t_days), {k: np.asarray(v).copy() for k, v in maps.items()}))

    for line in lines:
        tm = _TIME_DAYS.search(line)
        if tm:
            t_new = float(tm.group(1))
            if t_new > t_max_days + 1.0e-9 and t_days >= t_max_days - 1.0e-9:
                stash()
                break
            if t_new > t_days + 1.0e-16:
                stash()
                maps = {}
            else:
                flush()
            planes = {}
            kind = None
            current = None
            i_ids = []
            t_days = t_new
        low = line.strip().lower()
        new_kind = None
        if "matrix pressure - fracture" in low:
            flush()
            kind = None
            current = None
            continue
        if "pressure  ( kpa)" in low:
            new_kind = "pressure"
        elif "oil saturation" in low:
            new_kind = "so"
        elif "gas saturation" in low:
            new_kind = "sg"
        elif "water saturation" in low:
            new_kind = "sw"
        elif "current porosity" in low:
            new_kind = "poros"
        elif "i-direction permeabilities" in low:
            new_kind = "permi"
        if new_kind is not None:
            flush()
            kind = new_kind
            current = None
            kplane = 1
            i_ids = []
            continue
        if "Fundamental Grid - Matrix" in line:
            flush()
            current = "matrix"
        elif "Fundamental Grid - Fracture" in line:
            flush()
            current = "fracture"
        allv = _ALLVAL.search(line)
        if allv and kind:
            val = float(allv.group(1))
            pm_all = _PLANE.search(line)
            if pm_all:
                kplane = int(pm_all.group(1))
                const = [val] * nx
                layer = planes.setdefault(kplane, {})
                for j in range(1, ny + 1):
                    layer[j] = list(const)
            else:
                cont = current or "bulk"
                maps[(kind, cont)] = np.full(nx * ny * nz, val)
            continue
        im = _IHEAD.match(line)
        if im:
            i_ids = [int(float(x)) for x in im.group(1).split()]
            continue
        pm = _PLANE.search(line)
        if pm:
            kplane = int(pm.group(1))
            i_ids = []
        jm = _JROW.match(line.strip())
        if jm and kind is not None:
            j = int(jm.group(1))
            vals = [float(x) for x in jm.group(2).split()]
            ids = i_ids if i_ids else list(range(1, len(vals) + 1))
            row = planes.setdefault(kplane, {}).setdefault(j, [float("nan")] * nx)
            for ii, v in zip(ids, vals):
                if 1 <= int(ii) <= nx:
                    row[int(ii) - 1] = float(v)
    else:
        stash()
    if not snapshots:
        raise ValueError(f"no pressure maps in {out_path}")

    def pick(kind: str, snap: dict[tuple[str, str], NDArray[np.float64]]) -> NDArray[np.float64] | None:
        for cont in ("fracture", "matrix", "bulk"):
            arr = snap.get((kind, cont))
            if arr is not None:
                return np.asarray(arr, dtype=float)
        return None

    times_d = np.array([t for t, _ in snapshots], dtype=float)
    pressure = np.stack([pick("pressure", m) for _, m in snapshots]) * KPA_TO_PA
    def stack(name: str) -> NDArray[np.float64] | None:
        vals = [pick(name, m) for _, m in snapshots]
        if any(v is None for v in vals):
            return None
        return np.stack(vals, axis=0)

    return {
        "times_s": times_d * DAY_S,
        "t_days": times_d,
        "pressure": pressure,
        "sg": stack("sg"),
        "so": stack("so"),
        "sw": stack("sw"),
        "poros": stack("poros"),
        "permi_md": stack("permi"),
        "source": str(out_path),
    }


def parse_gem_well_totals(out_path: str | Path, t_max_days: float = YEAR_DAYS) -> list[dict[str, Any]]:
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    matches = list(_WELL_SUM.finditer(text))
    rows: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        t_days = float(match.group(1))
        if t_days > t_max_days + 1.0e-9:
            break
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.end() : end].split("Cumulative Field Total", 1)[0]
        name = None
        bhp_kpa = None
        for line in block.splitlines():
            head = _WELL_HEAD.match(line)
            if head:
                name = head.group(1)
                bhp_kpa = float(head.group(6))
                continue
            tot = _WELL_TOTAL.match(line)
            if tot and name is not None:
                qo = float(tot.group(1))
                qg = float(tot.group(2))
                qw = float(tot.group(3))
                q = float(tot.group(4))
                rows.append(
                    {
                        "time_s": t_days * DAY_S,
                        "t_days": t_days,
                        "well": name,
                        "pw_pa": (bhp_kpa if bhp_kpa is not None else float("nan")) * KPA_TO_PA,
                        "qo_m3_day": qo,
                        "qg_m3_day": qg,
                        "qw_m3_day": qw,
                        "q_m3_day": q,
                    }
                )
                name = None
                bhp_kpa = None
    return rows


def parse_deck_probes(deck: Path) -> list[tuple[str, int, int, int]]:
    probes: list[tuple[str, int, int, int]] = []
    in_special = False
    for line in deck.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "*OUTSRF" in line.upper() and "*SPECIAL" in line.upper():
            in_special = True
            continue
        if in_special:
            if line.strip().startswith("*") and not line.strip().upper().startswith("*PRES"):
                break
            m = _PRES_SPEC.match(line.replace("*", " *"))
            if m is None:
                m = re.match(r"^\s*\*?PRES\s+(\d+)\s+(\d+)\s+(\d+)", line, re.I)
            if m:
                i, j, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
                probes.append((f"P_{i}_{j}_{k}", i, j, k))
    if not probes:
        raise ValueError(f"no *PRES special points in {deck}")
    return probes


def parse_deck_wells(deck: Path) -> list[dict[str, Any]]:
    kinds: dict[str, str] = {}
    perfs: dict[str, list[tuple[int, int, int]]] = {}
    current_perf: str | None = None
    for line in deck.read_text(encoding="utf-8", errors="ignore").splitlines():
        inj = _INJ.match(line)
        if inj:
            kinds[inj.group(1)] = "injector"
            continue
        prod = _PROD.match(line)
        if prod:
            kinds[prod.group(1)] = "producer"
            continue
        perf = _PERF.match(line)
        if perf:
            current_perf = perf.group(1)
            perfs.setdefault(current_perf, [])
            continue
        if current_perf is not None:
            stripped = line.strip()
            if not stripped or stripped.startswith("**"):
                continue
            if stripped.startswith("*") or re.match(r"^\s*WELL\s+", line, re.I):
                current_perf = None
                continue
            row = _PERF_ROW.match(line)
            if row:
                perfs[current_perf].append((int(row.group(1)), int(row.group(2)), int(row.group(3))))
    wells = []
    for name, cells in perfs.items():
        if not cells:
            continue
        heel, toe = cells[0], cells[-1]
        x, y, z = cell_center(*heel)
        x2, y2, z2 = cell_center(*toe)
        wells.append(
            {
                "id": name,
                "kind": kinds.get(name, "producer"),
                "x": x,
                "y": y,
                "z": z,
                "x2": x2,
                "y2": y2,
                "z2": z2,
                "ijk": cells,
            }
        )
    if not wells:
        raise ValueError(f"no PERF wells in {deck}")
    return wells


def _near(t: float, target: float, atol: float = 1.0) -> bool:
    return abs(float(t) - float(target)) <= atol


def _month_mask(times_s: NDArray[np.float64]) -> NDArray[np.bool_]:
    times = np.asarray(times_s, dtype=float).ravel()
    targets = np.array(MONTH_DAYS, dtype=float) * DAY_S
    return np.array([any(_near(t, m) for m in targets) for t in times], dtype=bool)


def out_has_year1(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if re.search(r"Time\s*=\s*360\b", line):
                return True
    return False


def run_gem(
    deck: Path,
    work: Path,
    *,
    exe: Path | None = None,
    timeout_s: float = 14400.0,
    stop_after_days: float = YEAR_DAYS,
) -> dict[str, Any]:
    binary = exe or find_gem_exe()
    if binary is None:
        return {"ok": False, "blocked": "GEM executable not found"}
    work.mkdir(parents=True, exist_ok=True)
    dest = work / deck.name
    if dest.resolve() != deck.resolve():
        shutil.copy2(deck, dest)
    out_path = work / (dest.stem + ".out")
    proc = subprocess.Popen(
        [str(binary), "-f", dest.name],
        cwd=str(work),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    t0 = time.time()
    stopped_early = False
    next_mark = stop_after_days + 30.0
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                stdout, stderr = proc.communicate()
                return {
                    "ok": rc == 0 and out_path.is_file(),
                    "returncode": int(rc),
                    "exe": str(binary),
                    "workdir": str(work),
                    "stopped_early": stopped_early,
                    "elapsed_s": time.time() - t0,
                    "stdout_tail": (stdout or "")[-2000:],
                    "stderr_tail": (stderr or "")[-2000:],
                    "out_files": [str(out_path)] if out_path.is_file() else [],
                }
            if time.time() - t0 > timeout_s:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return {
                    "ok": out_has_year1(out_path),
                    "returncode": -1,
                    "blocked": "timeout",
                    "exe": str(binary),
                    "workdir": str(work),
                    "stopped_early": True,
                    "elapsed_s": time.time() - t0,
                    "out_files": [str(out_path)] if out_path.is_file() else [],
                }
            if out_path.is_file():
                # Next report time after the year-1 cut means 360-day maps are complete.
                needle = f"Time = {int(next_mark)}"
                try:
                    sample = out_path.read_text(encoding="utf-8", errors="ignore")[-400_000:]
                except OSError:
                    sample = ""
                if needle in sample or f"Time = {next_mark:.1f}" in sample:
                    stopped_early = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    stdout, stderr = "", ""
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except Exception:
                        pass
                    return {
                        "ok": out_has_year1(out_path),
                        "returncode": 0,
                        "exe": str(binary),
                        "workdir": str(work),
                        "stopped_early": True,
                        "elapsed_s": time.time() - t0,
                        "stdout_tail": (stdout or "")[-2000:],
                        "stderr_tail": (stderr or "")[-2000:],
                        "out_files": [str(out_path)],
                    }
            time.sleep(10.0)
    except Exception:
        proc.kill()
        raise


def write_case(
    dest: Path,
    *,
    deck: Path,
    truth: dict[str, Any],
    well_rows: list[dict[str, Any]],
    probes: list[tuple[str, int, int, int]],
    wells: list[dict[str, Any]],
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    times = np.asarray(truth["times_s"], dtype=float)
    mask = _month_mask(times)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("no monthly snapshots in GEM maps")

    with (dest / "probes.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x_m", "y_m", "z_m"])
        for name, i, j, k in probes:
            x, y, z = cell_center(i, j, k)
            w.writerow([name, f"{x:.6g}", f"{y:.6g}", f"{z:.6g}"])

    with (dest / "wells.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "x_m", "y_m", "z_m", "x2_m", "y2_m", "z2_m"])
        for well in wells:
            w.writerow(
                [
                    well["id"],
                    well["kind"],
                    f"{well['x']:.6g}",
                    f"{well['y']:.6g}",
                    f"{well['z']:.6g}",
                    f"{well['x2']:.6g}",
                    f"{well['y2']:.6g}",
                    f"{well['z2']:.6g}",
                ]
            )

    p = np.asarray(truth["pressure"], dtype=float)
    sg = np.zeros_like(p) if truth["sg"] is None else np.asarray(truth["sg"], dtype=float)
    so = np.zeros_like(p) if truth["so"] is None else np.asarray(truth["so"], dtype=float)
    sw = np.zeros_like(p) if truth["sw"] is None else np.asarray(truth["sw"], dtype=float)

    with (dest / "observations.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "probe", "quantity", "value", "unit"])
        for it in idx:
            t = float(times[it])
            for name, i, j, k in probes:
                c = cell_index(i, j, k)
                w.writerow([f"{t:.9g}", name, "pressure", f"{float(p[it, c]):.9g}", "Pa"])
                w.writerow([f"{t:.9g}", name, "sg", f"{float(sg[it, c]):.9g}", ""])
                w.writerow([f"{t:.9g}", name, "so", f"{float(so[it, c]):.9g}", ""])
                w.writerow([f"{t:.9g}", name, "sw", f"{float(sw[it, c]):.9g}", ""])

    by_tw: dict[tuple[int, str], dict[str, Any]] = {}
    for row in well_rows:
        key = (int(round(float(row["t_days"]))), str(row["well"]))
        by_tw[key] = row
    well_ids = [w["id"] for w in wells]
    with (dest / "series.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "well", "pw_pa", "q_m3s", "qw_m3s", "qo_m3s", "qg_m3s"])
        for it in idx:
            t = float(times[it])
            t_days = t / DAY_S
            for well in wells:
                row = by_tw.get((int(round(t_days)), well["id"]))
                if row is None:
                    pw = 20_000_000.0 if well["kind"] == "injector" else 19_000_000.0
                    q = qw = qo = qg = 0.0
                else:
                    pw = float(row["pw_pa"])
                    q = float(row["q_m3_day"]) / DAY_S
                    qw = float(row["qw_m3_day"]) / DAY_S
                    qo = float(row["qo_m3_day"]) / DAY_S
                    qg = float(row["qg_m3_day"]) / DAY_S
                w.writerow(
                    [
                        f"{t:.9g}",
                        well["id"],
                        f"{pw:.9g}",
                        f"{q:.9g}",
                        f"{qw:.9g}",
                        f"{qo:.9g}",
                        f"{qg:.9g}",
                    ]
                )

    poros = truth["poros"]
    permi = truth["permi_md"]
    packed = {
        "times_s": times[idx],
        "pressure": p[idx],
        "sg": sg[idx],
        "so": so[idx],
        "sw": sw[idx],
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "phi0": 0.05,
        "k0_md": 0.03,
    }
    if poros is not None:
        packed["poros"] = np.asarray(poros, dtype=float)[idx]
    if permi is not None:
        packed["permi_m2"] = np.asarray(permi, dtype=float)[idx] * MD_TO_M2
    np.savez_compressed(dest / "cmg_truth.npz", **packed)

    yaml_text = f"""# Packed from GEM shailoil first-year monthly snapshots.
geometry:
  origin_m: [0.0, 0.0, 0.0]
  extent_m: [0.30, 0.30, 0.30]
  nx: {NX}
  ny: {NY}
  nz: {NZ}
rock:
  phi0: 0.05
  k0_md: 0.03
probes:
  file: probes.csv
  observations: observations.csv
wells:
  file: wells.csv
  series: series.csv
interpolation:
  method: auto
  rock_method: kriging
  power: 2.0
inversion:
  model: total_mobility
black_oil:
  ct_1pa: 1.2e-9
  Swc: 0.0
  Sor: 0.0
  Sgc: 0.0
well:
  rw: 0.003
  skin: 0.0
  kv_kh: 0.2
similarity:
  field_length_m: 0.30
  field_width_m: 0.30
  field_height_m: 0.30
experiment:
  name: shailoil_cmg_year1
  source_deck: {deck.as_posix()}
  source_out: {truth["source"]}
  n_probes: {len(probes)}
  n_wells: {len(wells)}
  well_ids: {well_ids}
"""
    (dest / "case.yaml").write_text(yaml_text, encoding="utf-8")
    return dest / "case.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deck", type=Path, default=ROOT / "examples" / "shailoil.dat")
    p.add_argument("--work", type=Path, default=ROOT / "results" / "shailoil_cmg")
    p.add_argument("--case-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="existing GEM .out; skip launch if set")
    p.add_argument("--run", action="store_true", help="force a new GEM run even if an .out exists")
    p.add_argument("--timeout", type=float, default=14400.0)
    args = p.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    case_dir = Path(args.case_dir) if args.case_dir else work / "case"
    deck = Path(args.deck)
    prior = ROOT / "results" / "lab_v1" / "cmg_gem_physical_3d" / "sanwei_co2.out"
    rec: dict[str, Any] = {"deck": str(deck)}

    out_path = args.out
    if out_path is None and not args.run:
        if out_has_year1(prior):
            out_path = prior
            rec["reused_out"] = str(prior)
            rec["note"] = (
                "shailoil.dat matches sanwei_co2.dat after stripping RESULTS SPEC; "
                "reusing the existing GEM year-1+ .out"
            )
        elif out_has_year1(work / "shailoil.out"):
            out_path = work / "shailoil.out"
            rec["reused_out"] = str(out_path)

    if args.run or out_path is None:
        gem = run_gem(deck, work, timeout_s=float(args.timeout))
        rec.update(gem)
        outs = gem.get("out_files") or []
        if not gem.get("ok") or not outs:
            (work / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(json.dumps(rec, indent=2), flush=True)
            return 2 if gem.get("blocked") else 1
        out_path = Path(outs[0])

    rec["out"] = str(out_path)
    print(f"parsing {out_path}", flush=True)
    truth = parse_gem_out_maps(out_path, t_max_days=YEAR_DAYS)
    well_rows = parse_gem_well_totals(out_path, t_max_days=YEAR_DAYS)
    probes = parse_deck_probes(deck)
    wells = parse_deck_wells(deck)
    case_path = write_case(
        case_dir, deck=deck, truth=truth, well_rows=well_rows, probes=probes, wells=wells
    )
    rec.update(
        {
            "ok": True,
            "case": str(case_path),
            "n_map_times": int(np.asarray(truth["times_s"]).size),
            "n_month_times": 12,
            "n_probes": len(probes),
            "n_wells": len(wells),
            "n_well_rows": len(well_rows),
            "t_days": [float(x) for x in truth["t_days"]],
        }
    )
    (work / "run.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in {"stdout_tail", "stderr_tail"}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
