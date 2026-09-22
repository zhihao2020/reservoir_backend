"""YAML/CSV contract for the four laboratory reconstruction programs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..exceptions import CaseSchemaError, InvalidObservation
from ..programs.rock import FluidParams, RelpermTable, WellModelParams
from ..programs.similarity import VOLUME_BASES
from .units import MD_TO_M2, to_m3_s, to_metres, to_pa, to_seconds


def _as_mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CaseSchemaError(["expected a mapping"])
    return raw


def _parse_relperm_table(raw: Any) -> RelpermTable | None:
    """Parse an optional ``relperm_table: {sgt: [...], swt: [...]}`` block.

    ``sgt`` rows are ``[Sg, Krg, Krog]`` and ``swt`` rows ``[Sw, Krw, Krow]``
    (CMG ``*SGT`` / ``*SWT``). Returns ``None`` when the block is absent, so the
    forward model falls back to the Corey power-law rel-perm.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CaseSchemaError(["relperm_table must be a mapping with sgt/swt lists"])
    sgt = raw.get("sgt")
    swt = raw.get("swt")
    if sgt is None or swt is None:
        raise CaseSchemaError(["relperm_table needs both sgt and swt lists"])
    return RelpermTable.from_rows(list(sgt), list(swt))


def _vec3(raw: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise CaseSchemaError([f"{name} must be a length-3 list"])
    return (float(raw[0]), float(raw[1]), float(raw[2]))


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WellSpec:
    id: str
    kind: str
    x: float
    y: float
    z: float
    x2: float | None = None
    y2: float | None = None
    z2: float | None = None

    @property
    def trajectory(self) -> tuple[tuple[float, float, float], ...]:
        """Well path heel -> toe; a single point for a point well."""
        if self.x2 is None or self.y2 is None or self.z2 is None:
            return ((self.x, self.y, self.z),)
        return ((self.x, self.y, self.z), (self.x2, self.y2, self.z2))


@dataclass
class LabCase:
    origin: tuple[float, float, float]
    extent: tuple[float, float, float]
    nx: int
    ny: int
    nz: int
    phi0: float
    k0: float
    probes: list[ProbeSpec]
    wells: list[WellSpec]
    times: NDArray[np.float64]
    pressure: NDArray[np.float64]
    sw: NDArray[np.float64]
    so: NDArray[np.float64]
    sg: NDArray[np.float64]
    well_pw: NDArray[np.float64]
    well_q: NDArray[np.float64]
    well_qw: NDArray[np.float64]
    well_qo: NDArray[np.float64]
    well_qg: NDArray[np.float64]
    method: str = "kriging"
    rock_model: str = "total_mobility"
    forward_model: str = "none"
    relperm: tuple[str, ...] = ()
    k_smoothness: float = 2.0
    k_homogeneous: bool = False
    transient: bool = False
    fractional_weight: float = 1.0
    black_oil: FluidParams = field(default_factory=FluidParams)
    well: WellModelParams = field(default_factory=WellModelParams)
    field_length_m: float = 0.30
    field_width_m: float = 0.30
    field_height_m: float = 0.30
    field_flow_path_m: float | None = None
    model_flow_path_m: float | None = None
    volume_basis: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @property
    def probe_ids(self) -> list[str]:
        return [p.id for p in self.probes]

    @property
    def well_ids(self) -> list[str]:
        return [w.id for w in self.wells]

    @property
    def probe_xyz(self) -> NDArray[np.float64]:
        return np.array([[p.x, p.y, p.z] for p in self.probes], dtype=float)

    @property
    def well_xyz(self) -> NDArray[np.float64]:
        return np.array([[w.x, w.y, w.z] for w in self.wells], dtype=float)

    @property
    def well_trajectories(self) -> list[tuple[tuple[float, float, float], ...]]:
        return [w.trajectory for w in self.wells]

    def static_issues(self) -> list[str]:
        issues: list[str] = []
        if min(self.nx, self.ny, self.nz) < 1:
            issues.append("nx, ny, nz must be positive")
        if min(self.extent) <= 0.0:
            issues.append("extent must be positive")
        if self.phi0 <= 0.0 or self.phi0 >= 1.0:
            issues.append("phi0 must be in (0, 1)")
        if self.k0 <= 0.0:
            issues.append("k0 must be positive")
        if not self.probes:
            issues.append("at least one probe is required")
        if not self.wells:
            issues.append("at least one well is required")
        ids = [p.id for p in self.probes] + [w.id for w in self.wells]
        if len(ids) != len(set(ids)):
            issues.append("probe/well ids must be unique")
        if min(self.field_length_m, self.field_width_m, self.field_height_m) <= 0.0:
            issues.append("similarity.field_*_m must be positive")
        if self.field_flow_path_m is not None and self.field_flow_path_m <= 0.0:
            issues.append("similarity.field_flow_path_m must be positive")
        if self.model_flow_path_m is not None and self.model_flow_path_m <= 0.0:
            issues.append("similarity.model_flow_path_m must be positive")
        if self.volume_basis is not None and self.volume_basis not in VOLUME_BASES:
            issues.append(f"similarity.volume_basis must be one of {VOLUME_BASES}")
        if self.well.kv_kh < 0.0:
            issues.append("well.kv_kh must be non-negative")
        return issues

    def series_issues(self) -> list[str]:
        issues: list[str] = []
        if self.times.size == 0:
            issues.append("no observation/control times")
        if not np.isfinite(self.pressure).any():
            issues.append("no finite pressure observations")
        if not (np.isfinite(self.sw).any() or np.isfinite(self.so).any() or np.isfinite(self.sg).any()):
            issues.append("no finite Sw/So/Sg observations")
        return issues

    def issues(self, *, require_series: bool = True) -> list[str]:
        issues = self.static_issues()
        if require_series:
            issues.extend(self.series_issues())
        return issues

    def append_step(
        self,
        time_s: float,
        pressure: NDArray[np.float64],
        sw: NDArray[np.float64],
        so: NDArray[np.float64],
        sg: NDArray[np.float64],
        well_pw: NDArray[np.float64],
        well_q: NDArray[np.float64],
        well_qw: NDArray[np.float64],
        well_qo: NDArray[np.float64],
        well_qg: NDArray[np.float64],
    ) -> None:
        """Append one time, or overwrite the last row when ``time_s`` matches."""
        n_p, n_w = len(self.probes), len(self.wells)
        p = np.asarray(pressure, dtype=float).ravel()
        sw_a = np.asarray(sw, dtype=float).ravel()
        so_a = np.asarray(so, dtype=float).ravel()
        sg_a = np.asarray(sg, dtype=float).ravel()
        pw = np.asarray(well_pw, dtype=float).ravel()
        q = np.asarray(well_q, dtype=float).ravel()
        qw = np.asarray(well_qw, dtype=float).ravel()
        qo = np.asarray(well_qo, dtype=float).ravel()
        qg = np.asarray(well_qg, dtype=float).ravel()
        if p.size != n_p or sw_a.size != n_p or so_a.size != n_p or sg_a.size != n_p:
            raise InvalidObservation("probe arrays must match the init probe count")
        if pw.size != n_w or q.size != n_w or qw.size != n_w or qo.size != n_w or qg.size != n_w:
            raise InvalidObservation("well arrays must match the init well count")
        t = float(time_s)
        if self.times.size == 0:
            self.times = np.array([t], dtype=float)
            self.pressure = p[None, :].copy()
            self.sw = sw_a[None, :].copy()
            self.so = so_a[None, :].copy()
            self.sg = sg_a[None, :].copy()
            self.well_pw = pw[None, :].copy()
            self.well_q = q[None, :].copy()
            self.well_qw = qw[None, :].copy()
            self.well_qo = qo[None, :].copy()
            self.well_qg = qg[None, :].copy()
            return
        last = float(self.times[-1])
        if t < last:
            raise InvalidObservation(f"time {t} is earlier than last {last}")
        if t == last:
            i = int(self.times.size) - 1
            self.pressure[i] = p
            self.sw[i] = sw_a
            self.so[i] = so_a
            self.sg[i] = sg_a
            self.well_pw[i] = pw
            self.well_q[i] = q
            self.well_qw[i] = qw
            self.well_qo[i] = qo
            self.well_qg[i] = qg
            return
        self.times = np.append(self.times, t)
        self.pressure = np.vstack([self.pressure, p])
        self.sw = np.vstack([self.sw, sw_a])
        self.so = np.vstack([self.so, so_a])
        self.sg = np.vstack([self.sg, sg_a])
        self.well_pw = np.vstack([self.well_pw, pw])
        self.well_q = np.vstack([self.well_q, q])
        self.well_qw = np.vstack([self.well_qw, qw])
        self.well_qo = np.vstack([self.well_qo, qo])
        self.well_qg = np.vstack([self.well_qg, qg])

    def trim_to(self, window: int) -> None:
        """Drop the oldest rows so only the most recent ``window`` times remain.

        Used by the online (streaming) path to keep memory and per-step cost
        bounded as the session runs indefinitely. ``window <= 0`` is a no-op.
        """
        if window <= 0 or self.times.size <= window:
            return
        keep = int(self.times.size) - int(window)
        self.times = self.times[keep:]
        self.pressure = self.pressure[keep:]
        self.sw = self.sw[keep:]
        self.so = self.so[keep:]
        self.sg = self.sg[keep:]
        self.well_pw = self.well_pw[keep:]
        self.well_q = self.well_q[keep:]
        self.well_qw = self.well_qw[keep:]
        self.well_qo = self.well_qo[keep:]
        self.well_qg = self.well_qg[keep:]


def load_lab_case(path: str | Path, *, require_series: bool = True) -> LabCase:
    case_path = Path(path)
    raw = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CaseSchemaError(["case YAML must be a mapping"])
    return lab_case_from_mapping(
        raw, base=case_path.parent, source=case_path, require_series=require_series
    )


def lab_case_from_mapping(
    raw: dict[str, Any],
    *,
    base: Path,
    source: Path | None = None,
    require_series: bool = True,
) -> LabCase:
    geom = _as_mapping(raw.get("geometry"))
    rock = _as_mapping(raw.get("rock"))
    interp = _as_mapping(raw.get("interpolation"))
    inv = _as_mapping(raw.get("inversion"))
    fwd = _as_mapping(raw.get("forward"))
    oil_raw = _as_mapping(raw.get("black_oil"))
    well_raw = _as_mapping(raw.get("well"))
    sim_raw = _as_mapping(raw.get("similarity"))
    origin = _vec3(
        geom.get("origin_m", [0.0, 0.0, 0.0]), "geometry.origin_m"
    )
    extent = _vec3(
        geom.get("extent_m", [0.30, 0.30, 0.30]), "geometry.extent_m"
    )
    try:
        nx, ny, nz = int(geom["nx"]), int(geom["ny"]), int(geom["nz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaseSchemaError(
            ["geometry.nx/ny/nz are required integers"]
        ) from exc
    phi0 = float(rock.get("phi0", 0.05))
    if "k0_md" in rock:
        k0 = float(rock["k0_md"]) * MD_TO_M2
    elif "k0" in rock:
        k0 = float(rock["k0"])
    else:
        k0 = 0.03 * MD_TO_M2
    probes = _load_probes(_as_mapping(raw.get("probes")), base)
    wells = _load_wells(_as_mapping(raw.get("wells")), base)
    times, pressure, sw, so, sg = _load_observations(
        _as_mapping(raw.get("probes")).get("observations"),
        probes,
        base,
        required=require_series,
    )
    well_times, well_pw, well_q, well_qw, well_qo, well_qg = _load_well_series(
        _as_mapping(raw.get("wells")).get("series"),
        wells,
        base,
        required=require_series,
    )
    aligned = _align_times(
        times,
        pressure,
        sw,
        so,
        sg,
        well_times,
        well_pw,
        well_q,
        well_qw,
        well_qo,
        well_qg,
    )
    times, pressure, sw, so, sg, well_pw, well_q, well_qw, well_qo, well_qg = aligned
    oil = FluidParams(
        mu_w=float(oil_raw.get("mu_w_pa_s", 5.0e-4)),
        mu_o=float(oil_raw.get("mu_o_pa_s", 2.0e-3)),
        mu_g=float(oil_raw.get("mu_g_pa_s", 2.0e-5)),
        krw_end=float(oil_raw.get("krw_end", 0.3)),
        kro_end=float(oil_raw.get("kro_end", 0.8)),
        krg_end=float(oil_raw.get("krg_end", 0.9)),
        nw=float(oil_raw.get("nw", 2.0)),
        no=float(oil_raw.get("no", 2.0)),
        ng=float(oil_raw.get("ng", 2.0)),
        swc=float(oil_raw.get("Swc", oil_raw.get("swc", 0.05))),
        sor=float(oil_raw.get("Sor", oil_raw.get("sor", 0.15))),
        sgc=float(oil_raw.get("Sgc", oil_raw.get("sgc", 0.02))),
        ct=float(oil_raw.get("ct_1pa", 1.0e-9)),
        rs_slope=float(oil_raw.get("rs_slope", 0.0)),
        rs_eq_slope=float(oil_raw.get("rs_eq_slope", 0.0)),
        bo_slope=float(oil_raw.get("bo_slope", 0.0)),
        rho_w=float(oil_raw.get("rho_w", 0.0)),
        rho_o=float(oil_raw.get("rho_o", 0.0)),
        rho_g=float(oil_raw.get("rho_g", 0.0)),
        rho_s=float(oil_raw.get("rho_s", 0.0)),
        c_sat=float(oil_raw.get("c_sat", 0.66)),
        bg=float(oil_raw.get("bg", 1.0)),
        relperm_table=_parse_relperm_table(oil_raw.get("relperm_table")),
        k_diss=float(oil_raw.get("k_diss", 0.0)),
    )
    well = WellModelParams(
        rw=float(well_raw.get("rw", 0.005)),
        skin=float(well_raw.get("skin", 0.0)),
        kv_kh=float(well_raw.get("kv_kh", 1.0)),
        rho_g=float(well_raw.get("rho_g", 0.0)),
    )
    field_length_m = float(sim_raw.get("field_length_m", 0.30))
    field_width_m = float(sim_raw.get("field_width_m", 0.30))
    field_height_m = float(sim_raw.get("field_height_m", 0.30))
    field_flow_path_m = float(sim_raw.get("field_flow_path_m")) if "field_flow_path_m" in sim_raw else None
    model_flow_path_m = float(sim_raw.get("model_flow_path_m")) if "model_flow_path_m" in sim_raw else None
    _raw_basis = sim_raw.get("volume_basis")
    volume_basis = str(_raw_basis).strip().lower() if _raw_basis else None
    case = LabCase(
        origin=origin,
        extent=extent,
        nx=nx,
        ny=ny,
        nz=nz,
        phi0=phi0,
        k0=k0,
        probes=probes,
        wells=wells,
        times=times,
        pressure=pressure,
        sw=sw,
        so=so,
        sg=sg,
        well_pw=well_pw,
        well_q=well_q,
        well_qw=well_qw,
        well_qo=well_qo,
        well_qg=well_qg,
        method=str(interp.get("method", "kriging")),
        rock_model=str(inv.get("model", "total_mobility")),
        forward_model=str(fwd.get("model", "none")),
        relperm=tuple(str(x) for x in (inv.get("relperm") or [])),
        k_smoothness=float(inv.get("k_smoothness", 2.0)),
        k_homogeneous=bool(inv.get("k_homogeneous", False)),
        transient=bool(inv.get("transient", False)),
        fractional_weight=float(inv.get("fractional_weight", 1.0)),
        black_oil=oil,
        well=well,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        field_height_m=field_height_m,
        field_flow_path_m=field_flow_path_m,
        model_flow_path_m=model_flow_path_m,
        volume_basis=volume_basis,
        meta=dict(_as_mapping(raw.get("experiment"))),
        source=source,
    )
    issues = case.issues(require_series=require_series)
    if issues:
        raise CaseSchemaError(issues)
    return case


def summarize_lab_case(case: LabCase) -> dict[str, Any]:
    return {
        "nx": case.nx,
        "ny": case.ny,
        "nz": case.nz,
        "n_cells": case.nx * case.ny * case.nz,
        "n_probes": len(case.probes),
        "n_wells": len(case.wells),
        "n_times": int(case.times.size),
        "phi0": case.phi0,
        "k0_md": case.k0 / MD_TO_M2,
        "method": case.method,
        "rock_model": case.rock_model,
        "forward_model": case.forward_model,
        "field_length_m": case.field_length_m,
        "field_width_m": case.field_width_m,
        "field_height_m": case.field_height_m,
        "field_flow_path_m": case.field_flow_path_m,
        "model_flow_path_m": case.model_flow_path_m,
        "volume_basis": case.volume_basis,
        "well": {
            "rw": case.well.rw,
            "skin": case.well.skin,
            "kv_kh": case.well.kv_kh,
            "rho_g": case.well.rho_g,
        },
        "injectors": [w.id for w in case.wells if w.kind == "injector"],
        "producers": [w.id for w in case.wells if w.kind == "producer"],
        "experiment": case.meta,
    }


def _resolve(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path


def _load_probes(raw: dict[str, Any], base: Path) -> list[ProbeSpec]:
    items = list(raw.get("items") or [])
    path = _resolve(base, raw.get("file"))
    if path is not None:
        items.extend(_read_table(path))
    if not items:
        raise CaseSchemaError(["probes.items or probes.file is required"])
    probes: list[ProbeSpec] = []
    for row in items:
        name = str(row.get("id") or row.get("name"))
        probes.append(
            ProbeSpec(
                id=name,
                x=_coord(row, "x"),
                y=_coord(row, "y"),
                z=_coord(row, "z"),
            )
        )
    return probes


def _load_wells(raw: dict[str, Any], base: Path) -> list[WellSpec]:
    items = list(raw.get("items") or [])
    path = _resolve(base, raw.get("file"))
    if path is not None:
        items.extend(_read_table(path))
    if not items:
        raise CaseSchemaError(["wells.items or wells.file is required"])
    wells: list[WellSpec] = []
    for row in items:
        kind = str(
            row.get("kind") or row.get("type") or "producer"
        ).strip().lower()
        if kind in {"inj", "injection", "injector"}:
            kind = "injector"
        elif kind in {"prod", "production", "producer"}:
            kind = "producer"
        else:
            raise CaseSchemaError([f"unknown well kind {kind!r}"])
        wells.append(
            WellSpec(
                id=str(row.get("id") or row.get("name")),
                kind=kind,
                x=_coord(row, "x"),
                y=_coord(row, "y"),
                z=_coord(row, "z"),
                x2=_coord_optional(row, "x2"),
                y2=_coord_optional(row, "y2"),
                z2=_coord_optional(row, "z2"),
            )
        )
    return wells


def _coord(row: dict[str, Any], axis: str) -> float:
    key_m = f"{axis}_m"
    if key_m in row:
        return float(row[key_m])
    if axis in row:
        unit = str(row.get("unit") or row.get(f"{axis}_unit") or "m")
        return to_metres(float(row[axis]), unit)
    raise CaseSchemaError([f"missing {axis} coordinate"])


def _coord_optional(row: dict[str, Any], axis: str) -> float | None:
    key_m = f"{axis}_m"
    if key_m in row and row[key_m] not in ("", None):
        return float(row[key_m])
    if axis in row and row.get(axis) not in ("", None):
        unit = str(row.get("unit") or row.get(f"{axis}_unit") or "m")
        return to_metres(float(row[axis]), unit)
    return None


def _empty_probe_series(n_probes: int) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n = int(n_probes)
    empty_t = np.zeros(0, dtype=float)
    empty = np.zeros((0, n), dtype=float)
    return empty_t, empty.copy(), empty.copy(), empty.copy(), empty.copy()


def _empty_well_series(n_wells: int) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n = int(n_wells)
    empty_t = np.zeros(0, dtype=float)
    empty = np.zeros((0, n), dtype=float)
    return empty_t, empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy()


def _load_observations(
    path_raw: Any,
    probes: list[ProbeSpec],
    base: Path,
    *,
    required: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    path = _resolve(base, path_raw)
    if not required:
        # Online (streaming) mode: the observation/series history is populated
        # step-by-step over TCP, so start empty even if a file is referenced.
        return _empty_probe_series(len(probes))
    if path is None:
        raise CaseSchemaError(["probes.observations CSV is required"])
    index = {p.id: n for n, p in enumerate(probes)}
    times: list[float] = []
    records: list[tuple[float, str, str, float]] = []
    for row in _read_table(path):
        t = _time(row)
        name = str(row.get("probe") or row.get("id"))
        quantity = str(
            row.get("quantity") or row.get("kind") or ""
        ).strip().lower()
        value = _value(row, quantity)
        if name not in index:
            raise InvalidObservation(f"unknown probe {name!r} in observations")
        times.append(t)
        records.append((t, name, quantity, value))
    uniq = np.array(sorted(set(times)), dtype=float)
    t_index = {float(t): n for n, t in enumerate(uniq)}
    n_t, n_p = uniq.size, len(probes)
    pressure = np.full((n_t, n_p), np.nan)
    sw = np.full((n_t, n_p), np.nan)
    so = np.full((n_t, n_p), np.nan)
    sg = np.full((n_t, n_p), np.nan)
    for t, name, quantity, value in records:
        i, j = t_index[float(t)], index[name]
        if quantity in {"pressure", "p"}:
            pressure[i, j] = value
        elif quantity in {"sw", "swtest"}:
            sw[i, j] = value
        elif quantity in {"so", "sotest"}:
            so[i, j] = value
        elif quantity in {"sg", "sgtest"}:
            sg[i, j] = value
        else:
            raise InvalidObservation(
                f"unknown observation quantity {quantity!r}"
            )
    return uniq, pressure, sw, so, sg


def _load_well_series(
    path_raw: Any,
    wells: list[WellSpec],
    base: Path,
    *,
    required: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    path = _resolve(base, path_raw)
    if not required:
        return _empty_well_series(len(wells))
    if path is None:
        raise CaseSchemaError(["wells.series CSV is required"])
    index = {w.id: n for n, w in enumerate(wells)}
    times: list[float] = []
    rows = _read_table(path)
    for row in rows:
        times.append(_time(row))
        name = str(row.get("well") or row.get("id"))
        if name not in index:
            raise InvalidObservation(f"unknown well {name!r} in series")
    uniq = np.array(sorted(set(times)), dtype=float)
    t_index = {float(t): n for n, t in enumerate(uniq)}
    n_t, n_w = uniq.size, len(wells)
    pw = np.full((n_t, n_w), np.nan)
    q = np.zeros((n_t, n_w))
    qw = np.zeros((n_t, n_w))
    qo = np.zeros((n_t, n_w))
    qg = np.zeros((n_t, n_w))
    for row in rows:
        i = t_index[float(_time(row))]
        j = index[str(row.get("well") or row.get("id"))]
        if "pw_pa" in row:
            pw[i, j] = float(row["pw_pa"])
        elif "pw" in row:
            pw[i, j] = to_pa(float(row["pw"]), str(row.get("pw_unit") or "Pa"))
        elif "pin_pa" in row:
            pw[i, j] = float(row["pin_pa"])
        elif "pin" in row:
            pw[i, j] = to_pa(
                float(row["pin"]), str(row.get("pin_unit") or "Pa")
            )
        q[i, j] = _rate(row, "q")
        qw[i, j] = _rate(row, "qw")
        qo[i, j] = _rate(row, "qo")
        qg[i, j] = _rate(row, "qg")
        if "pin_pa" in row and not np.isfinite(pw[i, j]):
            pw[i, j] = float(row["pin_pa"])
    return uniq, pw, q, qw, qo, qg


def _align_times(
    obs_t: NDArray[np.float64],
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    well_t: NDArray[np.float64],
    well_pw: NDArray[np.float64],
    well_q: NDArray[np.float64],
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    times = np.array(
        sorted(set(obs_t.tolist()) | set(well_t.tolist())), dtype=float
    )
    pressure = _reindex(obs_t, pressure, times)
    sw = _reindex(obs_t, sw, times)
    so = _reindex(obs_t, so, times)
    sg = _reindex(obs_t, sg, times)
    well_pw = _reindex(well_t, well_pw, times)
    well_q = _reindex(well_t, well_q, times, fill=0.0)
    well_qw = _reindex(well_t, well_qw, times, fill=0.0)
    well_qo = _reindex(well_t, well_qo, times, fill=0.0)
    well_qg = _reindex(well_t, well_qg, times, fill=0.0)
    return times, pressure, sw, so, sg, well_pw, well_q, well_qw, well_qo, well_qg


def _reindex(
    src_t: NDArray[np.float64],
    values: NDArray[np.float64],
    dest_t: NDArray[np.float64],
    *,
    fill: float = np.nan,
) -> NDArray[np.float64]:
    out = np.full((dest_t.size, values.shape[1]), fill, dtype=float)
    lookup = {float(t): n for n, t in enumerate(src_t)}
    for n, t in enumerate(dest_t):
        src = lookup.get(float(t))
        if src is not None:
            out[n] = values[src]
    return out


def _time(row: dict[str, Any]) -> float:
    if "time_s" in row:
        return float(row["time_s"])
    if "time" in row:
        return to_seconds(float(row["time"]), str(row.get("time_unit") or "s"))
    raise InvalidObservation("observation/series row missing time")


def _value(row: dict[str, Any], quantity: str) -> float:
    if "value" in row:
        raw = float(row["value"])
        unit = str(row.get("unit") or "")
        if quantity in {"pressure", "p"} and unit:
            return to_pa(raw, unit)
        return raw
    raise InvalidObservation("observation row missing value")


def _rate(row: dict[str, Any], key: str) -> float:
    if f"{key}_m3s" in row:
        return float(row[f"{key}_m3s"])
    if key in row and row[key] not in ("", None):
        unit = str(row.get(f"{key}_unit") or "m3/s")
        return to_m3_s(float(row[key]), unit)
    return 0.0


def _read_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CaseSchemaError([f"missing file {path}"])
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CaseSchemaError([f"{path} has no header"])
        rows = []
        for raw in reader:
            row = {
                k.strip(): v.strip()
                for k, v in raw.items()
                if k is not None and v is not None
            }
            rows.append(row)
        return rows
