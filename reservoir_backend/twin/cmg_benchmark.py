"""CMG-GEM cross-simulator benchmark. Hidden 3-D truth is scoring-only.

Iron law: inversion never receives CMG full-field arrays. Only Q_inj, P_prod,
P_obs, S_obs — the same streams a laboratory would provide.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from numpy.typing import NDArray

from reservoir_backend.comp.wells import surface_gas_rate_to_mol
from reservoir_backend.domain.types import ControlSeries, ObservationSeries
from reservoir_backend.io.case import _read_control_csv, _read_observation_csv
from reservoir_backend.twin.lab_v1 import CF_TRUE_M2, TMF_TRUE, load_lab_v1
from reservoir_backend.twin.offline import DigitalTwin, Posterior


SPEC_PATH = Path(__file__).resolve().parents[2] / "examples" / "lab_v1" / "cmg_gem" / "spec.yaml"
MD_M2 = 9.869233e-16
KPI_ORDER = (
    "pressure_field_nrmse",
    "sg_field_rmse",
    "so_field_rmse",
    "sw_field_rmse",
    "well_curve_rmse",
    "holdout_sensor_rmse",
    "component_field_rmse",
    "cf_rel_error",
    "tmf_rel_error",
)
# Provisional until the first real GEM export exists. Not a product PASS.
PROVISIONAL_NRMSE_P = 0.10
PROVISIONAL_RMSE_SG = 0.05
# When GEM (Pmax−Pmin) is ASCII print noise (~0.1 kPa), NRMSE explodes.
# Floor at instrument σ_P = 2 kPa (M1c). Raw plan formula stays unfloored.
PRESSURE_SPAN_FLOOR_PA = 2.0e3


@dataclass
class HiddenTruth:
    """CMG 3-D snapshots. Scoring only — never pass to ES-MDA."""

    times_s: NDArray[np.float64]
    pressure: NDArray[np.float64]
    sg: NDArray[np.float64] | None = None
    so: NDArray[np.float64] | None = None
    sw: NDArray[np.float64] | None = None
    z: NDArray[np.float64] | None = None
    pressure_fracture: NDArray[np.float64] | None = None
    pressure_matrix: NDArray[np.float64] | None = None
    p_inj: NDArray[np.float64] | None = None
    q_prod: NDArray[np.float64] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.pressure.shape[1])


def load_alignment_spec(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else SPEC_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def check_alignment(spec: dict[str, Any] | None = None, twin: DigitalTwin | None = None) -> dict[str, Any]:
    """Compare spec.yaml to case_dev. Does not run GEM or our forward."""
    spec = spec if spec is not None else load_alignment_spec()
    twin = twin if twin is not None else load_lab_v1(dev=True)
    rock = spec["rock"]
    geom = [float(x) for x in spec["geometry_m"]]
    grid = [int(x) for x in spec["grid"]]
    mismatches: list[str] = []
    size = [float(x) for x in twin.grid.size_m()]
    if any(abs(a - b) > 1.0e-12 for a, b in zip(size, geom)):
        mismatches.append("geometry")
    if (twin.grid.nx, twin.grid.ny, twin.grid.nz) != tuple(grid):
        mismatches.append("grid")
    if abs(float(twin.physics.phi_fracture) - float(rock["phi_fracture"])) > 1.0e-12:
        mismatches.append("phi_fracture")
    if abs(float(twin.physics.k_matrix_m2) - float(rock["k_matrix_m2"])) > 1.0e-18:
        mismatches.append("k_matrix")
    if abs(float(twin.physics.shape_factor) - float(rock["shape_factor_m2"])) > 1.0e-12:
        mismatches.append("shape_factor")
    if abs(float(twin.physics.p_init) - 1.0e3 * float(spec["physics"]["p_init_kpa"])) > 1.0e-3:
        mismatches.append("p_init")
    if abs(float(twin.physics.temperature_k) - float(spec["fluid"]["temperature_k"])) > 1.0e-9:
        mismatches.append("temperature")
    cf_md = float(rock["cf_m2"]) / MD_M2
    if abs(cf_md - float(rock["cf_md"])) > 0.05:
        mismatches.append("cf_md_conversion")
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "n_cells": int(twin.grid.n_cells),
        "theta_true": dict(spec["theta_true"]),
    }


def nrmse_range(
    pred: NDArray[np.float64],
    truth: NDArray[np.float64],
    *,
    span_floor: float = 0.0,
) -> float:
    """Plan pressure NRMSE: RMS(diff) / (max_truth − min_truth).

    ``span_floor`` is optional (Pa). Use ``PRESSURE_SPAN_FLOOR_PA`` when the
    GEM map span is print noise. Default 0 keeps the written plan formula.
    """
    a = np.asarray(pred, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"field size {a.size} != {b.size}")
    span = float(np.max(b) - np.min(b))
    span = max(span, float(span_floor))
    if span <= 1.0e-18:
        span = max(float(np.linalg.norm(b)), 1.0e-18)
    return float(np.sqrt(np.mean((a - b) ** 2)) / span)


def rmse(pred: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    a = np.asarray(pred, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"field size {a.size} != {b.size}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def improvement(err_prior: float, err_post: float) -> float:
    """1 - RMSE_post / RMSE_prior. Positive means the invert moved toward CMG."""
    den = max(float(err_prior), 1.0e-18)
    return float(1.0 - float(err_post) / den)


def load_observation_dataset(export_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Public invert inputs. Does not open hidden/."""
    root = Path(export_dir)
    obs_path = root / "observations.csv"
    ctrl_path = root / "controls.csv"
    if not obs_path.is_file():
        raise FileNotFoundError(f"missing invert input {obs_path}")
    observations = _read_observation_csv(obs_path)
    controls = _read_control_csv(ctrl_path) if ctrl_path.is_file() else []
    return observations, controls


def _load_npy(folder: Path, name: str) -> NDArray[np.float64] | None:
    path = folder / name
    if not path.is_file():
        return None
    return np.asarray(np.load(path), dtype=float)


def _stack_time_slices(folder: Path, prefix: str) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Plan layout: pressure_t000.npy, sg_t000.npy, ... plus optional times_s in meta."""
    files = sorted(folder.glob(f"{prefix}_t*.npy"))
    if not files:
        return None
    slices = [np.asarray(np.load(p), dtype=float).ravel() for p in files]
    arr = np.stack(slices, axis=0)
    times = np.arange(arr.shape[0], dtype=float)
    return times, arr


def load_hidden_truth(hidden_dir: str | Path) -> HiddenTruth:
    """Scoring-only loader. Invert paths must not call this.

    Accepts stacked ``pressure.npy`` + ``meta.json``, plan-style
    ``pressure_t000.npy``, or a unified ``truth.npz``.
    """
    folder = Path(hidden_dir)
    npz_path = folder / "truth.npz" if folder.is_dir() else folder
    if npz_path.is_file() and npz_path.suffix.lower() == ".npz":
        blob = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(blob["meta"][0])) if "meta" in blob.files else {}
        return HiddenTruth(
            times_s=np.asarray(blob["times_s"], dtype=float),
            pressure=np.asarray(blob["pressure"], dtype=float),
            sg=None if "sg" not in blob.files else np.asarray(blob["sg"], dtype=float),
            so=None if "so" not in blob.files else np.asarray(blob["so"], dtype=float),
            sw=None if "sw" not in blob.files else np.asarray(blob["sw"], dtype=float),
            z=None if "z" not in blob.files else np.asarray(blob["z"], dtype=float),
            pressure_fracture=None if "pressure_fracture" not in blob.files else np.asarray(blob["pressure_fracture"], dtype=float),
            pressure_matrix=None if "pressure_matrix" not in blob.files else np.asarray(blob["pressure_matrix"], dtype=float),
            p_inj=None if "p_inj" not in blob.files else np.asarray(blob["p_inj"], dtype=float),
            q_prod=None if "q_prod" not in blob.files else np.asarray(blob["q_prod"], dtype=float),
            meta=meta,
        )
    if not folder.is_dir():
        raise FileNotFoundError(f"hidden truth not found: {hidden_dir}")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    pressure = _load_npy(folder, "pressure.npy")
    times = None
    if pressure is None:
        stacked = _stack_time_slices(folder, "pressure")
        if stacked is None:
            raise FileNotFoundError(f"missing pressure.npy or pressure_t*.npy in {folder}")
        times, pressure = stacked
    if times is None:
        times = np.asarray(meta.get("times_s", np.arange(pressure.shape[0], dtype=float)), dtype=float)
    if times.size != pressure.shape[0]:
        raise ValueError("hidden times_s length must match pressure axis 0")
    sg = _load_npy(folder, "sg.npy")
    if sg is None:
        sg_stack = _stack_time_slices(folder, "sg")
        sg = None if sg_stack is None else sg_stack[1]
    so = _load_npy(folder, "so.npy")
    if so is None:
        so_stack = _stack_time_slices(folder, "so")
        so = None if so_stack is None else so_stack[1]
    return HiddenTruth(
        times_s=times,
        pressure=pressure,
        sg=sg,
        so=so,
        sw=_load_npy(folder, "sw.npy"),
        z=_load_npy(folder, "z.npy"),
        pressure_fracture=_load_npy(folder, "pressure_fracture.npy"),
        pressure_matrix=_load_npy(folder, "pressure_matrix.npy"),
        p_inj=_load_npy(folder, "p_inj.npy"),
        q_prod=_load_npy(folder, "q_prod.npy"),
        meta=meta,
    )


def attach_cmg_observations(
    twin: DigitalTwin,
    export_dir: str | Path,
    *,
    hidden_dir: str | Path | None = None,
) -> DigitalTwin:
    """Replace twin observations/controls with GEM export. Hidden truth is forbidden."""
    if hidden_dir is not None:
        raise ValueError("inversion must not receive CMG hidden truth")
    obs_rows, ctrl_rows = load_observation_dataset(export_dir)
    observations: list[ObservationSeries] = []
    for o in obs_rows:
        observations.append(
            ObservationSeries(
                sensor_name=str(o["sensor"]),
                kind=str(o.get("kind", "pressure")),
                times_s=np.asarray(o["times"], dtype=float),
                values=np.asarray(o["values"], dtype=float),
                sigma=np.asarray(o.get("sigma", 1.0), dtype=float),
                holdout=bool(o.get("holdout", False)),
            )
        )
    known = {s.name for s in twin.experiment.sensors}
    unknown = sorted({o.sensor_name for o in observations} - known)
    if unknown:
        raise ValueError(f"CMG observations name sensors not in case: {unknown}")
    controls: list[ControlSeries] = []
    for c in ctrl_rows:
        controls.append(
            ControlSeries(
                str(c["port"]),
                str(c["kind"]),
                np.asarray(c["times"], dtype=float),
                np.asarray(c["values"], dtype=float),
            )
        )
    twin.experiment.observations = observations
    if controls:
        twin.experiment.controls = controls
    ensure_molar_injector_rate(twin)
    return twin


def ensure_molar_injector_rate(
    twin: DigitalTwin,
    spec: dict[str, Any] | None = None,
) -> float:
    """YAML ``q_inj_m3_s`` is GEM ``*STG``. Convert to mol/s once.

    ``well_molar_sources`` treats rate as mol/s (compositional API). The
    M2 contract is surface gas m³/s matching ``*STG``. Does not change
    ``case_dev.yaml`` / M1 controls unless they still hold the m³/s number.
    """
    spec = spec if spec is not None else load_alignment_spec()
    q_m3 = float(spec["physics"]["q_inj_m3_s"])
    fluid = twin.physics.fluid
    if fluid is None:
        return q_m3
    q_mol = surface_gas_rate_to_mol(q_m3, fluid)
    new_ctrls: list[ControlSeries] = []
    for c in twin.experiment.controls:
        if str(c.port_name).upper() == "INJ" and c.kind == "rate":
            vals = np.asarray(c.values, dtype=float)
            if np.allclose(vals, q_m3, rtol=1.0e-6, atol=1.0e-12):
                c = ControlSeries(c.port_name, c.kind, c.times_s, np.full_like(vals, q_mol))
        new_ctrls.append(c)
    twin.experiment.controls = new_ctrls
    return q_mol


def invert_from_cmg_observations(
    export_dir: str | Path,
    *,
    hidden_dir: str | Path | None = None,
    twin: DigitalTwin | None = None,
) -> Posterior:
    """ES-MDA on GEM gauges only. Passing hidden_dir is a hard error."""
    if hidden_dir is not None:
        raise ValueError("inversion must not receive CMG hidden truth")
    from reservoir_backend.twin.history_match import HistoryMatchWorkflow

    twin = twin if twin is not None else load_lab_v1(dev=True)
    attach_cmg_observations(twin, export_dir)
    if not twin.experiment.observations:
        raise ValueError("no CMG observations to assimilate")
    return HistoryMatchWorkflow().run(twin)


def pack_visual_fields(traj, times: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Pack F_ours snapshots on the visual (bulk) continuum."""
    t = np.asarray(times, dtype=float).ravel()
    p, sg, so, sw = [], [], [], []
    z_rows: list[NDArray[np.float64]] = []
    have_z = False
    for tf in t:
        st = traj.state_at(float(tf))
        p.append(np.asarray(st.pressure, dtype=float).ravel())
        sg_v = np.zeros_like(st.sw) if st.sg is None else np.asarray(st.sg, dtype=float).ravel()
        sw_v = np.asarray(st.sw, dtype=float).ravel()
        sg.append(sg_v)
        sw.append(sw_v)
        so.append(1.0 - sw_v - sg_v)
        if st.moles is not None:
            moles = np.asarray(st.moles, dtype=float)
            tot = np.maximum(moles.sum(axis=1, keepdims=True), 1.0e-30)
            z_rows.append(moles / tot)
            have_z = True
    out: dict[str, NDArray[np.float64]] = {
        "pressure": np.stack(p, axis=0),
        "sg": np.stack(sg, axis=0),
        "so": np.stack(so, axis=0),
        "sw": np.stack(sw, axis=0),
    }
    if have_z:
        out["z"] = np.stack(z_rows, axis=0)
    p_inj, q_prod, p_mat = [], [], []
    have_mat = False
    for tf in t:
        st = traj.state_at(float(tf))
        rates, bhp = traj.rates_and_bhp_at(float(tf))
        p_inj.append(float(bhp.get("INJ", np.nan)))
        q_prod.append(float(rates.get("PROD", np.nan)))
        if st.pressure_matrix is not None:
            p_mat.append(np.asarray(st.pressure_matrix, dtype=float).ravel())
            have_mat = True
    out["p_inj"] = np.asarray(p_inj, dtype=float)
    out["q_prod"] = np.asarray(q_prod, dtype=float)
    if have_mat:
        out["pressure_matrix"] = np.stack(p_mat, axis=0)
    if traj.reports:
        last = traj.reports[-1].mass
        rel = getattr(last, "relative_balance_error", None)
        if rel is not None:
            out["mass_relative_error"] = np.asarray([float(rel)], dtype=float)
    return out


def compare_fields(
    ours: Mapping[str, NDArray[np.float64]],
    truth: HiddenTruth,
) -> dict[str, float]:
    rec: dict[str, float] = {
        "pressure_field_rmse": rmse(ours["pressure"], truth.pressure),
        "pressure_field_nrmse": nrmse_range(ours["pressure"], truth.pressure),
        "pressure_field_nrmse_sigma": nrmse_range(
            ours["pressure"], truth.pressure, span_floor=PRESSURE_SPAN_FLOOR_PA
        ),
    }
    if truth.sg is not None and "sg" in ours:
        rec["sg_field_rmse"] = rmse(ours["sg"], truth.sg)
    if truth.so is not None and "so" in ours:
        rec["so_field_rmse"] = rmse(ours["so"], truth.so)
    if truth.sw is not None and "sw" in ours:
        rec["sw_field_rmse"] = rmse(ours["sw"], truth.sw)
    if truth.z is not None and "z" in ours:
        rec["component_field_rmse"] = rmse(ours["z"], truth.z)
    if truth.p_inj is not None and "p_inj" in ours:
        rec["well_curve_rmse"] = rmse(ours["p_inj"], truth.p_inj)
    elif truth.q_prod is not None and "q_prod" in ours:
        rec["well_curve_rmse"] = rmse(ours["q_prod"], truth.q_prod)
    if "mass_relative_error" in ours:
        rec["mass_relative_error"] = float(np.asarray(ours["mass_relative_error"]).ravel()[0])
    return rec


def forward_at_theta(
    twin: DigitalTwin,
    theta: NDArray[np.float64],
    times: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    ensure_molar_injector_rate(twin)
    t = np.asarray(times, dtype=float).ravel()
    t_end = float(np.max(t)) if t.size else float(twin.experiment.history_end_s or 60.0)
    traj = twin.simulate(parameters=np.asarray(theta, dtype=float), t_end=t_end, report_times=t)
    return pack_visual_fields(traj, t)


def theta_true_from_spec(twin: DigitalTwin, spec: dict[str, Any] | None = None) -> NDArray[np.float64]:
    spec = spec if spec is not None else load_alignment_spec()
    th = spec["theta_true"]
    cf = float(th.get("cf_m2", CF_TRUE_M2))
    tmf = float(th.get("tmf_multiplier", TMF_TRUE))
    n = int(twin.parameterization.n_params)
    phys = np.array([cf, tmf], dtype=float)[:n]
    return twin.parameterization.encode(phys)


def forward_equivalence_report(
    ours: Mapping[str, NDArray[np.float64]],
    truth: HiddenTruth,
) -> dict[str, Any]:
    metrics = compare_fields(ours, truth)
    nrmse_p = float(metrics.get("pressure_field_nrmse_sigma", metrics.get("pressure_field_nrmse", float("inf"))))
    rmse_sg = float(metrics.get("sg_field_rmse", 0.0))
    return {
        "gate": "m2a_forward_equivalence",
        "metrics": metrics,
        "kpi_order": list(KPI_ORDER),
        "provisional_nrmse_p": PROVISIONAL_NRMSE_P,
        "provisional_rmse_sg": PROVISIONAL_RMSE_SG,
        "span_floor_pa": PRESSURE_SPAN_FLOOR_PA,
        "pass": bool(nrmse_p < PROVISIONAL_NRMSE_P and (truth.sg is None or rmse_sg < PROVISIONAL_RMSE_SG)),
    }


def reconstruction_report(
    *,
    prior: Mapping[str, NDArray[np.float64]],
    posterior: Mapping[str, NDArray[np.float64]],
    truth: HiddenTruth,
    phys_prior: dict[str, float],
    phys_post: dict[str, float],
    phys_true: dict[str, float],
    holdout_rmse: float | None = None,
) -> dict[str, Any]:
    prior_m = compare_fields(prior, truth)
    post_m = compare_fields(posterior, truth)
    p_prior = float(prior_m["pressure_field_nrmse"])
    p_post = float(post_m["pressure_field_nrmse"])
    cf_true = float(phys_true["cf_m2"])
    tmf_true = float(phys_true["tmf_multiplier"])
    param = {
        "cf_rel_error": abs(float(phys_post["cf_m2"]) - cf_true) / max(abs(cf_true), 1.0e-30),
        "tmf_rel_error": abs(float(phys_post["tmf_multiplier"]) - tmf_true) / max(abs(tmf_true), 1.0e-30),
    }
    if holdout_rmse is not None:
        post_m = dict(post_m)
        post_m["holdout_sensor_rmse"] = float(holdout_rmse)
    return {
        "gate": "m2c_hidden_field_reconstruction",
        "kpi_order": list(KPI_ORDER),
        "prior": prior_m,
        "posterior": post_m,
        "improvement_pressure": improvement(p_prior, p_post),
        "parameters": param,
        "phys_prior": phys_prior,
        "phys_post": phys_post,
        "phys_true": phys_true,
        "note": "field error outranks parameter error; Cf_ours != Cf_CMG is not by itself a fail",
    }


def export_blocked_reason(export_dir: str | Path) -> str | None:
    root = Path(export_dir)
    if not (root / "observations.csv").is_file():
        return "missing observations.csv — run GEM and export virtual gauges"
    hidden = root / "hidden"
    has_stack = (hidden / "pressure.npy").is_file() or (hidden / "truth.npz").is_file()
    has_slices = bool(list(hidden.glob("pressure_t*.npy"))) if hidden.is_dir() else False
    if not has_stack and not has_slices:
        return "missing hidden/ 3-D truth — scoring blocked, invert may still run"
    return None


def write_grid_csv(twin: DigitalTwin, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = twin.grid.cell_centers()
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("cell,i,j,k,x_m,y_m,z_m\n")
        for c, (x, y, z) in enumerate(xyz):
            i, j, k = twin.grid.ijk(c)
            fh.write(f"{c},{i},{j},{k},{x:.8g},{y:.8g},{z:.8g}\n")


def write_hidden_truth(folder: str | Path, truth: HiddenTruth) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(truth.meta)
    meta["times_s"] = [float(t) for t in np.asarray(truth.times_s, dtype=float)]
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(folder / "pressure.npy", np.asarray(truth.pressure, dtype=float))
    for name in ("sg", "so", "sw", "z", "pressure_fracture", "pressure_matrix", "p_inj", "q_prod"):
        arr = getattr(truth, name)
        if arr is not None:
            np.save(folder / f"{name}.npy", np.asarray(arr, dtype=float))


def sample_observations_from_hidden(
    twin: DigitalTwin,
    truth: HiddenTruth,
    *,
    holdout: set[str] | None = None,
) -> list[ObservationSeries]:
    """H(F_CMG): virtual gauges on hidden snapshots. Export-prep only, not invert."""
    from reservoir_backend.domain.types import State

    hold = holdout or set()
    series: list[ObservationSeries] = []
    times = np.asarray(truth.times_s, dtype=float)
    for sen in twin.experiment.sensors:
        vals = []
        for it, tf in enumerate(times):
            p = np.asarray(truth.pressure[it], dtype=float).ravel()
            sg = None if truth.sg is None else np.asarray(truth.sg[it], dtype=float).ravel()
            sw = np.zeros_like(p) if truth.sw is None else np.asarray(truth.sw[it], dtype=float).ravel()
            st = State(
                pressure=p,
                sw=sw,
                sg=sg,
                time_s=float(tf),
                pressure_matrix=None if truth.pressure_matrix is None else np.asarray(truth.pressure_matrix[it], dtype=float).ravel(),
            )
            vals.append(float(twin.operator.sample(sen, st)))
        series.append(
            ObservationSeries(
                sensor_name=sen.name,
                kind=sen.kind,
                times_s=times,
                values=np.asarray(vals, dtype=float),
                sigma=np.full(times.size, float(sen.sigma)),
                holdout=sen.name in hold,
            )
        )
    return series


_TIME_DAYS = re.compile(r"Time\s*=\s*([0-9.Ee+\-]+)", re.I)
_PLANE = re.compile(r"Plane\s+K\s*=\s*(\d+)", re.I)
_JROW = re.compile(r"J=\s*(\d+)\s+(.*)")
_ALLVAL = re.compile(r"All values are\s+([0-9.Ee+\-]+)", re.I)


def _flatten_planes(planes: dict[int, dict[int, list[float]]], nx: int, ny: int, nz: int) -> NDArray[np.float64]:
    field = np.full(nx * ny * nz, np.nan, dtype=float)
    for k, rows in planes.items():
        for j, vals in rows.items():
            for i, v in enumerate(vals):
                if i >= nx:
                    break
                cell = (int(k) - 1) * ny * nx + (int(j) - 1) * nx + i
                if 0 <= cell < field.size:
                    field[cell] = float(v)
    return field


def parse_gem_out_maps(out_path: str | Path, *, nx: int = 4, ny: int = 4, nz: int = 2) -> HiddenTruth:
    """Parse last GEM ASCII grid maps (kPa) into SI hidden truth. Scoring only."""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    t_days = 0.0
    maps: dict[tuple[str, str], NDArray[np.float64]] = {}
    kind = None
    current: str | None = None
    planes: dict[int, dict[int, list[float]]] = {}
    kplane = 1

    def flush() -> None:
        nonlocal planes, current, kind
        if kind and current and planes:
            maps[(kind, current)] = _flatten_planes(planes, nx, ny, nz)
        planes = {}

    for line in lines:
        tm = _TIME_DAYS.search(line)
        if tm:
            t_days = float(tm.group(1))
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
        if new_kind is not None:
            flush()
            kind = new_kind
            current = None
            kplane = 1
            continue
        if "Fundamental Grid - Matrix" in line:
            flush()
            current = "matrix"
        elif "Fundamental Grid - Fracture" in line:
            flush()
            current = "fracture"
        allv = _ALLVAL.search(line)
        if allv and kind and current:
            maps[(kind, current)] = np.full(nx * ny * nz, float(allv.group(1)))
            continue
        if "Fundamental Grid - Matrix" in line or "Fundamental Grid - Fracture" in line:
            continue
        pm = _PLANE.search(line)
        if pm:
            kplane = int(pm.group(1))
        jm = _JROW.match(line.strip())
        if jm and current is not None:
            planes.setdefault(kplane, {})[int(jm.group(1))] = [float(x) for x in jm.group(2).split()]
    flush()
    t_s = float(t_days) * 86400.0
    pf = maps.get(("pressure", "fracture"))
    pm = maps.get(("pressure", "matrix"))
    if pf is None and pm is None:
        raise ValueError(f"no pressure maps in {out_path}")
    if pf is None:
        pf = pm
    pf_pa = np.asarray(pf, dtype=float) * 1.0e3
    pm_pa = None if pm is None else np.asarray(pm, dtype=float) * 1.0e3
    sg = maps.get(("sg", "fracture"))
    so = maps.get(("so", "fracture"))
    sw = maps.get(("sw", "fracture"))
    return HiddenTruth(
        times_s=np.array([t_s], dtype=float),
        pressure=pf_pa.reshape(1, -1),
        sg=None if sg is None else np.asarray(sg, dtype=float).reshape(1, -1),
        so=None if so is None else np.asarray(so, dtype=float).reshape(1, -1),
        sw=None if sw is None else np.asarray(sw, dtype=float).reshape(1, -1),
        pressure_fracture=pf_pa.reshape(1, -1),
        pressure_matrix=None if pm_pa is None else pm_pa.reshape(1, -1),
        meta={"source": str(out_path), "t_days": t_days, "unit_pressure": "Pa"},
    )


_AVE = re.compile(r"Ave\.\s+(oil|gas) saturation\s+=\s+([0-9.Ee+\-]+)", re.I)
_ZGAS = re.compile(r"Ave\.\s+gas phase Z factor\s+=\s+([0-9.Ee+\-]+)", re.I)
_MOLES = re.compile(r"^\s+(C1|NC10|METHANE|DECANE)\s+=\s+([0-9.Ee+\-]+)", re.I | re.M)


def our_init_flash(*, p_pa: float = 1.2e7, t_k: float = 350.0, z=None) -> dict[str, float]:
    """Reference t=0 flash from our PR card. No wells."""
    from reservoir_backend.eos.example import example_c1_nc10
    from reservoir_backend.eos.flash import flash_tp

    z = np.asarray([0.55, 0.45] if z is None else z, dtype=float)
    fl = flash_tp(example_c1_nc10(), float(p_pa), float(t_k), z)
    beta = float(fl.vapor_frac)
    vl, vv = float(fl.v_liq), float(fl.v_vap)
    vmix = max(float(fl.v_mix), 1.0e-30)
    sg = beta * vv / vmix
    return {
        "vapor_frac": beta,
        "sg": sg,
        "so": 1.0 - sg,
        "v_liq": vl,
        "v_vap": vv,
        "p_pa": float(p_pa),
        "t_k": float(t_k),
        "z_c1": float(z[0]),
    }


def parse_gem_init_fluid(out_path: str | Path) -> dict[str, float]:
    """t=0 GEM flash from the Initial Reservoir Conditions block."""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    block = text
    if "Initial Reservoir Conditions" in text:
        block = text.split("Initial Reservoir Conditions", 1)[1]
        block = block.split("TIME:", 1)[0]
    rec: dict[str, float] = {}
    for m in _AVE.finditer(block):
        rec[f"sg" if m.group(1).lower() == "gas" else "so"] = float(m.group(2))
    zm = _ZGAS.search(block)
    if zm:
        rec["z_factor_gas"] = float(zm.group(1))
    moles = {}
    for m in _MOLES.finditer(block):
        moles[m.group(1).upper()] = float(m.group(2))
    if moles:
        tot = sum(moles.values()) or 1.0
        rec["z_c1"] = moles.get("C1", moles.get("METHANE", 0.0)) / tot
    return rec


def init_flash_report(out_path: str | Path, *, sg_tol: float = 0.05) -> dict[str, Any]:
    """Compare GEM t=0 saturations to our PR flash. First M2a sub-gate."""
    ours = our_init_flash()
    gem = parse_gem_init_fluid(out_path)
    dsg = abs(float(gem.get("sg", float("nan"))) - float(ours["sg"]))
    return {
        "gate": "m2a_init_flash",
        "ours": ours,
        "gem": gem,
        "d_sg": dsg,
        "sg_tol": sg_tol,
        "pass": bool(dsg < sg_tol),
    }


def find_gem_exe() -> Path | None:
    env = Path(__import__("os").environ.get("CMG_GEM_EXE", ""))
    if env.is_file():
        return env
    candidates = [
        Path(r"D:\Tool\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe"),
        Path(r"D:\Tool\CMG2021\GEM\2021.10\Win_x64\EXE\gm202110.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def run_gem(deck: str | Path, workdir: str | Path, *, exe: Path | None = None, timeout_s: float = 120.0) -> dict[str, Any]:
    """Launch local GEM. Does not invent results if the binary is missing."""
    import shutil
    import subprocess

    binary = exe or find_gem_exe()
    if binary is None:
        return {"ok": False, "blocked": "GEM executable not found"}
    deck = Path(deck)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    dest = work / deck.name
    if dest.resolve() != deck.resolve():
        shutil.copy2(deck, dest)
    proc = subprocess.run(
        [str(binary), "-f", dest.name],
        cwd=str(work),
        capture_output=True,
        text=True,
        timeout=float(timeout_s),
        check=False,
    )
    out_files = sorted(work.glob("*.out"))
    return {
        "ok": proc.returncode == 0 and bool(out_files),
        "returncode": int(proc.returncode),
        "exe": str(binary),
        "workdir": str(work),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "out_files": [str(p) for p in out_files],
    }


def write_comparison_plot(
    truth: HiddenTruth,
    prior: Mapping[str, NDArray[np.float64]],
    posterior: Mapping[str, NDArray[np.float64]],
    path: str | Path,
) -> None:
    """Three-panel pressure snapshot: CMG / prior / posterior. Optional matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    it = int(np.argmax(truth.times_s))
    panels = [
        ("CMG truth", truth.pressure[it]),
        ("Prior F_ours", np.asarray(prior["pressure"])[min(it, prior["pressure"].shape[0] - 1)]),
        ("Posterior F_ours", np.asarray(posterior["pressure"])[min(it, posterior["pressure"].shape[0] - 1)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
    vmin = min(float(np.min(a)) for _, a in panels)
    vmax = max(float(np.max(a)) for _, a in panels)
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(np.asarray(field).reshape(-1, 1).T if field.size <= 8 else np.asarray(field).reshape(-1)[None, :], aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
