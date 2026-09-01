"""M1c experiment-design gate: H + R + u(t), no ES-MDA.

Each candidate is a handful of deterministic forwards. Feasibility uses an
explicit laboratory envelope (pump, PV, pressure), not “the solver ran”.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Sensor
from reservoir_backend.exceptions import PhysicsConvergenceError, TimeStepUnderflow
from reservoir_backend.inverse.observation_r import fisher_from_sensitivity, mahalanobis_d, observation_covariance
from reservoir_backend.twin.lab_v1 import (
    CF_TRUE_M2,
    D_CF_MIN,
    SIGMA_P,
    SIGMA_S,
    TMF_TRUE,
    _sample_sensor,
    load_lab_v1,
)


# Assumed bench envelope until the laboratory fills these in. SI.
Q_MAX_M3_S = 1.67e-6  # 100 mL/min
P_MIN_PA = 5.0e6
P_MAX_PA = 2.5e7
DP_SAFE_PA = 5.0e6
PV_MAX = 3.0
T_MAX_S = 1800.0
P_PROD_PA = 1.18e7
PHI_MATRIX = 0.08
PHI_FRACTURE = 0.02
RHO_BIAS = 0.30
TAU_S = 5.0
D_TMF_MIN = 2.0
H_KINDS = ("bulk_gauges", "tapped_channel", "dp_transducer")
YAML_PATH = Path(__file__).resolve().parents[2] / "examples" / "lab_v1" / "experiment_design.yaml"
# Steady 5% Cf signal is one ΔP, not a time series of the same drop.
STEADY_DP_PA = 2093.0  # measured at q_max on case_dev; do not invent a larger ΔP.


@dataclass
class Stage:
    duration_s: float
    q_inj: float


@dataclass
class LabEnvelope:
    q_max_m3_s: float = Q_MAX_M3_S
    p_min_pa: float = P_MIN_PA
    p_max_pa: float = P_MAX_PA
    dp_safe_pa: float = DP_SAFE_PA
    pv_max: float = PV_MAX
    t_max_s: float = T_MAX_S


@dataclass
class Instrument:
    """H and R. ``h`` is an explicit assumption about what the gauges see."""

    h: str = "bulk_gauges"  # bulk_gauges | tapped_channel | dp_transducer
    pressure_sigma_pa: float = SIGMA_P
    saturation_sigma: float = SIGMA_S
    dp_sigma_pa: float | None = None  # only if a differential transducer exists
    rho_bias: float = RHO_BIAS
    tau_s: float | None = TAU_S


@dataclass
class Design:
    name: str
    stages: list[Stage]
    instrument: Instrument
    note: str = ""
    evaluate: bool = True


@dataclass
class DesignResult:
    name: str
    d_cf: float
    d_tmf: float
    cond: float
    corr: float
    lambda_min: float
    p_max: float
    dp_max: float
    v_inj_m3: float
    n_pv: float
    t_end_s: float
    q_max_used: float
    n_obs: int
    solver: str
    feasible: bool
    min_dt: float = float("nan")
    infeasible_reasons: list[str] = field(default_factory=list)
    note: str = ""
    h: str = ""
    joint_ok: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "h": self.h,
            "d_cf": self.d_cf,
            "d_tmf": self.d_tmf,
            "cond": self.cond,
            "corr": self.corr,
            "lambda_min": self.lambda_min,
            "p_max": self.p_max,
            "dp_max": self.dp_max,
            "v_inj_m3": self.v_inj_m3,
            "n_pv": self.n_pv,
            "t_end_s": self.t_end_s,
            "q_max_used": self.q_max_used,
            "n_obs": self.n_obs,
            "min_dt": self.min_dt,
            "solver": self.solver,
            "feasible": self.feasible,
            "infeasible_reasons": list(self.infeasible_reasons),
            "note": self.note,
            "d_cf_ok": bool(self.d_cf >= D_CF_MIN),
            "d_tmf_ok": bool(self.d_tmf >= D_TMF_MIN),
            "joint_ok": bool(self.joint_ok),
        }


def pore_volume_m3(twin) -> float:
    vol = np.asarray(twin.grid.cell_volumes(), dtype=float)
    return float((PHI_MATRIX + PHI_FRACTURE) * vol.sum())


def injected_volume_m3(stages: list[Stage]) -> float:
    return float(sum(max(float(s.q_inj), 0.0) * float(s.duration_s) for s in stages))


def trajectory_min_dt(traj) -> float:
    """Smallest accepted solver dt. Reports keep every step even if states are downsampled."""
    dts = [float(r.dt) for r in (getattr(traj, "reports", None) or []) if getattr(r, "dt", None) is not None]
    dts = [d for d in dts if d > 0.0 and np.isfinite(d)]
    return float(min(dts)) if dts else float("nan")


def cf_detectability_bound(
    dp_pa: float,
    sigma_pa: float,
    *,
    rel: float = 0.05,
    n_indep: float = 1.0,
) -> float:
    """Upper bound on D_Cf,5%. Steady Cf is one ΔP (n_indep=1); time-series n is t/τ."""
    return float(abs(rel) * abs(dp_pa) / max(float(sigma_pa), 1.0e-30) * np.sqrt(max(float(n_indep), 0.0)))


def independent_samples(t_end_s: float, tau_s: float | None) -> float:
    if tau_s is None or float(tau_s) <= 0.0:
        return max(float(t_end_s), 1.0)
    return max(1.0, float(t_end_s) / float(tau_s))


def two_gauge_delta_sigma(sigma_pa: float) -> float:
    """σ of P1−P2 from two independent absolute gauges. Subtraction does not improve SNR."""
    return float(np.sqrt(2.0) * float(sigma_pa))


def joint_identifiable(corr: float, cond: float, lambda_min: float) -> bool:
    if corr != corr or not np.isfinite(cond) or not np.isfinite(lambda_min):
        return False
    return bool(abs(float(corr)) < 0.9 and float(cond) < 1.0e6 and float(lambda_min) > 1.0e-8)


def _as_tau(raw: Any) -> float | None:
    if raw is None or raw == "" or raw is False:
        return None
    return float(raw)


def instrument_from_mapping(raw: dict[str, Any] | None, *, default_h: str = "bulk_gauges") -> Instrument:
    raw = dict(raw or {})
    h = str(raw.get("h", default_h) or default_h)
    if h not in H_KINDS:
        raise ValueError(f"unknown H {h!r}; expected one of {H_KINDS}")
    dp_raw = raw.get("dp_sigma_pa", None)
    inst = Instrument(
        h=h,
        pressure_sigma_pa=float(raw.get("pressure_sigma_pa", SIGMA_P)),
        saturation_sigma=float(raw.get("saturation_sigma", SIGMA_S)),
        dp_sigma_pa=None if dp_raw in (None, "") else float(dp_raw),
        rho_bias=float(raw.get("rho_bias", RHO_BIAS)),
        tau_s=_as_tau(raw.get("tau_s", TAU_S)),
    )
    if inst.h == "dp_transducer" and inst.dp_sigma_pa is None:
        raise ValueError("dp_transducer requires the bench dp_sigma_pa; do not invent 30 Pa")
    return inst


def envelope_from_mapping(raw: dict[str, Any] | None) -> LabEnvelope:
    raw = dict(raw or {})
    return LabEnvelope(
        q_max_m3_s=float(raw.get("q_max_m3_s", Q_MAX_M3_S)),
        p_min_pa=float(raw.get("p_min_pa", P_MIN_PA)),
        p_max_pa=float(raw.get("p_max_pa", P_MAX_PA)),
        dp_safe_pa=float(raw.get("dp_safe_pa", DP_SAFE_PA)),
        pv_max=float(raw.get("pv_max", PV_MAX)),
        t_max_s=float(raw.get("t_max_s", T_MAX_S)),
    )


def stages_from_mapping(raw: list[Any]) -> list[Stage]:
    stages: list[Stage] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each stage must be a mapping with duration_s and q_inj")
        stages.append(Stage(duration_s=float(row["duration_s"]), q_inj=float(row["q_inj"])))
    if not stages:
        raise ValueError("design has no stages")
    return stages


def design_from_mapping(
    raw: dict[str, Any],
    *,
    default_instrument: Instrument | None = None,
    default_h: str = "bulk_gauges",
) -> Design:
    inst_raw = dict(raw.get("instrument") or {})
    if default_instrument is not None:
        inst_raw.setdefault("h", raw.get("h", default_instrument.h))
        inst_raw.setdefault("pressure_sigma_pa", default_instrument.pressure_sigma_pa)
        inst_raw.setdefault("saturation_sigma", default_instrument.saturation_sigma)
        inst_raw.setdefault("rho_bias", default_instrument.rho_bias)
        inst_raw.setdefault("tau_s", default_instrument.tau_s)
        if default_instrument.dp_sigma_pa is not None:
            inst_raw.setdefault("dp_sigma_pa", default_instrument.dp_sigma_pa)
    else:
        inst_raw.setdefault("h", raw.get("h", default_h))
    if "dp_sigma_pa" in raw and "dp_sigma_pa" not in inst_raw:
        inst_raw["dp_sigma_pa"] = raw["dp_sigma_pa"]
    stages_raw = raw.get("stages")
    if stages_raw is None and isinstance(raw.get("controls"), dict):
        stages_raw = raw["controls"].get("stages")
    return Design(
        name=str(raw.get("name") or "yaml"),
        stages=stages_from_mapping(list(stages_raw or [])),
        instrument=instrument_from_mapping(inst_raw, default_h=str(inst_raw.get("h", default_h))),
        note=str(raw.get("note") or ""),
        evaluate=bool(raw.get("evaluate", True)),
    )


def load_experiment_design_yaml(path: str | Path | None = None) -> tuple[LabEnvelope, list[Design]]:
    """Load instrument + envelope + staged designs. Missing designs fall back to the catalog."""
    path = Path(path) if path is not None else YAML_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    env = envelope_from_mapping(data.get("envelope"))
    default_h = str(data.get("h") or "bulk_gauges")
    default_inst = instrument_from_mapping(data.get("instrument"), default_h=default_h)
    designs: list[Design] = []
    if isinstance(data.get("designs"), list) and data["designs"]:
        for row in data["designs"]:
            if not isinstance(row, dict):
                raise ValueError("designs entries must be mappings")
            designs.append(design_from_mapping(row, default_instrument=default_inst, default_h=default_h))
    elif isinstance(data.get("controls"), dict) and data["controls"].get("stages"):
        designs.append(
            design_from_mapping(
                {"name": str(data.get("name") or path.stem), "controls": data["controls"], "note": str(data.get("note") or "")},
                default_instrument=default_inst,
                default_h=default_h,
            )
        )
    else:
        designs = default_candidates(env)
    return env, designs


def report_times(stages: list[Stage]) -> tuple[float, NDArray[np.float64]]:
    t_end = float(sum(float(s.duration_s) for s in stages))
    t = 0.0
    times = [min(0.5, 0.25 * float(stages[0].duration_s))]
    for st in stages:
        dur = float(st.duration_s)
        times.append(t + 0.5 * dur)
        t += dur
        times.append(t)
    arr = np.unique(np.clip(np.asarray(times, dtype=float), 1.0e-3, t_end))
    return t_end, arr


def instrument_sensors(inst: Instrument) -> list[Sensor]:
    if inst.h not in H_KINDS:
        raise ValueError(f"unknown H {inst.h!r}; expected one of {H_KINDS}")
    y0, z0 = 0.15, 0.15
    if inst.h == "tapped_channel":
        med = "fracture"
    else:
        med = "bulk"
    sig_p = float(inst.pressure_sigma_pa)
    sig_s = float(inst.saturation_sigma)
    return [
        Sensor("P_in", "pressure", 0.05, y0, z0, sigma=sig_p, medium=med, probe_diameter_m=0.006),
        Sensor("P_mid", "pressure", 0.15, y0, z0, sigma=sig_p, medium=med, probe_diameter_m=0.006),
        Sensor("P_out", "pressure", 0.25, y0, z0, sigma=sig_p, medium=med, probe_diameter_m=0.006),
        Sensor("S_in", "sg", 0.08, y0, z0, sigma=sig_s, medium="bulk", probe_diameter_m=0.006),
        Sensor("S_out", "sg", 0.22, y0, z0, sigma=sig_s, medium="bulk", probe_diameter_m=0.006),
    ]


def apply_stages(twin, stages: list[Stage]) -> float:
    t_end, _ = report_times(stages)
    t_knots: list[float] = []
    q_knots: list[float] = []
    t = 0.0
    for st in stages:
        t_knots.append(t)
        q_knots.append(float(st.q_inj))
        t += float(st.duration_s)
        t_knots.append(t)
        q_knots.append(float(st.q_inj))
    times = np.asarray(t_knots, dtype=float)
    qs = np.asarray(q_knots, dtype=float)
    controls = [
        ControlSeries("INJ", "rate", times, qs),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.95)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, P_PROD_PA)),
    ]
    twin.experiment.controls = controls
    twin.experiment.history_end_s = float(t_end)
    return float(t_end)


def _pack_vector(twin, traj, times: NDArray[np.float64], inst: Instrument):
    values: list[float] = []
    sigma: list[float] = []
    names: list[str] = []
    kinds: list[str] = []
    ts: list[float] = []
    pin: dict[float, float] = {}
    pout: dict[float, float] = {}
    for t in times:
        tf = float(t)
        for sen in twin.experiment.sensors:
            v = float(_sample_sensor(twin, sen, traj, tf))
            values.append(v)
            sigma.append(float(sen.sigma))
            names.append(sen.name)
            kinds.append(sen.kind)
            ts.append(tf)
            if sen.name == "P_in":
                pin[tf] = v
            if sen.name == "P_out":
                pout[tf] = v
    if inst.h == "dp_transducer" and inst.dp_sigma_pa is not None:
        sig = float(inst.dp_sigma_pa)
        for tf in times:
            tf = float(tf)
            if tf in pin and tf in pout:
                values.append(pin[tf] - pout[tf])
                sigma.append(sig)
                names.append("dP_io")
                kinds.append("pressure")
                ts.append(tf)
    return (
        np.asarray(values, dtype=float),
        np.asarray(sigma, dtype=float),
        names,
        kinds,
        np.asarray(ts, dtype=float),
    )


def _traj_pressure_stats(traj) -> tuple[float, float]:
    p_max = 0.0
    dp_max = 0.0
    for st in traj.states:
        p = np.asarray(st.pressure, dtype=float)
        p_max = max(p_max, float(p.max()))
        dp_max = max(dp_max, float(p.max() - p.min()))
    return p_max, dp_max


def evaluate_design(
    design: Design,
    *,
    envelope: LabEnvelope | None = None,
    cf_true: float = CF_TRUE_M2,
    tmf_true: float = TMF_TRUE,
) -> DesignResult:
    env = envelope or LabEnvelope()
    twin = load_lab_v1(dev=True)
    twin.experiment.sensors = instrument_sensors(design.instrument)
    t_end = apply_stages(twin, design.stages)
    _, times = report_times(design.stages)
    v_inj = injected_volume_m3(design.stages)
    n_pv = v_inj / max(pore_volume_m3(twin), 1.0e-30)
    q_peak = max(float(s.q_inj) for s in design.stages)
    reasons: list[str] = []
    if t_end > env.t_max_s + 1.0e-9:
        reasons.append("duration")
    if q_peak > env.q_max_m3_s + 1.0e-15:
        reasons.append("q_max")
    if n_pv > env.pv_max + 1.0e-12:
        reasons.append("pv")

    param = twin.parameterization
    theta0 = param.encode(np.array([float(cf_true), float(tmf_true)], dtype=float))
    theta_cf = param.encode(np.array([float(cf_true) * 1.05, float(tmf_true)], dtype=float))
    theta_tmf = param.encode(np.array([float(cf_true), float(tmf_true) * 1.10], dtype=float))
    solver = "PASS"
    p_max = float("nan")
    dp_max = float("nan")
    d_cf = 0.0
    d_tmf = 0.0
    cond = float("inf")
    corr = float("nan")
    lam_min = 0.0
    n_obs = 0
    min_dt = float("nan")
    try:
        traj0 = twin.simulate(parameters=theta0, t_end=t_end, report_times=times)
        y0, sig, names, kinds, ts = _pack_vector(twin, traj0, times, design.instrument)
        n_obs = int(y0.size)
        p_max, dp_max = _traj_pressure_stats(traj0)
        min_dt = trajectory_min_dt(traj0)
        if p_max > env.p_max_pa or p_max < env.p_min_pa:
            reasons.append("pressure")
        if dp_max > env.dp_safe_pa:
            reasons.append("dp_safe")
        traj_cf = twin.simulate(parameters=theta_cf, t_end=t_end, report_times=times)
        y_cf, _, _, _, _ = _pack_vector(twin, traj_cf, times, design.instrument)
        traj_tmf = twin.simulate(parameters=theta_tmf, t_end=t_end, report_times=times)
        y_tmf, _, _, _, _ = _pack_vector(twin, traj_tmf, times, design.instrument)
        r = observation_covariance(
            names, ts, sig, kinds, rho_bias=design.instrument.rho_bias, tau_s=design.instrument.tau_s
        )
        d_cf = mahalanobis_d(y_cf - y0, r)
        d_tmf = mahalanobis_d(y_tmf - y0, r)
        dth_cf = float(theta_cf[0] - theta0[0])
        dth_tmf = float(theta_tmf[1] - theta0[1])
        s = np.column_stack(((y_cf - y0) / max(dth_cf, 1.0e-12), (y_tmf - y0) / max(dth_tmf, 1.0e-12)))
        fish = fisher_from_sensitivity(s, r)
        evals = np.sort(np.linalg.eigvalsh(0.5 * (fish + fish.T)))
        lam_min = float(max(evals[0], 0.0))
        cond = float(evals[-1] / max(evals[0], 1.0e-30))
        if np.std(s[:, 0]) > 0 and np.std(s[:, 1]) > 0:
            corr = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    except (PhysicsConvergenceError, TimeStepUnderflow, ValueError, ArithmeticError) as exc:
        solver = f"{type(exc).__name__}"
        reasons.append("solver")

    feasible = solver == "PASS" and not reasons
    joint = joint_identifiable(corr, cond, lam_min)
    return DesignResult(
        name=design.name,
        d_cf=float(d_cf),
        d_tmf=float(d_tmf),
        cond=float(cond),
        corr=float(corr) if corr == corr else float("nan"),
        lambda_min=float(lam_min),
        p_max=float(p_max),
        dp_max=float(dp_max),
        v_inj_m3=float(v_inj),
        n_pv=float(n_pv),
        t_end_s=float(t_end),
        q_max_used=float(q_peak),
        n_obs=n_obs,
        solver=solver,
        feasible=bool(feasible),
        min_dt=float(min_dt),
        infeasible_reasons=reasons,
        note=design.note,
        h=design.instrument.h,
        joint_ok=bool(joint),
    )


def default_candidates(env: LabEnvelope | None = None) -> list[Design]:
    """Discrete laboratory catalog: constant, long-constant, pulse-1, pulse-rest, multistep.

    Long-constant is listed but ``evaluate=False``: wall time is hours and the
    steady-ΔP bound already shows D_Cf,5% stays below 2 at 2 kPa.
    """
    env = env or LabEnvelope()
    q = float(env.q_max_m3_s)
    bulk = Instrument(h="bulk_gauges")
    tapped = Instrument(h="tapped_channel")
    dp = Instrument(h="dp_transducer", dp_sigma_pa=200.0)
    pulse_rest = [Stage(10.0, q), Stage(20.0, 0.0), Stage(20.0, 0.2 * q), Stage(10.0, q)]
    pulse_1 = [Stage(10.0, q), Stage(50.0, 0.0)]
    multistep = [Stage(15.0, 0.25 * q), Stage(15.0, 0.5 * q), Stage(15.0, q), Stage(15.0, 0.5 * q)]
    return [
        Design(
            "constant",
            [Stage(60.0, q)],
            bulk,
            note="H=bulk gauges (real default). 60 s at q_max=100 mL/min. 2 kPa abs.",
        ),
        Design(
            "constant_tapped",
            [Stage(60.0, q)],
            tapped,
            note="ASSUMPTION: probes tap the fracture channel. Same u(t).",
        ),
        Design(
            "long_constant",
            [Stage(float(env.t_max_s), q)],
            bulk,
            note="600–1800 s constant q_max. Not forwarded: steady Cf is one ΔP.",
            evaluate=False,
        ),
        Design(
            "pulse_1",
            pulse_1,
            bulk,
            note="short pulse then rest. Same q_max ΔP; bound D_Cf≪2. Forward with --include-long.",
            evaluate=False,
        ),
        Design(
            "pulse_rest",
            pulse_rest,
            bulk,
            note="pulse / rest / moderate / pulse. H=bulk gauges.",
        ),
        Design(
            "pulse_rest_tapped",
            pulse_rest,
            tapped,
            note="same u(t); ASSUMPTION tapped fracture channel.",
        ),
        Design(
            "pulse_rest_dp",
            pulse_rest,
            dp,
            note="same u(t); bulk gauges + hypothetical 200 Pa DP transducer (must match the bench).",
        ),
        Design(
            "multistep",
            multistep,
            bulk,
            note="four rate steps at or below q_max. Same ΔP ceiling; bound D_Cf≪2.",
            evaluate=False,
        ),
        Design(
            "legacy_m1b_rate",
            [Stage(60.0, 3.0e-4)],
            Instrument(h="tapped_channel"),
            note="M1b q=3e-4 for 60 s. Expected infeasible (PV and q_max).",
        ),
    ]


def select_designs(designs: list[Design], *, include_long: bool = False, name: str | None = None) -> list[Design]:
    out = list(designs)
    if name:
        want = str(name)
        out = [d for d in out if d.name == want]
        if not out:
            raise ValueError(f"no design named {want!r}")
        return out
    if include_long:
        return out
    return [d for d in out if d.evaluate]
