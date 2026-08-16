"""Registry of CMG invert cases. VARI/DTOP is listed but not run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
VAL = REPO / "black_oil" / "validation"

DAY_S = 86400.0
FT_TO_M = 0.3048
MD_TO_M2 = 9.869233e-16
PSI = 6894.757293168

DEFAULT_WEIGHTS = {
    "hold": 1.0,
    "forecast": 1.0,
    "p": 0.5,
    "sw": 0.5,
    "bt": 0.3,
}

# Field IMEX is years-long black-oil; forecast p is not a lab-F pass bar.
FIELD_WEIGHTS = {
    "hold": 1.0,
    "forecast": 0.25,
    "p": 0.25,
    "sw": 0.5,
    "bt": 0.3,
}


@dataclass(frozen=True)
class CaseSpec:
    id: str
    kind: str
    status: str  # ready | unsupported | need_imex
    case_dir: Path
    truth_name: str
    out_name: str
    dat_name: str
    parameterization: str  # region | coarse
    coarse: tuple[int, int, int] | None = None
    history_days: tuple[float, ...] = (0.25, 0.50, 1.00)
    forecast_day: float | None = 2.00
    prior_k_md: float = 100.0
    dt_init_s: float = 30.0
    dt_max_s: float = 120.0
    invert_in_fast: bool = False
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    note: str = ""

    @property
    def truth_path(self) -> Path:
        return self.case_dir / self.truth_name

    @property
    def out_path(self) -> Path:
        return self.case_dir / self.out_name

    @property
    def dat_path(self) -> Path:
        return self.case_dir / self.dat_name


def _cases() -> dict[str, CaseSpec]:
    return {
        "lab_layers": CaseSpec(
            id="lab_layers",
            kind="layers",
            status="ready",
            case_dir=VAL / "cmg_lab_layers",
            truth_name="truth_lab_layers.json",
            out_name="lab_layers.out",
            dat_name="lab_layers.dat",
            parameterization="region",
            history_days=(0.25, 0.50, 1.00),
            forecast_day=2.00,
            invert_in_fast=True,
            note="lab analog, two-layer 50/500, 0.25–2 d",
        ),
        "fivespot": CaseSpec(
            id="fivespot",
            kind="fivespot",
            status="ready",
            case_dir=VAL / "cmg_fivespot",
            truth_name="truth_fivespot.json",
            out_name="mxspr006_fivespot.out",
            dat_name="mxspr006_fivespot.dat",
            parameterization="region",
            history_days=(1.0, 60.0, 242.0),
            forecast_day=426.0,
            prior_k_md=60.0,
            dt_init_s=3600.0,
            dt_max_s=86400.0,
            invert_in_fast=True,
            weights=dict(FIELD_WEIGHTS),
            note="field five-spot, IMEX times 1–791 d",
        ),
        "fault": CaseSpec(
            id="fault",
            kind="fault",
            status="ready",
            case_dir=VAL / "cmg_fault_3d",
            truth_name="truth_fault.json",
            out_name="mxspr006_fault.out",
            dat_name="mxspr006_fault.dat",
            parameterization="region",
            history_days=(1.0, 60.0, 242.0),
            forecast_day=426.0,
            prior_k_md=50.0,
            dt_init_s=3600.0,
            dt_max_s=86400.0,
            invert_in_fast=True,
            weights=dict(FIELD_WEIGHTS),
            note="field fault+window, IMEX times 1–791 d",
        ),
        "channel": CaseSpec(
            id="channel",
            kind="channel",
            status="ready",
            case_dir=VAL / "cmg_channel_3d",
            truth_name="truth_channel.json",
            out_name="mxspr006_channel.out",
            dat_name="mxspr006_channel.dat",
            parameterization="region",
            history_days=(1.0, 60.0, 242.0),
            forecast_day=426.0,
            prior_k_md=50.0,
            dt_init_s=3600.0,
            dt_max_s=86400.0,
            invert_in_fast=True,
            weights=dict(FIELD_WEIGHTS),
            note="CART surrogate of VARI channel (DTOP ignored)",
        ),
        "lab_box": CaseSpec(
            id="lab_box",
            kind="lab_box",
            status="need_imex",
            case_dir=VAL / "lab_box_30cm",
            truth_name="truth_lab_box.json",
            out_name="lab_box_30cm.out",
            dat_name="lab_box_30cm.dat",
            parameterization="region",
            note="30 cm draped high-k; .out not in tree",
        ),
        "shale_s1": CaseSpec(
            id="shale_s1",
            kind="shale",
            status="unsupported",
            case_dir=REPO / "shale_oil" / "validation" / "cmg_s1_hw5frac",
            truth_name="truth_s1.json",
            out_name="mxshale_s1.out",
            dat_name="mxshale_s1.dat",
            parameterization="region",
            note="shale depletion / frac strips — F is two-phase black-oil waterflood",
        ),
    }


def list_cases(*, include_blocked: bool = False) -> list[CaseSpec]:
    out = []
    for spec in _cases().values():
        if spec.status == "ready" or include_blocked:
            out.append(spec)
    return out


def get_case(case_id: str) -> CaseSpec:
    cases = _cases()
    if case_id not in cases:
        raise KeyError(f"unknown harness case {case_id!r}; have {sorted(cases)}")
    return cases[case_id]
