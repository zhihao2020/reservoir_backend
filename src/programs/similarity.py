"""三维物理模拟几何-运动相似准则及跨尺度换算函数。

固定物理模型：L_m = W_m = H_m = 0.30 m，总体积 V_m = 0.027 m³。等线速度条件
C_t = C_Lc，控制体积比 C_V = λ_x·λ_y·λ_z，体积流量比 C_qV = C_V/C_Lc。

单位约定（矿场 ↔ 实验室）：
- 时间     t_p (d)    ↔ t_m (min)
- 体积流量 q_V,p (m³/d) ↔ q_V,m (mL/min)
- 累计体积 V_inj,p (m³) ↔ V_inj,m (mL)

CO₂ 体积计量状态（标准态 / 实际温压态）由调用方保证输入输出统一，本模块不做
状态方程换算。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MODEL_LENGTH_M = 0.30  # 模型固定边长 L_m (m)
MODEL_WIDTH_M = 0.30   # 模型固定宽度 W_m (m)
MODEL_HEIGHT_M = 0.30  # 模型固定高度 H_m (m)
MODEL_VOLUME_M3 = MODEL_LENGTH_M * MODEL_WIDTH_M * MODEL_HEIGHT_M  # 0.027 m³
MODEL_VOLUME_ML = MODEL_VOLUME_M3 * 1.0e6                        # 27000 mL

_MIN_PER_DAY = 1440.0
_ML_PER_M3 = 1.0e6

# CO₂ 体积计量状态（必须统一）：标准状态 / 实际温压状态
VOLUME_BASES = ("standard", "reservoir", "actual")


def similarity_ratios(
    field_length_m: float,
    field_width_m: float,
    field_height_m: float,
    field_flow_path_m: float | None = None,
    *,
    model_flow_path_m: float = MODEL_LENGTH_M,
) -> dict[str, float]:
    """由矿场几何计算无量纲相似比。

    返回 ``lambda_x/y/z``（几何比）、``c_v``（控制体积比）、``c_lc``（代表流动
    路径比）、``c_t``（时间比 = c_lc，等线速度）、``c_qv``（体积流量比 = c_v/c_lc）、
    ``c_vinj``（累计注入体积比 = c_v）。
    """
    if min(field_length_m, field_width_m, field_height_m) <= 0.0:
        raise ValueError("field dimensions must be positive")
    if model_flow_path_m <= 0.0:
        raise ValueError("model flow path must be positive")
    lam_x = field_length_m / MODEL_LENGTH_M
    lam_y = field_width_m / MODEL_LENGTH_M
    lam_z = field_height_m / MODEL_LENGTH_M
    c_v = lam_x * lam_y * lam_z
    lc_p = field_flow_path_m if field_flow_path_m is not None else field_length_m
    if lc_p <= 0.0:
        raise ValueError("field flow path must be positive")
    c_lc = lc_p / model_flow_path_m
    return {
        "lambda_x": lam_x,
        "lambda_y": lam_y,
        "lambda_z": lam_z,
        "c_v": c_v,
        "c_lc": c_lc,
        "c_t": c_lc,
        "c_qv": c_v / c_lc,
        "c_vinj": c_v,
    }


# ---- 矿场 → 实验室 ----
def field_to_lab_time(time_field_d: float, c_lc: float) -> float:
    """t_m (min) = 1440 · t_p (d) / C_Lc"""
    return _MIN_PER_DAY * time_field_d / c_lc


def field_to_lab_rate(rate_field_m3_d: float, c_lc: float, c_v: float) -> float:
    """q_V,m (mL/min) = 694.444 · q_V,p (m³/d) · C_Lc / C_V"""
    return (_ML_PER_M3 / _MIN_PER_DAY) * rate_field_m3_d * c_lc / c_v


def field_to_lab_volume(volume_field_m3: float, c_v: float) -> float:
    """V_inj,m (mL) = 1e6 · V_inj,p (m³) / C_V"""
    return _ML_PER_M3 * volume_field_m3 / c_v


# ---- 实验室 → 矿场 ----
def lab_to_field_time(time_lab_min: float, c_lc: float) -> float:
    """t_p (d) = C_Lc · t_m (min) / 1440"""
    return c_lc * time_lab_min / _MIN_PER_DAY


def lab_to_field_rate(rate_lab_ml_min: float, c_lc: float, c_v: float) -> float:
    """q_V,p (m³/d) = 0.00144 · q_V,m (mL/min) · C_V / C_Lc"""
    return (_MIN_PER_DAY / _ML_PER_M3) * rate_lab_ml_min * c_v / c_lc


def lab_to_field_volume(volume_lab_ml: float, c_v: float) -> float:
    """V_inj,p (m³) = C_V · V_inj,m (mL) / 1e6"""
    return c_v * volume_lab_ml / _ML_PER_M3


# ---- 井位 / 坐标换算 ----
def _lambdas(ratios: dict[str, float] | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(ratios, dict):
        return (
            float(ratios["lambda_x"]),
            float(ratios["lambda_y"]),
            float(ratios["lambda_z"]),
        )
    lam = tuple(float(v) for v in ratios)
    if len(lam) != 3:
        raise ValueError("ratios must be a similarity dict or a (lambda_x, lambda_y, lambda_z) triple")
    return lam


def field_to_lab_point(
    x: float,
    y: float,
    z: float,
    lambda_x: float,
    lambda_y: float,
    lambda_z: float,
) -> tuple[float, float, float]:
    """矿场坐标 → 模型坐标：x_m = x_p / λ_x（y/z 同理）。"""
    return (x / lambda_x, y / lambda_y, z / lambda_z)


def lab_to_field_point(
    x: float,
    y: float,
    z: float,
    lambda_x: float,
    lambda_y: float,
    lambda_z: float,
) -> tuple[float, float, float]:
    """模型坐标 → 矿场坐标：x_p = λ_x·x_m（y/z 同理）。"""
    return (x * lambda_x, y * lambda_y, z * lambda_z)


def field_to_lab_coords(
    xyz: NDArray[np.float64],
    ratios: dict[str, float] | tuple[float, float, float],
) -> NDArray[np.float64]:
    """逐轴把矿场坐标（m）换算到模型坐标。``xyz`` 形如 ``(n, 3)``。"""
    lam = np.asarray(_lambdas(ratios), dtype=float)
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    return arr / lam[None, :]


def lab_to_field_coords(
    xyz: NDArray[np.float64],
    ratios: dict[str, float] | tuple[float, float, float],
) -> NDArray[np.float64]:
    """逐轴把模型坐标（m）换算到矿场坐标。``xyz`` 形如 ``(n, 3)``。"""
    lam = np.asarray(_lambdas(ratios), dtype=float)
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    return arr * lam[None, :]


# ---- 时间 / 流量 / 累计体积：任意两项求第三项 ----
def complete_injection(
    time: float | None = None,
    rate: float | None = None,
    volume: float | None = None,
) -> float:
    """由时间、流量、累计体积中任意两项求第三项（V = q·t）。

    三个量必须采用同一套单位：矿场用 (d, m³/d, m³)，实验室用 (min, mL/min, mL)。
    恰好给定两项、缺失项传 ``None``，返回缺失项。
    """
    given = sum(x is not None for x in (time, rate, volume))
    if given != 2:
        raise ValueError("provide exactly two of time, rate, volume (the third is derived)")
    if volume is None:
        return float(rate) * float(time)
    if rate is None:
        return float(volume) / float(time)
    return float(volume) / float(rate)


def field_to_lab(
    ratios: dict[str, float],
    *,
    time_d: float | None = None,
    rate_m3_d: float | None = None,
    volume_m3: float | None = None,
) -> dict[str, float]:
    """矿场 → 实验室的三量换算（时间、流量、累计体积），逐项可选。"""
    c_lc, c_v = ratios["c_lc"], ratios["c_v"]
    out: dict[str, float] = {}
    if time_d is not None:
        out["time_min"] = field_to_lab_time(time_d, c_lc)
    if rate_m3_d is not None:
        out["rate_ml_min"] = field_to_lab_rate(rate_m3_d, c_lc, c_v)
    if volume_m3 is not None:
        out["volume_ml"] = field_to_lab_volume(volume_m3, c_v)
    return out


def lab_to_field(
    ratios: dict[str, float],
    *,
    time_min: float | None = None,
    rate_ml_min: float | None = None,
    volume_ml: float | None = None,
) -> dict[str, float]:
    """实验室 → 矿场的三量换算（时间、流量、累计体积），逐项可选。"""
    c_lc, c_v = ratios["c_lc"], ratios["c_v"]
    out: dict[str, float] = {}
    if time_min is not None:
        out["time_d"] = lab_to_field_time(time_min, c_lc)
    if rate_ml_min is not None:
        out["rate_m3_d"] = lab_to_field_rate(rate_ml_min, c_lc, c_v)
    if volume_ml is not None:
        out["volume_m3"] = lab_to_field_volume(volume_ml, c_v)
    return out
