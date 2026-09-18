"""Similarity conversion functions: round-trips and unit consistency."""

import numpy as np
import pytest

from src.programs.similarity import (
    complete_injection,
    field_to_lab,
    field_to_lab_coords,
    field_to_lab_point,
    lab_to_field,
    lab_to_field_coords,
    lab_to_field_point,
    similarity_ratios,
)


def _ratios():
    return similarity_ratios(300.0, 300.0, 30.0, 150.0, model_flow_path_m=0.15)


def test_ratios_geometry():
    r = _ratios()
    assert r["lambda_x"] == pytest.approx(1000.0)
    assert r["lambda_y"] == pytest.approx(1000.0)
    assert r["lambda_z"] == pytest.approx(100.0)
    assert r["c_v"] == pytest.approx(1000 * 1000 * 100)
    assert r["c_lc"] == pytest.approx(150.0 / 0.15)
    assert r["c_t"] == r["c_lc"]
    assert r["c_qv"] == pytest.approx(r["c_v"] / r["c_lc"])


def test_time_rate_volume_roundtrip():
    r = _ratios()
    lab = field_to_lab(r, time_d=1.0, rate_m3_d=1.0, volume_m3=1.0)
    field = lab_to_field(r, time_min=lab["time_min"], rate_ml_min=lab["rate_ml_min"], volume_ml=lab["volume_ml"])
    assert field["time_d"] == pytest.approx(1.0)
    assert field["rate_m3_d"] == pytest.approx(1.0)
    assert field["volume_m3"] == pytest.approx(1.0)


def test_point_roundtrip():
    r = _ratios()
    lx, ly, lz = r["lambda_x"], r["lambda_y"], r["lambda_z"]
    xf, yf, zf = 150.0, 150.0, 15.0
    xm, ym, zm = field_to_lab_point(xf, yf, zf, lx, ly, lz)
    assert lab_to_field_point(xm, ym, zm, lx, ly, lz) == pytest.approx((xf, yf, zf))


def test_coords_roundtrip():
    r = _ratios()
    xyz = np.array([[150.0, 150.0, 15.0], [0.0, 0.0, 0.0]])
    lab = field_to_lab_coords(xyz, r)
    assert lab_to_field_coords(lab, r) == pytest.approx(xyz)


def test_complete_injection():
    assert complete_injection(time=2.0, rate=3.0) == pytest.approx(6.0)
    assert complete_injection(volume=6.0, rate=3.0) == pytest.approx(2.0)
    assert complete_injection(time=2.0, volume=6.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        complete_injection(time=2.0, rate=3.0, volume=6.0)
