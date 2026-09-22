"""YAML/CSV case loading and unit conversion at the I/O boundary."""

from pathlib import Path

import numpy as np
import pytest

from src.core.lab_case import load_lab_case
from src.core.units import MD_TO_M2, to_m2, to_m3_s, to_metres, to_pa, to_seconds
from src.exceptions import CaseSchemaError


def test_unit_conversions():
    assert to_metres(5, "cm") == pytest.approx(0.05)
    assert to_seconds(1, "min") == pytest.approx(60.0)
    assert to_seconds(1, "d") == pytest.approx(86400.0)
    assert to_pa(1, "MPa") == pytest.approx(1.0e6)
    assert to_pa(100, "kPa") == pytest.approx(1.0e5)
    assert to_m2(0.03, "md") == pytest.approx(0.03 * MD_TO_M2)
    assert to_m3_s(1, "mL/min") == pytest.approx(1.0e-6 / 60.0)
    assert to_m3_s(1, "m3/d") == pytest.approx(1.0 / 86400.0)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_lab_case_units_and_order(tmp_path: Path):
    _write(tmp_path / "probes.csv", "id,x_m,y_m,z_m\nP1,0.05,0.15,0.15\nP2,0.25,0.15,0.15\n")
    _write(tmp_path / "wells.csv", "id,kind,x_m,y_m,z_m,x2_m,y2_m,z2_m\nINJ,injector,0.15,0.15,0.29,0.15,0.15,0.09\nPROD,producer,0.25,0.25,0.15,,,\n")
    _write(tmp_path / "obs.csv", "time_s,probe,quantity,value,unit\n2592000,P1,pressure,19.1,MPa\n2592000,P1,sg,0,,\n2592000,P1,so,1,,\n2592000,P1,sw,0,,\n2592000,P2,pressure,19.0,MPa\n2592000,P2,sg,0,,\n2592000,P2,so,1,,\n2592000,P2,sw,0,,\n")
    _write(tmp_path / "series.csv", "time_s,well,pw_pa,q_m3s,qw_m3s,qo_m3s,qg_m3s\n2592000,INJ,19325000,8.3e-8,0,0,8.3e-8\n2592000,PROD,19000000,-1.3e-8,0,-1.3e-8,0\n")
    yaml = (
        "geometry:\n  origin_m: [0,0,0]\n  extent_m: [0.30,0.30,0.30]\n  nx: 8\n  ny: 8\n  nz: 8\n"
        "rock:\n  phi0: 0.05\n  k0_md: 0.03\n"
        "probes:\n  file: probes.csv\n  observations: obs.csv\n"
        "wells:\n  file: wells.csv\n  series: series.csv\n"
        "inversion:\n  model: total_mobility\n"
        "interpolation:\n  method: auto\n"
        "similarity:\n  field_length_m: 300.0\n  field_width_m: 300.0\n  field_height_m: 30.0\n"
    )
    _write(tmp_path / "case.yaml", yaml)

    case = load_lab_case(tmp_path / "case.yaml")
    assert case.probe_ids == ["P1", "P2"]
    assert case.well_ids == ["INJ", "PROD"]
    assert case.k0 == pytest.approx(0.03 * MD_TO_M2)
    # pressure in MPa converted to Pa
    assert case.pressure[0, 0] == pytest.approx(19.1e6)
    # injector gas rate positive, producer oil rate negative
    assert case.well_qg[0, 0] == pytest.approx(8.3e-8)
    assert case.well_qo[0, 1] == pytest.approx(-1.3e-8)


def test_load_lab_case_missing_series_raises(tmp_path: Path):
    _write(tmp_path / "probes.csv", "id,x_m,y_m,z_m\nP1,0.05,0.15,0.15\n")
    _write(tmp_path / "wells.csv", "id,kind,x_m,y_m,z_m\nINJ,injector,0.15,0.15,0.15\n")
    yaml = (
        "geometry:\n  origin_m: [0,0,0]\n  extent_m: [0.30,0.30,0.30]\n  nx: 4\n  ny: 4\n  nz: 4\n"
        "rock:\n  phi0: 0.05\n  k0_md: 0.03\n"
        "probes:\n  file: probes.csv\n"
        "wells:\n  file: wells.csv\n"
    )
    _write(tmp_path / "case.yaml", yaml)
    # observations/series required unless require_series=False
    with pytest.raises(CaseSchemaError):
        load_lab_case(tmp_path / "case.yaml")
    case = load_lab_case(tmp_path / "case.yaml", require_series=False)
    assert case.times.size == 0


def test_require_series_false_skips_present_files(tmp_path: Path):
    # Online (streaming) mode must start empty even when the yaml references
    # observation/series files (those belong to the offline path).
    case_dir = Path(__file__).resolve().parents[1] / "examples" / "small"
    case = load_lab_case(case_dir / "case.yaml", require_series=False)
    assert case.times.size == 0
    assert case.pressure.shape == (0, len(case.probes))
    assert case.well_pw.shape == (0, len(case.wells))
