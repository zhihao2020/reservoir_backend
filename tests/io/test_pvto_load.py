"""CMG/IMEX *PVTO parser: tables actually drive BlackOilPVT."""

from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.io.pvt_cfg import pvt_from_cfg
from reservoir_backend.io.pvto_load import looks_like_cmg_pvt, parse_pvto
from reservoir_backend.physics.pvt import PSI, SCF_PER_STB, BlackOilPVT


def _spe_pvt_text() -> str:
    p_psi = [14.7, 264.7, 514.7, 1014.7, 2014.7, 2514.7, 3014.7, 4014.7, 5014.7, 9014.7]
    rs = [1.0, 90.5, 180.0, 371.0, 636.0, 775.0, 930.0, 1270.0, 1600.0, 2984.0]
    bo = [1.062, 1.15, 1.207, 1.295, 1.435, 1.5, 1.565, 1.695, 1.827, 2.36]
    eg = [6.0, 82.692, 159.388, 312.793, 619.579, 772.798, 925.926, 1233.046, 1540.832, 2590.674]
    viso = [1.04, 0.975, 0.91, 0.83, 0.69, 0.64, 0.594, 0.51, 0.449, 0.203]
    visg = [0.0080, 0.0096, 0.0112, 0.0140, 0.0189, 0.0208, 0.0228, 0.0268, 0.0309, 0.0470]
    lines = ["*INUNIT *FIELD", "*PVT *GRAPH"]
    for i in range(len(p_psi)):
        lines.append(
            "  %s  %s  %s  %s  %s  %s"
            % (p_psi[i], rs[i], bo[i], eg[i], viso[i], visg[i])
        )
    lines.append("*PVTW")
    lines.append("  14.7  1.04  3.04e-6  1.1")
    lines.append("*CO  1.3687e-5")
    lines.append("*PB  2500")
    return "\n".join(lines) + "\n"


def test_looks_like_cmg_pvt_keywords() -> None:
    assert looks_like_cmg_pvt("*PVTO\n  1 14.7 1.06 1.04\n")
    assert looks_like_cmg_pvt("*PVTW\n  14.7 1.04 3.04e-6 1.1\n")
    assert looks_like_cmg_pvt("*PVDG\n  14.7 0.16 0.008\n")
    assert looks_like_cmg_pvt("*PVT\n  14.7 1 1.06 6 1.04 0.008\n")
    assert not looks_like_cmg_pvt("p_tab: [1.0e5]\nrs_tab: [0.0]\n")


def test_parse_pvto_snippet_field_units() -> None:
    parsed = parse_pvto("*PVTO\n  1.0  14.7  1.06  1.04\n")
    assert abs(parsed["p_tab"][0] - 14.7 * PSI) < 1.0e-9
    assert abs(parsed["rs_tab"][0] - 1.0 * SCF_PER_STB) < 1.0e-12
    assert abs(parsed["bo_tab"][0] - 1.06) < 1.0e-15
    assert abs(parsed["muo_tab"][0] - 1.04e-3) < 1.0e-15


def test_parse_pvto_skips_undersaturated_rows() -> None:
    text = (
        "*PVTO\n"
        "  1.0   14.7   1.062  1.04\n"
        "        264.7  1.050  1.10\n"
        "  90.5  264.7  1.150  0.975\n"
    )
    parsed = parse_pvto(text)
    assert len(parsed["p_tab"]) == 2
    assert abs(parsed["rs_tab"][1] - 90.5 * SCF_PER_STB) < 1.0e-12
    assert abs(parsed["bo_tab"][1] - 1.150) < 1.0e-15


def test_parse_pvt_roundtrip_cmg_seawater() -> None:
    parsed = parse_pvto(_spe_pvt_text())
    base = BlackOilPVT.cmg_seawater()
    assert np.allclose(parsed["p_tab"], base.p_tab)
    assert np.allclose(parsed["rs_tab"], base.rs_tab)
    assert np.allclose(parsed["bo_tab"], base.bo_tab)
    assert np.allclose(parsed["eg_tab"], base.eg_tab)
    assert np.allclose(parsed["muo_tab"], base.muo_tab)
    assert np.allclose(parsed["mug_tab"], base.mug_tab)
    assert abs(parsed["cw"] - float(base.cw)) < 1.0e-18
    assert abs(parsed["bw_ref"] - float(base.bw_ref)) < 1.0e-15
    assert abs(parsed["mu_w"] - float(base.mu_w)) < 1.0e-15
    assert abs(parsed["pref_w"] - float(base.pref_w)) < 1.0e-6
    assert abs(parsed["co"] - float(base.co)) < 1.0e-18
    assert abs(parsed["pb"] - float(base.pb)) < 1.0e-6


def test_parse_pvdg_bg_and_eg() -> None:
    bg = parse_pvto("*PVDG\n  14.7  0.1666667  0.0080\n  2514.7  0.001294  0.0208\n")
    assert bg["p_tab"][0] == pytest.approx(14.7 * PSI)
    assert bg["eg_tab"][0] == pytest.approx((1.0 / 0.1666667) * SCF_PER_STB, rel=1.0e-6)
    assert bg["mug_tab"][1] == pytest.approx(0.0208e-3)
    eg = parse_pvto("*PVDG\n  14.7  6.0  0.0080\n  2514.7  772.798  0.0208\n")
    assert eg["eg_tab"][0] == pytest.approx(6.0 * SCF_PER_STB)
    assert eg["eg_tab"][1] == pytest.approx(772.798 * SCF_PER_STB)


def test_parse_pvtw_table() -> None:
    parsed = parse_pvto("*PVTW\n  14.7  1.04  1.1\n  3000  1.02  1.2\n")
    assert len(parsed["p_w_tab"]) == 2
    assert parsed["bw_tab"][0] == pytest.approx(1.04)
    assert parsed["muw_tab"][1] == pytest.approx(1.2e-3)


def test_parse_inunit_si() -> None:
    parsed = parse_pvto("*INUNIT *SI\n*PVTO\n  0.178  101.325  1.06  1.04\n")
    assert parsed["p_tab"][0] == pytest.approx(101.325e3)
    assert parsed["rs_tab"][0] == pytest.approx(0.178)
    assert parsed["muo_tab"][0] == pytest.approx(1.04e-3)


def test_parse_empty_section_errors() -> None:
    with pytest.raises(ValueError, match="no \\*PVTO"):
        parse_pvto("*PVTO\n** comment only\n")


def test_pvt_from_cfg_pvto_file_used(tmp_path) -> None:
    side = tmp_path / "tables.inc"
    side.write_text("*PVTO\n  1.0  14.7  1.06  1.04\n", encoding="utf-8")
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "file": "tables.inc"}},
        p_init=1.0e5,
        cfg_dir=tmp_path,
    )
    assert pvt.has_live_oil()
    p0 = 14.7 * PSI
    assert abs(float(pvt.rs(p0)) - 1.0 * SCF_PER_STB) < 1.0e-12
    assert abs(float(1.0 / pvt.b_o(p0)) - 1.06) < 1.0e-12
    assert abs(float(pvt.viscosity_o(p0)) - 1.04e-3) < 1.0e-15


def test_pvt_from_cfg_spe_file_matches_cmg_seawater(tmp_path) -> None:
    side = tmp_path / "spe.inc"
    side.write_text(_spe_pvt_text(), encoding="utf-8")
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "file": "spe.inc"}},
        p_init=3000.0 * PSI,
        cfg_dir=tmp_path,
    )
    base = BlackOilPVT.cmg_seawater()
    assert np.allclose(pvt.p_tab, base.p_tab)
    assert np.allclose(pvt.rs_tab, base.rs_tab)
    assert np.allclose(pvt.eg_tab, base.eg_tab)
    assert abs(float(pvt.rs(2514.7 * PSI)) - float(base.rs(2514.7 * PSI))) < 1.0e-12
    assert abs(float(pvt.b_g(2514.7 * PSI)) - float(base.b_g(2514.7 * PSI))) < 1.0e-12
    assert abs(float(pvt.cw) - float(base.cw)) < 1.0e-18
    assert abs(float(pvt.mu_w) - 1.1e-3) < 1.0e-15
