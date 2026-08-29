"""Case PVT factory: presets stamp one fluid; μ not inverted."""

from __future__ import annotations

import warnings

import numpy as np

import pytest

from reservoir_backend.io.case import load_case
from reservoir_backend.io.pvt_cfg import pvt_from_cfg, pvt_preset_name
from reservoir_backend.physics.pvt import PSI, SCF_PER_STB, BlackOilPVT


def test_pvt_from_cfg_incompressible_default() -> None:
    pvt = pvt_from_cfg({}, p_init=1.0e5)
    assert not pvt.has_live_oil()
    assert not pvt.has_storage()
    assert pvt_preset_name({}) == "incompressible"


def test_pvt_from_cfg_legacy_compressibility() -> None:
    pvt = pvt_from_cfg({"compressibility": 2.0e-9}, p_init=1.5e5)
    assert pvt.has_storage()
    assert abs(float(pvt.cr) - 2.0e-9) < 1.0e-20
    assert pvt_preset_name({"compressibility": 2.0e-9}) == "slightly_compressible"


def test_pvt_from_cfg_cmg_seawater_string_and_alias() -> None:
    pvt = pvt_from_cfg({"pvt": "cmg_seawater"}, p_init=3000.0 * 6894.757293168)
    assert pvt.has_live_oil()
    assert abs(float(pvt.mu_o) - 0.64e-3) < 1.0e-9
    assert pvt_from_cfg({"pvt": "cmg"}, p_init=1.0e5).has_live_oil()
    assert pvt_from_cfg({}, p_init=1.0e5, model="black_oil").has_live_oil()


def test_pvt_from_cfg_mapping_mu_override_dead_oil() -> None:
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "mu_w": 1.1e-3, "mu_o": 0.64e-3, "mu_g": 2.08e-5}},
        p_init=1.0e5,
    )
    assert abs(float(pvt.mu_w) - 1.1e-3) < 1.0e-15
    assert abs(float(pvt.mu_o) - 0.64e-3) < 1.0e-15
    assert abs(float(pvt.mu_g) - 2.08e-5) < 1.0e-15


def test_pvt_from_cfg_unknown_map_keys_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pvt_from_cfg({"pvt": {"preset": "incompressible", "not_a_column": 1}}, p_init=1.0e5)
    assert any("ignores unknown keys" in str(w.message) for w in caught)
    assert not any("tables not parsed yet" in str(w.message) for w in caught)


def test_pvt_from_cfg_unknown_preset_errors() -> None:
    with pytest.raises(ValueError, match="unknown physics.pvt"):
        pvt_from_cfg({"pvt": "winprop_custom"}, p_init=1.0e5)


def test_load_case_stamps_relperm_mu_from_pvt() -> None:
    twin = load_case("examples/lab/lab_30cm.yaml")
    assert abs(float(twin.physics.relperm.mu_o) - float(twin.physics.pvt.mu_o)) < 1.0e-15
    assert abs(float(twin.physics.relperm.mu_w) - float(twin.physics.pvt.mu_w)) < 1.0e-15
    assert not twin.physics.pvt.has_live_oil()


def test_load_case_cmg_seawater_guard(tmp_path) -> None:
    yml = tmp_path / "cmg_mini.yaml"
    yml.write_text(
        """
geometry: {size_m: [0.08, 0.08, 0.08]}
grid: {type: cartesian, spacing_m: 0.04}
physics:
  model: two_phase_immiscible
  capillary: none
  pvt: cmg_seawater
  p_init: 2.068e7
  sw_init: 0.2
  dt_init: 1.0
  dt_max: 10.0
  transport: implicit
rock: {porosity: 0.3}
ports:
  - {name: INJ, role: injector, control: rate, x: 0.02, y: 0.04, z: 0.04, sw_inj: 1.0}
  - {name: PROD, role: producer, control: pressure, x: 0.06, y: 0.04, z: 0.04}
sensors: []
inverse: {parameterization: region, n_regions: 1}
experiment:
  history_end_s: 10
  controls:
    - {port: INJ, kind: rate, times: [0, 10], values: [1.0e-9, 1.0e-9]}
    - {port: PROD, kind: pressure, times: [0, 10], values: [2.0e7, 2.0e7]}
""",
        encoding="utf-8",
    )
    twin = load_case(yml)
    assert twin.physics.pvt.has_live_oil()
    assert abs(float(twin.physics.relperm.mu_o) - float(twin.physics.pvt.mu_o)) < 1.0e-15
    assert abs(float(twin.physics.pvt.mu_o) - 0.64e-3) < 1.0e-9
    # Same object class as factory
    assert isinstance(twin.physics.pvt, BlackOilPVT)



def test_pvt_from_cfg_string_incompressible() -> None:
    pvt = pvt_from_cfg({"pvt": "incompressible"}, p_init=1.0e5)
    assert not pvt.has_live_oil()
    assert pvt.p_tab is None
    assert abs(float(pvt.b_w(2.0e5)) - 1.0) < 1.0e-15


def test_pvt_from_cfg_mapping_tables_used() -> None:
    p = [1.0e5, 2.0e5, 3.0e5]
    rs = [0.0, 10.0, 20.0]
    bo = [1.1, 1.2, 1.3]
    eg = [5.0, 8.0, 12.0]
    muo = [2.0e-3, 1.5e-3, 1.0e-3]
    mug = [2.0e-5, 2.5e-5, 3.0e-5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pvt = pvt_from_cfg(
            {
                "pvt": {
                    "preset": "incompressible",
                    "p_tab": p,
                    "rs_tab": rs,
                    "bo_tab": bo,
                    "eg_tab": eg,
                    "muo_tab": muo,
                    "mug_tab": mug,
                }
            },
            p_init=1.0e5,
        )
    assert not any("unknown keys" in str(w.message) for w in caught)
    assert pvt.has_live_oil()
    assert np.allclose(pvt.p_tab, p)
    assert np.allclose(pvt.rs_tab, rs)
    assert abs(float(pvt.rs(2.0e5)) - 10.0) < 1.0e-12
    assert abs(float(pvt.b_g(1.0e5)) - 5.0) < 1.0e-12
    assert abs(float(pvt.viscosity_o(3.0e5)) - 1.0e-3) < 1.0e-15
    assert abs(float(1.0 / pvt.b_o(1.0e5)) - 1.1) < 1.0e-12


def test_pvt_from_cfg_table_aliases_and_bg() -> None:
    pvt = pvt_from_cfg(
        {
            "pvt": {
                "preset": "incompressible",
                "p": [1.0e5, 2.0e5],
                "rs": [0.0, 5.0],
                "bo": [1.1, 1.2],
                "bg": [0.2, 0.1],
                "muo": [1.0e-3, 0.8e-3],
                "mug": [2.0e-5, 3.0e-5],
            }
        },
        p_init=1.0e5,
    )
    assert np.allclose(pvt.eg_tab, [5.0, 10.0])
    assert abs(float(pvt.b_g(1.0e5)) - 5.0) < 1.0e-12
    assert abs(float(pvt.rs(2.0e5)) - 5.0) < 1.0e-12


def test_pvt_from_cfg_tables_overlay_preset() -> None:
    base = BlackOilPVT.cmg_seawater()
    custom_rs = np.linspace(0.0, 1.0, int(base.rs_tab.size))
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "cmg_seawater", "rs_tab": custom_rs.tolist()}},
        p_init=3000.0 * 6894.757293168,
    )
    assert np.allclose(pvt.rs_tab, custom_rs)
    assert np.allclose(pvt.p_tab, base.p_tab)
    assert np.allclose(pvt.bo_tab, base.bo_tab)


def test_pvt_from_cfg_file_sidecar(tmp_path) -> None:
    side = tmp_path / "my_pvt.yaml"
    side.write_text(
        "p_tab: [100000.0, 200000.0]\nrs_tab: [0.0, 4.0]\nbo_tab: [1.05, 1.15]\n",
        encoding="utf-8",
    )
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "file": "my_pvt.yaml"}},
        p_init=1.0e5,
        cfg_dir=tmp_path,
    )
    assert pvt.has_live_oil()
    assert abs(float(pvt.rs(2.0e5)) - 4.0) < 1.0e-12


def test_pvt_from_cfg_json_sidecar(tmp_path) -> None:
    side = tmp_path / "oil.json"
    side.write_text(
        '{"p_tab": [1.0e5, 2.0e5], "rs_tab": [0.0, 2.5], "bo_tab": [1.02, 1.08]}',
        encoding="utf-8",
    )
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "pvto": "oil.json"}},
        p_init=1.0e5,
        cfg_dir=tmp_path,
    )
    assert abs(float(pvt.rs(2.0e5)) - 2.5) < 1.0e-12


def test_pvt_from_cfg_cmg_text_used(tmp_path) -> None:
    side = tmp_path / "deck.dat"
    side.write_text("*PVTO\n  1.0  14.7  1.06  1.04\n", encoding="utf-8")
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "file": "deck.dat"}},
        p_init=1.0e5,
        cfg_dir=tmp_path,
    )
    assert pvt.has_live_oil()
    p0 = 14.7 * PSI
    assert abs(float(pvt.rs(p0)) - 1.0 * SCF_PER_STB) < 1.0e-12
    assert abs(float(1.0 / pvt.b_o(p0)) - 1.06) < 1.0e-12
    assert abs(float(pvt.viscosity_o(p0)) - 1.04e-3) < 1.0e-15


def test_pvt_from_cfg_water_table() -> None:
    pvt = pvt_from_cfg(
        {"pvt": {"preset": "incompressible", "p_w": [1.0e5, 2.0e5], "bw": [1.0, 1.02], "muw": [1.0e-3, 1.2e-3]}},
        p_init=1.0e5,
    )
    assert abs(float(pvt.b_w(1.0e5)) - 1.0) < 1.0e-12
    assert abs(float(pvt.b_w(2.0e5)) - (1.0 / 1.02)) < 1.0e-12
    assert abs(float(pvt.viscosity_w(2.0e5)) - 1.2e-3) < 1.0e-15
    # no water table: linear default still 1.0
    dead = pvt_from_cfg({"pvt": "incompressible"}, p_init=1.0e5)
    assert abs(float(dead.b_w(2.0e5)) - 1.0) < 1.0e-15


def test_load_case_custom_pvt_tables(tmp_path) -> None:
    yml = tmp_path / "custom.yaml"
    yml.write_text(
        """
geometry: {size_m: [0.08, 0.08, 0.08]}
grid: {type: cartesian, spacing_m: 0.04}
physics:
  model: two_phase_immiscible
  capillary: none
  pvt:
    preset: incompressible
    p_tab: [1.0e5, 2.0e5]
    rs_tab: [0.0, 3.0]
    bo_tab: [1.05, 1.10]
  p_init: 1.5e5
  sw_init: 0.2
  dt_init: 1.0
  dt_max: 10.0
  transport: implicit
rock: {porosity: 0.3}
ports:
  - {name: INJ, role: injector, control: rate, x: 0.02, y: 0.04, z: 0.04, sw_inj: 1.0}
  - {name: PROD, role: producer, control: pressure, x: 0.06, y: 0.04, z: 0.04}
sensors: []
inverse: {parameterization: region, n_regions: 1}
experiment:
  history_end_s: 10
  controls:
    - {port: INJ, kind: rate, times: [0, 10], values: [1.0e-9, 1.0e-9]}
    - {port: PROD, kind: pressure, times: [0, 10], values: [1.5e5, 1.5e5]}
""",
        encoding="utf-8",
    )
    twin = load_case(yml)
    assert twin.physics.pvt.has_live_oil()
    assert abs(float(twin.physics.pvt.rs(2.0e5)) - 3.0) < 1.0e-12
    assert abs(float(twin.physics.relperm.mu_o) - float(twin.physics.pvt.mu_o)) < 1.0e-15
