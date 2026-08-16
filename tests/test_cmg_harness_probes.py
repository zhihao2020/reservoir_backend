"""Adapter / catalog checks that do not run IMEX or ensemble invert."""

import numpy as np

from reservoir_backend.validation.cmg_harness.adapter import (
    controls_from_truth,
    grid_from_truth,
    load_truth,
    our_k_from_cmg,
    parameterization_from_truth,
    ports_from_truth,
    sensors_for_spec,
)
from reservoir_backend.validation.cmg_harness.catalog import get_case, list_cases
from reservoir_backend.validation.cmg_harness.search import neighbors


def test_catalog_lists_ready_and_blocks_shale() -> None:
    ready = {c.id: c for c in list_cases()}
    assert "lab_layers" in ready and "fivespot" in ready and "fault" in ready
    assert "channel" in ready
    assert ready["fivespot"].history_days[0] >= 1.0
    assert ready["channel"].dt_max_s >= 86400.0
    assert get_case("shale_s1").status == "unsupported"
    assert get_case("lab_box").status == "need_imex"


def test_lab_layers_perfs_match_truth() -> None:
    spec = get_case("lab_layers")
    truth = load_truth(spec)
    grid = grid_from_truth(truth)
    ports = {p.name: p for p in ports_from_truth(grid, truth)}
    inj = ports["INJ"]
    prod = ports["PROD"]
    inj_ijk = [grid.ijk(int(c)) for c in inj.cell_ids]
    prod_ijk = [grid.ijk(int(c)) for c in prod.cell_ids]
    assert all(i == 0 and j == 3 for i, j, _k in inj_ijk)
    assert {k for _i, _j, k in inj_ijk} == {0, 1, 2, 3}
    assert all(i == 11 and j == 3 for i, j, _k in prod_ijk)
    assert {k for _i, _j, k in prod_ijk} == {0, 1, 2, 3, 4, 5}
    assert inj.control == "pressure" and prod.control == "pressure"


def test_cmg_k_mapping_top_and_bottom() -> None:
    assert our_k_from_cmg(6, 6, k1_top=True) == 0
    assert our_k_from_cmg(6, 1, k1_top=True) == 5
    assert our_k_from_cmg(6, 1, k1_top=False) == 0
    assert our_k_from_cmg(6, 6, k1_top=False) == 5


def test_fivespot_sensors_follow_well_pattern() -> None:
    spec = get_case("fivespot")
    truth = load_truth(spec)
    grid = grid_from_truth(truth)
    sensors, hold = sensors_for_spec(spec, grid)
    names = {s.name for s in sensors}
    assert "P_ctr" in names and "S_ctr" in names
    assert "P_sw" in names and "Pbar" in names
    assert "P_ne" in hold
    assert "Pbar" not in hold


def test_fivespot_channel_is_two_region() -> None:
    spec = get_case("fivespot")
    truth = load_truth(spec)
    grid = grid_from_truth(truth)
    param = parameterization_from_truth(spec, grid, truth)
    assert param.n_params == 2
    from reservoir_backend.inverse.parameterization import ContrastParameterization

    assert isinstance(param, ContrastParameterization)
    ports = ports_from_truth(grid, truth)
    assert len(ports) == 5
    assert sum(p.role == "injector" for p in ports) == 4


def test_fault_controls_come_from_imex_liquid_rates() -> None:
    spec = get_case("fault")
    if not spec.out_path.is_file():
        return
    truth = load_truth(spec)
    grid = grid_from_truth(truth)
    ports = ports_from_truth(grid, truth)
    ctrls = controls_from_truth(truth, ports, np.array([0.0, 86400.0]), out_path=spec.out_path)
    by = {(c.port_name, c.kind): c for c in ctrls}
    assert ("INJ", "rate") in by and float(by[("INJ", "rate")].values[0]) > 1.0e-3
    assert ("PROD", "rate") in by and float(by[("PROD", "rate")].values[0]) < 0.0


def test_search_neighbors_change_one_axis() -> None:
    seed = {"algorithm": "esmda", "n_ensemble": 12, "n_assimilations": 4, "prior_std": 0.8, "inflation": 1.02}
    kids = neighbors(seed, "n_ensemble")
    assert kids
    for c in kids:
        assert c["algorithm"] == "esmda"
        assert c["n_ensemble"] != 12
        assert c["prior_std"] == 0.8
