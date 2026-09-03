import json
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.io.case import _read_observation_csv
from reservoir_backend.twin.cmg_benchmark import (
    KPI_ORDER,
    PRESSURE_SPAN_FLOOR_PA,
    HiddenTruth,
    attach_cmg_observations,
    check_alignment,
    compare_fields,
    export_blocked_reason,
    find_gem_exe,
    forward_equivalence_report,
    improvement,
    invert_from_cmg_observations,
    load_alignment_spec,
    load_hidden_truth,
    nrmse_range,
    reconstruction_report,
    rmse,
    sample_observations_from_hidden,
    write_hidden_truth,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1


def test_kpi_order_puts_field_before_parameters() -> None:
    assert KPI_ORDER[0] == "pressure_field_nrmse"
    assert KPI_ORDER[-2] == "cf_rel_error"
    assert KPI_ORDER[-1] == "tmf_rel_error"


def test_nrmse_range_matches_plan_formula() -> None:
    pred = np.array([2.0, 3.0, 4.0])
    truth = np.array([1.0, 3.0, 5.0])
    expected = float(np.sqrt(np.mean((pred - truth) ** 2)) / (5.0 - 1.0))
    assert nrmse_range(pred, truth) == pytest.approx(expected)
    assert rmse(pred, pred) == pytest.approx(0.0)


def test_nrmse_span_floor_uses_instrument_sigma() -> None:
    pred = np.array([100.0, 100.0, 400.0])
    truth = np.array([0.0, 0.0, 300.0])
    raw = nrmse_range(pred, truth)
    floored = nrmse_range(pred, truth, span_floor=PRESSURE_SPAN_FLOOR_PA)
    assert raw > floored
    assert floored == pytest.approx(float(np.sqrt(np.mean((pred - truth) ** 2)) / 2.0e3))


def test_improvement_is_one_minus_error_ratio() -> None:
    assert improvement(0.20, 0.05) == pytest.approx(0.75)


def test_alignment_spec_matches_case_dev() -> None:
    rec = check_alignment()
    assert rec["ok"] is True
    assert rec["mismatches"] == []
    assert rec["n_cells"] == 32
    assert rec["theta_true"]["cf_m2"] == pytest.approx(1.0e-12)
    assert rec["theta_true"]["tmf_multiplier"] == pytest.approx(2.0)


def test_sensor_id_column_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "observations.csv"
    path.write_text(
        "time_s,sensor_id,kind,value,sigma\n0.5,P_f_in,pressure,1.2e7,2000\n",
        encoding="utf-8",
    )
    rows = _read_observation_csv(path)
    assert rows[0]["sensor"] == "P_f_in"


def _write_export(tmp_path: Path, *, with_hidden: bool = True) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    (export / "observations.csv").write_text(
        "time_s,sensor,kind,value,sigma,holdout\n"
        "0.5,P_f_in,pressure,1.2001e7,30,0\n"
        "0.5,P_f_out,pressure,1.1802e7,30,0\n"
        "60.0,P_f_in,pressure,1.201e7,30,0\n"
        "60.0,P_f_out,pressure,1.181e7,30,1\n",
        encoding="utf-8",
    )
    (export / "controls.csv").write_text(
        "time_s,port,kind,value\n"
        "0.0,INJ,rate,0.0003\n"
        "60.0,INJ,rate,0.0003\n"
        "0.0,INJ,composition,0.95\n"
        "60.0,INJ,composition,0.95\n"
        "0.0,PROD,pressure,11800000.0\n"
        "60.0,PROD,pressure,11800000.0\n",
        encoding="utf-8",
    )
    if with_hidden:
        hidden = export / "hidden"
        hidden.mkdir()
        times = np.array([0.5, 60.0])
        p = np.ones((2, 32), dtype=float) * 1.2e7
        p[1] += 1.0e3
        np.save(hidden / "pressure.npy", p)
        np.save(hidden / "sg.npy", np.zeros((2, 32)))
        (hidden / "meta.json").write_text(
            json.dumps({"nx": 4, "ny": 4, "nz": 2, "times_s": times.tolist(), "cell_order": "k_j_i"}),
            encoding="utf-8",
        )
    return export


def test_invert_refuses_hidden_truth(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    with pytest.raises(ValueError, match="must not receive CMG hidden truth"):
        invert_from_cmg_observations(export, hidden_dir=export / "hidden")
    with pytest.raises(ValueError, match="must not receive CMG hidden truth"):
        attach_cmg_observations(load_lab_v1(dev=True), export, hidden_dir=export / "hidden")


def test_attach_observations_does_not_open_hidden(tmp_path: Path) -> None:
    export = _write_export(tmp_path, with_hidden=True)
    twin = load_lab_v1(dev=True)
    attach_cmg_observations(twin, export)
    names = {o.sensor_name for o in twin.experiment.observations}
    assert "P_f_in" in names
    assert twin.experiment.observations  # invert inputs only


def test_hidden_loader_and_forward_gate_metrics(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    truth = load_hidden_truth(export / "hidden")
    ours = {"pressure": truth.pressure.copy(), "sg": np.zeros_like(truth.pressure)}
    rec = forward_equivalence_report(ours, truth)
    assert rec["metrics"]["pressure_field_nrmse"] == pytest.approx(0.0)
    assert rec["pass"] is True
    prior = {"pressure": truth.pressure + 2.0e5, "sg": ours["sg"]}
    post = {"pressure": truth.pressure + 5.0e4, "sg": ours["sg"]}
    score = reconstruction_report(
        prior=prior,
        posterior=post,
        truth=truth,
        phys_prior={"cf_m2": 3.0e-13, "tmf_multiplier": 1.0},
        phys_post={"cf_m2": 1.02e-12, "tmf_multiplier": 2.05},
        phys_true={"cf_m2": 1.0e-12, "tmf_multiplier": 2.0},
    )
    assert score["improvement_pressure"] > 0.0
    assert "cf_rel_error" in score["parameters"]


def test_export_blocked_without_observations(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert export_blocked_reason(empty) is not None


def test_compare_fields_shape_mismatch_raises() -> None:
    truth = HiddenTruth(times_s=np.array([1.0]), pressure=np.ones((1, 4)))
    with pytest.raises(ValueError, match="field size"):
        compare_fields({"pressure": np.ones((1, 3))}, truth)


def test_spec_points_at_case_dev() -> None:
    spec = load_alignment_spec()
    assert spec["our_case"].endswith("case_dev.yaml")
    assert spec["grid"] == [4, 4, 2]
    names = {row["name"] for row in spec["robustness_truths"]}
    assert names == {"T1", "T2", "T3", "T4"}


def test_hidden_time_slices_and_npz(tmp_path: Path) -> None:
    hidden = tmp_path / "cmg_truth"
    hidden.mkdir()
    p0 = np.ones(8)
    p1 = np.ones(8) * 2.0
    np.save(hidden / "pressure_t000.npy", p0)
    np.save(hidden / "pressure_t001.npy", p1)
    truth = load_hidden_truth(hidden)
    assert truth.pressure.shape == (2, 8)
    packed = tmp_path / "bundle"
    write_hidden_truth(packed, truth)
    again = load_hidden_truth(packed)
    assert again.pressure.shape == (2, 8)


def test_sample_observations_from_hidden_is_h_of_cmg(tmp_path: Path) -> None:
    export = _write_export(tmp_path)
    twin = load_lab_v1(dev=True)
    truth = load_hidden_truth(export / "hidden")
    series = sample_observations_from_hidden(twin, truth)
    assert series
    assert all(s.values.size == truth.times_s.size for s in series)


def test_parse_gem_out_pressure_planes(tmp_path: Path) -> None:
    from reservoir_backend.twin.cmg_benchmark import parse_gem_out_maps

    snippet = """
 Time = 6.9444444E-4          ********         2026 JAN.  1
                                                          Pressure  ( kpa)
  Fundamental Grid - Matrix
 Plane K = 1
      I =  1        2        3        4
 J=  1 11875.6  11875.0  11874.5  11873.9
 J=  2 11875.6  11875.1  11874.5  11873.9
 J=  3 11875.6  11875.1  11874.5  11874.0
 J=  4 11875.8  11875.1  11874.5  11874.0
 Plane K = 2
      I =  1        2        3        4
 J=  1 11875.8  11875.2  11874.6  11874.0
 J=  2 11875.7  11875.2  11874.6  11874.1
 J=  3 11875.9  11875.2  11874.6  11874.1
 J=  4 11875.9  11875.2  11874.6  11874.1
  Fundamental Grid - Fracture
 Plane K = 1
      I =  1        2        3        4
 J=  1 11801.6  11801.3  11800.7  11800.0
 J=  2 11801.8  11801.4  11800.8  11800.0
 J=  3 11802.0  11801.5  11800.8  11800.0
 J=  4 11802.1  11801.6  11800.9  11800.0
 Plane K = 2
      I =  1        2        3        4
 J=  1 11802.3  11801.8  11801.0  11800.2
 J=  2 11803.0  11802.1  11801.2  11800.2
 J=  3 11804.7  11802.7  11801.3  11800.2
 J=  4 11804.8  11802.9  11801.4  11800.2
 Time = 6.9444444E-4          Oil Saturation
  Fundamental Grid - Matrix                            All values are  0.007
  Fundamental Grid - Fracture                          All values are  0.007
"""
    path = tmp_path / "gem.out"
    path.write_text(snippet, encoding="utf-8")
    truth = parse_gem_out_maps(path)
    assert truth.pressure.shape == (1, 32)
    assert truth.pressure[0, 0] == pytest.approx(11801.6e3)
    assert truth.pressure_matrix is not None
    assert truth.pressure_matrix[0, 0] == pytest.approx(11875.6e3)
    assert truth.times_s[0] == pytest.approx(60.0, rel=1e-3)
    assert truth.so is not None
    assert truth.so[0, 0] == pytest.approx(0.007)


def test_spec_fluid_uses_published_opm_vcrit() -> None:
    spec = load_alignment_spec()
    np.testing.assert_allclose(spec["fluid"]["vcrit_m3_kmol"], [0.09863, 0.60980])
    assert spec["fluid"]["kij"] == pytest.approx(0.049)
    assert spec["fluid"]["pvc3"] == pytest.approx(0.0)
    assert spec["fluid"]["gem_compname"] == ["METHANE", "DECANE"]


def test_gem_deck_uses_user_names_and_explicit_bin() -> None:
    deck = Path(__file__).resolve().parents[2] / "examples" / "lab_v1" / "cmg_gem" / "lab_v1_dev.dat"
    text = deck.read_text(encoding="utf-8")
    assert "*COMPNAME 'METHANE' 'DECANE'" in text
    assert "*HCFLAG 0 0" in text
    assert "*BIN 0.049" in text
    assert "*VCRIT 0.09863 0.60980" in text
    assert "*OMEGA 0.457235530 0.457235530" in text
    assert "*TRANSFER 0" in text
    assert "*SIGMAMF *CON 80" in text
    assert "*KAVMF *CON 1.01325" in text
    assert "*GRID *VARI 4 4 2" in text
    assert "*DEPTH-TOP *KVAR 0.0 0.0" in text
    assert "*VISCOR *MIX" in text
    assert "*VISCOSITY 0.020 0.30" in text
    assert "*PERF *WI 1" in text
    assert "304.08" in text


def test_lab_v1_relperm_matches_gem_linear_sgt() -> None:
    twin = load_lab_v1(dev=True)
    fl = twin.physics.fluid
    assert fl is not None
    assert float(fl.sorg) == pytest.approx(0.0)
    assert float(fl.sgr) == pytest.approx(0.0)
    assert float(fl.no) == pytest.approx(1.0)
    assert float(fl.ng) == pytest.approx(1.0)


def test_stg_surface_rate_converts_to_mol() -> None:
    from reservoir_backend.comp.wells import surface_gas_rate_to_mol
    from reservoir_backend.twin.cmg_benchmark import ensure_molar_injector_rate

    twin = load_lab_v1(dev=True)
    q_mol = surface_gas_rate_to_mol(3.0e-4, twin.physics.fluid)
    assert q_mol == pytest.approx(0.0127, rel=0.15)
    converted = ensure_molar_injector_rate(twin)
    assert converted == pytest.approx(q_mol)
    rates = [c.values[0] for c in twin.experiment.controls if c.port_name == "INJ" and c.kind == "rate"]
    assert rates and rates[0] == pytest.approx(q_mol)
    again = ensure_molar_injector_rate(twin)
    assert again == pytest.approx(q_mol)
    rates2 = [c.values[0] for c in twin.experiment.controls if c.port_name == "INJ" and c.kind == "rate"]
    assert rates2[0] == pytest.approx(q_mol)


def test_our_init_flash_is_two_phase() -> None:
    from reservoir_backend.twin.cmg_benchmark import our_init_flash

    rec = our_init_flash()
    assert 0.30 < rec["sg"] < 0.42
    assert rec["so"] == pytest.approx(1.0 - rec["sg"])


def test_parse_gem_init_fluid_reads_average_saturations(tmp_path: Path) -> None:
    from reservoir_backend.twin.cmg_benchmark import parse_gem_init_fluid

    path = tmp_path / "init.out"
    path.write_text(
        "\n".join(
            [
                "  * Initial Reservoir Conditions and Fluid Properties *",
                "      Ave. oil saturation                       = 0.64425",
                "      Ave. gas saturation                       = 0.35575",
                "      Ave. gas phase Z factor                   = 0.72000",
                "          METHANE                               = 1.06009E+01   1.70064E-01",
                "          DECANE                                = 8.67343E+00   1.23407E+00",
                " TIME: 0.0000  days",
            ]
        ),
        encoding="utf-8",
    )
    rec = parse_gem_init_fluid(path)
    assert rec["sg"] == pytest.approx(0.35575)
    assert rec["so"] == pytest.approx(0.64425)
    assert rec["z_c1"] == pytest.approx(1.06009e1 / (1.06009e1 + 8.67343e0))


def test_find_gem_exe_on_this_machine() -> None:
    exe = find_gem_exe()
    if exe is not None:
        assert exe.is_file()


def test_forward_gate_wiring_and_missing_export(tmp_path: Path) -> None:
    from scripts.lab_v1_cmg_forward_gate import main

    assert main(["--wiring", "--out", str(tmp_path / "wire")]) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--export", str(empty), "--out", str(tmp_path / "out")]) == 2
