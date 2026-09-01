import math
from types import SimpleNamespace

import pytest

from reservoir_backend.twin.experiment_design import (
    D_CF_MIN,
    H_KINDS,
    SIGMA_P,
    STEADY_DP_PA,
    YAML_PATH,
    Instrument,
    cf_detectability_bound,
    default_candidates,
    independent_samples,
    instrument_from_mapping,
    instrument_sensors,
    load_experiment_design_yaml,
    select_designs,
    trajectory_min_dt,
    two_gauge_delta_sigma,
)


def test_yaml_loads_plan_catalog() -> None:
    env, designs = load_experiment_design_yaml(YAML_PATH)
    names = {d.name for d in designs}
    assert names >= {"constant", "long_constant", "pulse_1", "pulse_rest", "multistep"}
    assert env.q_max_m3_s == pytest.approx(1.67e-6)
    by_name = {d.name: d for d in designs}
    assert by_name["long_constant"].evaluate is False
    assert by_name["pulse_rest_dp"].instrument.h == "dp_transducer"
    assert by_name["pulse_rest_dp"].instrument.dp_sigma_pa == pytest.approx(200.0)
    assert by_name["constant"].instrument.h == "bulk_gauges"
    runnable = select_designs(designs)
    assert {d.name for d in runnable} == {
        "constant",
        "constant_tapped",
        "pulse_rest",
        "pulse_rest_tapped",
        "pulse_rest_dp",
        "legacy_m1b_rate",
    }


def test_yaml_controls_stages_form(tmp_path) -> None:
    path = tmp_path / "one.yaml"
    path.write_text(
        "\n".join(
            [
                "instrument:",
                "  pressure_sigma_pa: 2000",
                "  saturation_sigma: 0.03",
                "controls:",
                "  stages:",
                "    - duration_s: 10",
                "      q_inj: 1.67e-6",
                "    - duration_s: 20",
                "      q_inj: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    env, designs = load_experiment_design_yaml(path)
    assert len(designs) == 1
    assert designs[0].stages[0].duration_s == pytest.approx(10.0)
    assert designs[0].stages[1].q_inj == pytest.approx(0.0)
    assert env.pv_max == pytest.approx(3.0)


def test_h_kinds_are_labeled() -> None:
    for h in H_KINDS:
        kwargs = {"h": h}
        if h == "dp_transducer":
            kwargs["dp_sigma_pa"] = 200.0
        sensors = instrument_sensors(Instrument(**kwargs))
        names = {s.name for s in sensors}
        media = {s.medium for s in sensors if s.kind == "pressure"}
        assert "P_in" in names and "P_out" in names
        if h == "tapped_channel":
            assert media == {"fracture"}
        else:
            assert media == {"bulk"}
    with pytest.raises(ValueError, match="dp_transducer"):
        instrument_from_mapping({"h": "dp_transducer"})
    with pytest.raises(ValueError, match="unknown H"):
        instrument_from_mapping({"h": "fracture_matrix_oracles"})


def test_catalog_is_discrete_argmax_over_plan_shapes() -> None:
    names = [d.name for d in default_candidates()]
    for required in ("constant", "long_constant", "pulse_1", "pulse_rest", "multistep"):
        assert required in names
    assert all(d.instrument.h in H_KINDS for d in default_candidates())


def test_steady_cf_bound_below_gate_at_2kpa() -> None:
    assert cf_detectability_bound(STEADY_DP_PA, SIGMA_P, n_indep=1.0) < D_CF_MIN
    n = independent_samples(1800.0, 5.0)
    assert cf_detectability_bound(STEADY_DP_PA, SIGMA_P, n_indep=n) < D_CF_MIN
    assert two_gauge_delta_sigma(SIGMA_P) == pytest.approx(SIGMA_P * (2.0**0.5))


def test_trajectory_min_dt_reads_solver_reports() -> None:
    reports = [SimpleNamespace(dt=0.5), SimpleNamespace(dt=0.125), SimpleNamespace(dt=1.0)]
    assert trajectory_min_dt(SimpleNamespace(reports=reports)) == pytest.approx(0.125)
    assert math.isnan(trajectory_min_dt(SimpleNamespace(reports=[])))
