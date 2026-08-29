"""Jiyang-pattern compositional case loads GEM well-history H. Does not run GEM."""

from pathlib import Path

from reservoir_backend.io.case import load_case


def test_jiyang_co2_hnp_case_wires_card_wells_and_bhp_obs() -> None:
    twin = load_case("examples/jiyang/jiyang_co2_hnp.yaml")
    assert twin.physics.model == "compositional"
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.eos.nc == 3
    assert twin.physics.fluid.eos.names[0] == "CO2"
    assert [p.name for p in twin.ports] == ["INJ", "P1", "P2", "P3", "P4"]
    assert all(p.use_productivity and p.axis == "i" and p.rw_m == 0.10 for p in twin.ports)
    assert twin.grid.n_cells == 21 * 21 * 5
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "ContrastParameterization"
    assert twin.physics.fluid.has_water is True
    assert 0.2 < float(twin.physics.fluid.sw_init) < 0.3
    kinds = {(o.sensor_name, o.kind, o.holdout) for o in twin.experiment.observations}
    assert ("INJ_bhp", "bhp", False) in kinds
    assert ("P1_q_oil", "q_oil", False) in kinds
    assert ("P4_q_oil", "q_oil", True) in kinds
    assert not any(o.kind == "bhp" and o.sensor_name.startswith("P") and not o.holdout for o in twin.experiment.observations)
    assert Path("examples/jiyang/fixtures/jiyang_co2_hnp_obs.csv").is_file()
