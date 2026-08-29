"""Compositional digital twin: F_comp + H + LM on EXAMPLE fluid."""

from pathlib import Path
import json

import numpy as np
import pytest

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case
from reservoir_backend.physics.rock import Rock
from reservoir_backend.synthetic import evaluate_synthetic, make_two_layer_compositional


def test_observations_come_from_compositional_forward() -> None:
    case = make_two_layer_compositional(n_times=3, t_end=20.0, seed=1)
    assert case.twin.physics.model == "compositional"
    assert case.twin.physics.fluid is not None
    assert case.twin.experiment.observations
    assert any(o.kind == "bhp" for o in case.twin.experiment.observations)
    assert any(o.holdout for o in case.twin.experiment.observations)
    assert any(not o.holdout for o in case.twin.experiment.observations)
    st = case.twin.initial_state()
    assert st.moles is not None
    assert st.moles.shape[1] == 2
    assert st.sg is not None


def test_lm_recovers_layer_permeability_compositional() -> None:
    from reservoir_backend.physics.rock import Rock
    from reservoir_backend.twin.similarity import field_nrmse

    case = make_two_layer_compositional(
        n=(4, 3, 1),
        size_m=(4.0, 3.0, 1.0),
        n_times=3,
        t_end=16.0,
        seed=2,
        history_frac=0.85,
    )
    post = case.twin.calibrate()
    metrics = evaluate_synthetic(case, post)
    assert metrics["posterior_data_nrmse"] < metrics["prior_data_nrmse"], metrics
    assert metrics["posterior_logk_rmse"] < metrics["prior_logk_rmse"], metrics
    assert metrics["forward_match_nrmse"] < 7.0, metrics
    assert 1.5 <= metrics["contrast_post"] <= 25.0, metrics
    assert np.allclose(post.k, case.twin.parameterization.expand(post.theta))
    assert post.history.reports[-1].mass.relative_balance_error < 0.08
    phi = float(case.twin.parameterization.phi)
    times = np.array([16.0])
    post_traj = case.twin.simulate(Rock(post.k, np.full(case.grid.n_cells, phi)), t_end=16.0, report_times=times)
    p_true = case.p_true_end if case.p_true_end is not None else post_traj.states[-1].pressure
    assert field_nrmse(post_traj.states[-1].pressure, p_true) < 0.25


def test_comp_example_yaml_validate_and_simulate(tmp_path: Path) -> None:
    code = main(["validate", "examples/compositional/comp_example.yaml", "--output", str(tmp_path)])
    assert code == 0
    twin = load_case("examples/compositional/comp_example.yaml")
    assert twin.physics.model == "compositional"
    rock = Rock.uniform(twin.grid.n_cells, k=5.0e-13, phi=0.20)
    traj = twin.simulate(rock)
    assert traj.states[-1].moles is not None
    assert np.all(np.isfinite(traj.states[-1].pressure))


@pytest.mark.slow
def test_comp_apply_demo(tmp_path: Path) -> None:
    out = tmp_path / "comp"
    code = main(["apply", "examples/compositional/comp_example_apply.yaml", "--demo", "--output", str(out)])
    assert code == 0
    report = json.loads((out / "apply.json").read_text(encoding="utf-8"))
    assert report["n_theta"] == 2
    acc = report["acceptance"]
    assert acc["pass"] is True
    assert acc["p_field_nrmse"] < 0.25
    assert 3.0 <= acc["contrast_post"] <= 40.0


def test_refuse_invented_gem_card(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
geometry: {size_m: [2, 2, 1]}
grid: {type: cartesian, nx: 2, ny: 2, nz: 1}
physics:
  model: compositional
  fluid: {gem_deck: missing.gem}
rock: {porosity: 0.2}
ports:
  - {name: INJ, role: injector, control: rate, ijk: [[1,1,1]]}
  - {name: PROD, role: producer, control: pressure, ijk: [[2,2,1]]}
experiment:
  controls:
    - {port: INJ, kind: rate, times: [0, 1], values: [0.01, 0.01]}
    - {port: PROD, kind: pressure, times: [0, 1], values: [1.0e7, 1.0e7]}
""",
        encoding="utf-8",
    )
    try:
        load_case(p)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "gem_deck" in str(exc).lower() or "jiyang" in str(exc).lower()
