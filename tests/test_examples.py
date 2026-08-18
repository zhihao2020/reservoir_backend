from pathlib import Path

import pytest
import yaml

from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case

TWO_LAYER = Path("examples/two_layer")
CHANNEL = Path("examples/channel")


def test_two_layer_self_check_case_has_no_observations() -> None:
    twin = load_case(TWO_LAYER / "case.yaml")
    assert not twin.experiment.observations
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "RegionParameterization"
    assert all(abs(s.probe_diameter_m - 0.006) < 1e-12 for s in twin.experiment.sensors)


def test_two_layer_from_csv_loads_all_sensors() -> None:
    twin = load_case(TWO_LAYER / "case_from_csv.yaml")
    names = {o.sensor_name for o in twin.experiment.observations}
    assert names == {s.name for s in twin.experiment.sensors}
    assert any(o.holdout for o in twin.experiment.observations)
    assert max(float(o.times_s.max()) for o in twin.experiment.observations) > float(twin.experiment.history_end_s)


def test_two_layer_lab_units_csv_matches_si(tmp_path: Path) -> None:
    si = load_case(TWO_LAYER / "case_from_csv.yaml")
    cfg = yaml.safe_load((TWO_LAYER / "case_from_csv.yaml").read_text(encoding="utf-8"))
    cfg["experiment"]["observations"] = str((TWO_LAYER / "observations_kpa_min.csv").resolve())
    path = tmp_path / "lab_units.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    lab = load_case(path)
    by_si = {o.sensor_name: o for o in si.experiment.observations}
    by_lab = {o.sensor_name: o for o in lab.experiment.observations}
    assert by_si.keys() == by_lab.keys()
    for name in by_si:
        assert by_si[name].times_s == pytest.approx(by_lab[name].times_s, rel=0, abs=1e-6)
        assert by_si[name].values == pytest.approx(by_lab[name].values, rel=0, abs=1e-4)
        assert by_si[name].sigma == pytest.approx(by_lab[name].sigma, rel=0, abs=1e-6)


def test_channel_cases_load() -> None:
    demo = load_case(CHANNEL / "case.yaml")
    assert not demo.experiment.observations
    assert type(demo.parameterization).__name__ == "ContrastParameterization"
    csv_case = load_case(CHANNEL / "case_from_csv.yaml")
    names = {o.sensor_name for o in csv_case.experiment.observations}
    assert names == {s.name for s in csv_case.experiment.sensors}


def test_apply_without_demo_or_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no observations"):
        main(["apply", str(TWO_LAYER / "case.yaml"), "--output", str(tmp_path / "out")])
