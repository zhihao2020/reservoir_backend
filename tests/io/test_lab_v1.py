from pathlib import Path

import pytest

from reservoir_backend.io.case import load_case
from reservoir_backend.twin.lab_v1 import spatial_holdout, zone_of_x


def test_load_lab_v1_dev_face_ports_and_sensors() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    assert twin.uses_dpdp()
    assert twin.inverse.algorithm == "esmda"
    assert twin.parameterization.n_params == 2
    assert twin.inverse.ensemble_size == 8
    assert twin.grid.n_cells == 4 * 4 * 2
    assert twin.ports[0].cell_ids.size == twin.grid.ny * twin.grid.nz
    assert twin.ports[1].cell_ids.size == twin.grid.ny * twin.grid.nz
    assert twin.ports[0].name == "INJ"
    assert twin.ports[0].control == "rate"
    assert twin.ports[1].control == "pressure"
    kinds = {s.kind for s in twin.experiment.sensors}
    media = {s.medium for s in twin.experiment.sensors}
    assert "pressure" in kinds
    assert "gas_saturation" in kinds
    assert "saturation" not in kinds
    assert "fracture" in media and "matrix" in media and "bulk" in media
    for s in twin.experiment.sensors:
        assert s.sigma > 0.0
        if s.kind == "pressure" and s.medium == "fracture":
            assert s.sigma < 200.0
        if s.kind == "pressure" and s.medium == "matrix":
            assert s.sigma >= 1.0e3
    assert twin.inverse.assimilation_steps == 5
    assert twin.inverse.outlier_nsigma == 120.0


def test_load_lab_v1_product_spec_is_30_cubed() -> None:
    twin = load_case("examples/lab_v1/case.yaml")
    assert twin.grid.nx == 30
    assert twin.grid.n_cells == 27_000
    assert twin.ports[0].cell_ids.size == 900
    assert twin.inverse.ensemble_size == 12
    assert twin.inverse.algorithm == "esmda"
    assert twin.parameterization.n_params == 2


def test_new_sensor_csv_requires_sigma(tmp_path: Path) -> None:
    from reservoir_backend.io.case import _read_sensors_csv

    path = tmp_path / "sensors.csv"
    path.write_text(
        "sensor_id,kind,x_m,y_m,z_m,continuum\nP001,pressure,0.05,0.1,0.15,bulk\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sigma"):
        _read_sensors_csv(path)


def test_spatial_holdout_covers_three_zones() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    held = spatial_holdout(list(twin.experiment.sensors), frac=0.25, seed=3)
    zones = {zone_of_x(s.x) for s in twin.experiment.sensors if s.name in held}
    remaining = {s.name for s in twin.experiment.sensors} - held
    assert remaining
    assert zones <= {"inlet", "middle", "outlet"}
