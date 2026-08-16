import numpy as np

from reservoir_backend.domain.types import Sensor, State, column_sensors
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.observation.operator import ObservationOperator


def test_trilinear_linear_field_off_center() -> None:
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 0.25)
    xyz = grid.cell_centers()
    field = xyz[:, 0] + 2.0 * xyz[:, 1] + 3.0 * xyz[:, 2]
    op = ObservationOperator(grid, [])
    point = (0.37, 0.41, 0.28)
    pred = op.sample_field(Sensor("p", "pressure", *point), field)
    truth = point[0] + 2.0 * point[1] + 3.0 * point[2]
    assert abs(pred - truth) < 1.0e-12


def test_volume_sensor_averages_block() -> None:
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 0.25)
    field = np.arange(grid.n_cells, dtype=float)
    op = ObservationOperator(grid, [])
    sensor = Sensor("v", "pressure", 0.5, 0.5, 0.5, volume_m3=0.25**3 * 8)
    pred = op.sample_field(sensor, field)
    assert np.isfinite(pred)


def test_state_sample_pressure_and_sw() -> None:
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    state = State(
        pressure=np.full(grid.n_cells, 1.2e5),
        sw=np.full(grid.n_cells, 0.33),
    )
    op = ObservationOperator(grid, [])
    p = op.sample(Sensor("p", "pressure", 0.15, 0.15, 0.15), state)
    s = op.sample(Sensor("s", "saturation", 0.15, 0.15, 0.15), state)
    assert abs(p - 1.2e5) < 1.0e-9
    assert abs(s - 0.33) < 1.0e-12


def test_column_sensors_sample_different_depths() -> None:
    grid = CartesianGrid.uniform((1.0, 1.0, 1.0), 0.25)
    xyz = grid.cell_centers()
    field = xyz[:, 2]
    gauges = column_sensors("P", "pressure", 0.5, 0.5, [0.20, 0.80], sigma=1.0, labels=("bot", "top"))
    assert [s.name for s in gauges] == ["P_bot", "P_top"]
    assert gauges[0].z != gauges[1].z
    op = ObservationOperator(grid, gauges)
    bot = op.sample_field(gauges[0], field)
    top = op.sample_field(gauges[1], field)
    assert top > bot + 0.4
