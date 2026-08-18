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
    so = op.sample(Sensor("so", "oil_saturation", 0.15, 0.15, 0.15), state)
    sg = op.sample(Sensor("sg", "gas_saturation", 0.15, 0.15, 0.15), state)
    assert abs(p - 1.2e5) < 1.0e-9
    assert abs(s - 0.33) < 1.0e-12
    assert abs(so - 0.67) < 1.0e-12
    assert abs(sg - 0.0) < 1.0e-12


def test_inflate_sigma_uses_clean_residual() -> None:
    from reservoir_backend.observation.error import inflate_sigma

    extra, sig = inflate_sigma([1.0, 2.0], [1.0, 2.0], 0.1)
    assert extra == 0.0
    assert abs(sig - 0.1) < 1.0e-12
    extra, sig = inflate_sigma([3.0, 3.0], [1.0, 1.0], 0.0)
    assert abs(extra - 2.0) < 1.0e-12
    assert abs(sig - 2.0) < 1.0e-12
    # Noisy "clean" would inflate R by the instrument draw — caller must not do that.
    noisy = [1.0, 1.4]
    extra_wrong, _ = inflate_sigma([1.0, 1.0], noisy, 0.4)
    extra_right, _ = inflate_sigma([1.0, 1.0], [1.0, 1.0], 0.4)
    assert extra_wrong > extra_right
    extra_capped, _ = inflate_sigma([3.0, 3.0], [1.0, 1.0], 0.1, extra_cap=0.2)
    assert abs(extra_capped - 0.2) < 1.0e-12


def test_state_sample_three_phase_saturations() -> None:
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    state = State(
        pressure=np.full(grid.n_cells, 1.2e5),
        sw=np.full(grid.n_cells, 0.30),
        sg=np.full(grid.n_cells, 0.10),
    )
    op = ObservationOperator(grid, [])
    assert abs(op.sample(Sensor("so", "so", 0.15, 0.15, 0.15), state) - 0.60) < 1.0e-12
    assert abs(op.sample(Sensor("sg", "sg", 0.15, 0.15, 0.15), state) - 0.10) < 1.0e-12


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


def test_six_mm_probe_matches_center_on_linear_field() -> None:
    """6 mm sphere average of a linear field equals the center (even off cell)."""
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.01)
    xyz = grid.cell_centers()
    field = xyz[:, 0] + 2.0 * xyz[:, 1] + 3.0 * xyz[:, 2]
    op = ObservationOperator(grid, [])
    point = (0.083, 0.151, 0.047)
    sensor = Sensor("p6", "pressure", *point, probe_diameter_m=0.006)
    pred = op.sample_field(sensor, field)
    truth = point[0] + 2.0 * point[1] + 3.0 * point[2]
    assert abs(pred - truth) < 2.0e-4


def test_lab_30cm_is_two_region_with_6mm_probes() -> None:
    from reservoir_backend.io.case import load_case

    twin = load_case("config/lab_30cm.yaml")
    assert twin.parameterization.n_params == 2
    assert all(abs(s.probe_diameter_m - 0.006) < 1e-12 for s in twin.experiment.sensors)
