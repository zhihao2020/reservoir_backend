import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.domain.types import ControlSeries, Experiment, Sensor
from reservoir_backend.twin.field import attach_probe_series, pressure_field, step_pressure
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec, stack_observations


def _tiny_twin() -> DigitalTwin:
    grid = CartesianGrid.uniform((0.12, 0.06, 0.06), 0.03)
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.015, 0.03, 0.03))
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.105, 0.03, 0.03))
    times = np.array([2.0, 4.0, 6.0])
    controls = [
        ControlSeries("INJ", "rate", times, np.full(times.size, 1.0e-8)),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.95)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, 1.18e7)),
    ]
    sensors = [Sensor("P1", "pressure", 0.04, 0.03, 0.03, sigma=2.0e3, medium="fracture")]
    experiment = Experiment(
        size_m=grid.size_m(),
        sensors=sensors,
        controls=controls,
        observations=[],
    )
    fluid = fluid_from_name("example")
    physics = PhysicsSpec(
        p_init=1.2e7,
        dt_init=0.5,
        dt_max=2.0,
        max_steps=80,
        model="compositional_dpdp",
        fluid=fluid,
        k_matrix_m2=1.0e-15,
        phi_fracture=0.02,
    )
    cond = FractureConductivityModel(
        n_cells=grid.n_cells, fracture_mask=np.ones(grid.n_cells, dtype=bool), k_matrix_m2=1.0e-15
    )
    param = LogConductivityParameterization(n_zones=1, phi=0.08, conductivity=cond)
    return DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        physics,
        param,
        inverse=InverseSpec(max_iter=4),
    )


def test_pressure_field_shape_from_tiny_series() -> None:
    twin = _tiny_twin()
    k = np.full(twin.grid.n_cells, 1.0e-12)
    times = np.array([2.0, 4.0, 6.0])
    series = {
        "times_s": times,
        "values": np.array([[1.10e5], [1.12e5], [1.13e5]]),
        "sigma": 2.0e3,
    }
    out = pressure_field(
        twin,
        probes=[("P1", 0.04, 0.03, 0.03)],
        series=series,
        k=k,
        report_times=times,
    )
    assert out.pressure.shape == (times.size, twin.grid.n_cells)
    assert out.times_s.shape == (times.size,)
    assert out.k.shape == (twin.grid.n_cells,)
    assert np.all(np.isfinite(out.pressure))
    st = step_pressure(twin, k, dt=1.0)
    assert st.pressure.shape == (twin.grid.n_cells,)
    assert float(st.time_s) > 0.0
    assert np.all(np.isfinite(st.pressure))

def test_pressure_field_csv_ixyzp_known_k(tmp_path) -> None:
    twin = _tiny_twin()
    k = np.full(twin.grid.n_cells, 1.0e-12)
    times = np.array([2.0, 4.0, 6.0])
    out = pressure_field(twin, k=k, report_times=times, output=tmp_path)
    path = tmp_path / "field.csv"
    assert path.is_file()
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    for name in ("I", "x", "y", "z", "p"):
        assert name in cols
    n_cells = twin.grid.n_cells
    assert len(rows) in (n_cells, times.size * n_cells)
    assert (tmp_path / "pressure.npy").is_file()
    assert (tmp_path / "k.npy").is_file()
    k_saved = np.load(tmp_path / "k.npy")
    assert k_saved.shape == (n_cells,)
    assert np.allclose(k_saved, k)
    xyz = twin.grid.cell_centers()
    stacked = len(rows) == times.size * n_cells
    for n, row in enumerate(rows):
        ic = int(row["I"])
        it = n // n_cells if stacked else -1
        assert 0 <= ic < n_cells
        assert abs(float(row["x"]) - xyz[ic, 0]) < 1.0e-12
        assert abs(float(row["y"]) - xyz[ic, 1]) < 1.0e-12
        assert abs(float(row["z"]) - xyz[ic, 2]) < 1.0e-12
        assert abs(float(row["p"]) - float(out.pressure[it, ic])) < 1.0e-8
        assert abs(float(row["k"]) - float(k[ic])) < 1.0e-20
    assert out.sw is not None
    assert out.sw.shape == (times.size, n_cells)
    assert "sw" in cols
    if out.so is not None:
        assert "so" in cols
    if out.sg is not None:
        assert "sg" in cols
    assert out.phi == 0.08
    assert "phi" in cols

def test_known_sw_series_used_not_ignored() -> None:
    """Mixed p + known-Sw series enter the invert data vector and H samples Sw."""
    twin = _tiny_twin()
    times = np.array([2.0, 4.0])
    attach_probe_series(
        twin,
        probes=[
            {"name": "P1", "kind": "pressure", "x": 0.04, "y": 0.03, "z": 0.03},
            {"name": "S1", "kind": "sw", "x": 0.08, "y": 0.03, "z": 0.03},
        ],
        series={
            "P1": {"times_s": times, "values": np.array([1.10e5, 1.12e5]), "sigma": 2.0e3},
            "S1": {"times_s": times, "values": np.array([0.41, 0.55]), "sigma": 0.04},
        },
    )
    assert {s.kind for s in twin.experiment.sensors} == {"pressure", "saturation"}
    assert {o.kind for o in twin.experiment.observations} == {"pressure", "saturation"}

    d = stack_observations(twin.experiment.assimilate_observations())
    assert "saturation" in d.kinds and "pressure" in d.kinds
    sw_idx = [i for i, k in enumerate(d.kinds) if k == "saturation"]
    p_idx = [i for i, k in enumerate(d.kinds) if k == "pressure"]
    assert d.values.size == 4
    assert set(np.round(d.values[sw_idx], 5)) == {0.41, 0.55}
    assert float(np.max(d.values[sw_idx])) < 2.0
    assert float(np.min(d.values[p_idx])) > 1.0e4

    theta = np.full(twin.parameterization.n_params, float(np.log(1.0e-12)))
    pred = twin._forward_vector(theta, list(twin.experiment.observations))
    assert pred.size == d.values.size
    assert np.all((pred[sw_idx] >= 0.0) & (pred[sw_idx] <= 1.0))
    assert np.all(pred[p_idx] > 1.0e4)
    # Ignored-as-pressure would put ~1e5 in the Sw slots.
    assert float(np.max(pred[sw_idx])) < 1.5
