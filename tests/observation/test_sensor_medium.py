import numpy as np
import pytest

from reservoir_backend.domain.types import Sensor, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.observation.operator import ObservationOperator


def test_medium_fracture_matrix_bulk() -> None:
    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    op = ObservationOperator(grid, [])
    st = State(
        pressure=np.array([1.0e7]),
        sw=np.array([0.2]),
        sg=np.array([0.1]),
        pressure_matrix=np.array([1.2e7]),
        sw_matrix=np.array([0.4]),
        sg_matrix=np.array([0.0]),
        phi_fracture=np.array([0.02]),
        phi_matrix=np.array([0.08]),
    )
    sf = Sensor("pf", "pressure", 0.05, 0.05, 0.05, medium="fracture")
    sm = Sensor("pm", "pressure", 0.05, 0.05, 0.05, medium="matrix")
    sb = Sensor("swb", "saturation", 0.05, 0.05, 0.05, medium="bulk")
    assert op.sample(sf, st) == pytest.approx(1.0e7)
    assert op.sample(sm, st) == pytest.approx(1.2e7)
    bulk_sw = (0.02 * 0.2 + 0.08 * 0.4) / 0.10
    assert op.sample(sb, st) == pytest.approx(bulk_sw)
