"""CMG .out grid parser smoke tests (uses existing validation artifacts if present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.cmg_io.grid_parse import parse_grid_series

CHANNEL_OUT = Path("validation/cmg_channel_3d/mxspr006_channel.out")


@pytest.mark.skipif(not CHANNEL_OUT.is_file(), reason="CMG channel .out not present")
def test_parse_sw_and_pressure_channel() -> None:
    sw = parse_grid_series(CHANNEL_OUT, field="sw", nx=7, ny=7, nz=5)
    pr = parse_grid_series(CHANNEL_OUT, field="pressure", nx=7, ny=7, nz=5)
    assert len(sw) >= 2
    assert len(pr) >= 2
    _, sw_last = sw[-1]
    _, p_last = pr[-1]
    assert sw_last.shape == (5, 7, 7)
    assert 0.0 <= float(np.nanmin(sw_last)) <= float(np.nanmax(sw_last)) <= 1.0
    # after t=0, reservoir pressure should vary and be ~thousands of psi
    assert float(np.nanmax(p_last)) > 1000.0
    assert float(np.nanstd(p_last)) > 1.0
