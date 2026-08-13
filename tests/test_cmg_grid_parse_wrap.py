"""Wide-grid IMEX Plane-K wrap (I=1..14 then I=15..n)."""

from __future__ import annotations

import sys
from pathlib import Path

_VAL = Path(__file__).resolve().parents[1] / "black_oil" / "validation"
if str(_VAL) not in sys.path:
    sys.path.insert(0, str(_VAL))
from cmg_io.grid_parse import _parse_plane_block


def test_wrapped_i_columns_assign_correct_indices() -> None:
    chunk = """
 Plane K = 1
      I =  1        2        3
 J=  1   0.10     0.20     0.30
 J=  2   0.11     0.21     0.31

      I =  4        5
 J=  1   0.40     0.50
 J=  2   0.41     0.51
"""
    arr = _parse_plane_block(chunk, nx=5, ny=2, nz=1)
    assert arr[0, 0, 0] == 0.10
    assert arr[0, 0, 2] == 0.30
    assert arr[0, 0, 3] == 0.40
    assert arr[0, 0, 4] == 0.50
    assert arr[0, 1, 4] == 0.51
