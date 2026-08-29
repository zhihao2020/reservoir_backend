"""Run report file outputs."""

from __future__ import annotations

from pathlib import Path

from reservoir_backend.cli.reporting import emit_invert_artifacts
from reservoir_backend.synthetic import make_two_layer_waterflood


def test_emit_invert_writes_json_and_check83(tmp_path: Path) -> None:
    case = make_two_layer_waterflood(n_times=3, t_end=200.0)
    twin = case.twin
    twin.inverse.max_iter = 2
    twin.inverse.post_ensemble_enabled = True
    twin.inverse.post_ensemble_ne = 4
    post = twin.calibrate(max_iter=2)
    emit_invert_artifacts(twin, post, tmp_path, fields={"k": post.k})
    assert (tmp_path / "invert.json").is_file()
    assert (tmp_path / "check83.json").is_file()
    assert (tmp_path / "k_std.npy").is_file()
