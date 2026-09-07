"""check.txt §83 report keys."""

from __future__ import annotations

from reservoir_backend.synthetic import make_lab_v1_face_twin
from reservoir_backend.twin.acceptance import build_check83_report


def test_check83_has_twelve_questions() -> None:
    case = make_lab_v1_face_twin(n_times=3, t_end=2.0, with_saturation=False)
    twin = case.twin
    twin.inverse.max_iter = 3
    twin.inverse.post_ensemble_enabled = True
    twin.inverse.post_ensemble_ne = 4
    post = twin.calibrate(max_iter=3)
    report = build_check83_report(twin, post)
    keys = [k for k in report if k.startswith("q")]
    assert len(keys) == 12
    assert "summary" in report
    assert report["summary"]["n_pass"] >= 0
