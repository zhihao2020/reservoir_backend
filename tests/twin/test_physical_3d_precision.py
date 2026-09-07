"""Actual physical_3d reach and honest report-time comparison regressions."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from reservoir_backend.exceptions import TimeStepUnderflow
from reservoir_backend.twin.cmg_benchmark import load_twin_case, theta_true_from_twin, forward_at_theta
from scripts.lab_v1_cmg_compare_plot import _cap_times, _forward_capped


def test_physical_3d_reaches_864():
    twin = load_twin_case(Path('examples/lab_v1/cmg_gem/physical_3d/case.yaml'))
    times = np.array([0., 8.64, 864.])
    traj = twin.simulate(parameters=theta_true_from_twin(twin), t_end=864., report_times=times)
    np.testing.assert_allclose(traj.times_s, times, rtol=0, atol=1e-9)
    assert traj.reports[-1].time_s == pytest.approx(864., abs=1e-9)
    assert len(traj.reports) < 100
    np.testing.assert_allclose(traj.states[0].pressure, 5e7, rtol=0, atol=1e-3)
    final = traj.states[-1].pressure
    assert np.isfinite(final).all()
    assert 500 < float(final.max() - 5e7) < 3000
    assert float(final.min()) > 5e7 - 3000
    assert max(r.mass.relative_balance_error for r in traj.reports) < 1e-8


def test_comparison_defaults_to_first_gem_report():
    np.testing.assert_array_equal(_cap_times(np.array([0.,864.,12096.]),None,3375),[0.,864.])


def test_comparison_never_falls_back_on_underflow(monkeypatch):
    def fail(*args):
        raise TimeStepUnderflow('t=433')
    monkeypatch.setattr('scripts.lab_v1_cmg_compare_plot.forward_at_theta',fail)
    with pytest.raises(TimeStepUnderflow,match='433'):
        _forward_capped(None,None,np.array([0.,864.]))


def test_comparison_rejects_unreached_snapshot(monkeypatch):
    monkeypatch.setattr('reservoir_backend.twin.cmg_benchmark.ensure_molar_injector_rate',lambda twin:None)
    twin=SimpleNamespace(simulate=lambda **kwargs:SimpleNamespace(times_s=np.array([0.,433.])))
    with pytest.raises(TimeStepUnderflow,match='accepted times'):
        forward_at_theta(twin,np.array([0.]),np.array([0.,864.]))
