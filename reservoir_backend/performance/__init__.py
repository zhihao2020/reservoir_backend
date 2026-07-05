"""Performance baseline utilities."""

__all__ = ["run_performance_baseline"]


def run_performance_baseline(*args, **kwargs):
    """Run the TASK-019 performance baseline report."""
    from reservoir_backend.performance.performance_report import run_performance_baseline as _run

    return _run(*args, **kwargs)
