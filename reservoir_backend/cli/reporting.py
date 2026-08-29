"""CLI helpers for unified run reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reservoir_backend.twin.acceptance import build_check83_report, write_check83_report
from reservoir_backend.twin.offline import DigitalTwin, Posterior
from reservoir_backend.twin.run_report import build_invert_report, write_run_report


def emit_invert_artifacts(
    twin: DigitalTwin,
    posterior: Posterior,
    output: Path | None,
    *,
    case_path: Path | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
    fields: dict | None = None,
) -> dict[str, Any]:
    report = build_invert_report(
        twin, posterior, case_path=case_path, seed=seed, extra=extra
    )
    check83 = build_check83_report(twin, posterior)
    report["check83_summary"] = check83.get("summary")
    print(__import__("json").dumps(report, indent=2))
    if output is not None:
        import numpy as np

        output.mkdir(parents=True, exist_ok=True)
        if fields:
            for name, value in fields.items():
                arr = np.asarray(value)
                if arr.ndim >= 1 and arr.dtype != object:
                    np.save(output / f"{name}.npy", arr)
        write_run_report(output, report, twin=twin, posterior=posterior)
        write_check83_report(output, check83)
    return report
