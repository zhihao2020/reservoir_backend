"""Derive per-probe and per-well results from a finished reconstruction.

The inversion outputs full-grid fields (``p, sw, so, sg``) plus static ``phi, k``.
This module pulls those fields back to the observation points and wells so the
dashboard can see, for every time, the reconstructed value next to the observed
value, the fit error per probe, and the reconstructed well conditions.

Well "reconstructed BHP" is the mean of ``p`` over the well's completion cells;
well phase saturations are the means of ``sw/so/sg`` over those same cells.
Rates are the *input* controls echoed back (the inverse model treats rates as
inputs, so there is no independently reconstructed rate; the global mass
residual in ``diagnostics`` is the model's consistency measure).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_rmse(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _series(arr: NDArray[np.float64]) -> list:
    out = np.asarray(arr, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in out]


def _mean_over_cells(field: NDArray[np.float64], cells: NDArray[np.int64]) -> NDArray[np.float64]:
    """Time-wise mean of ``field[:, cells]`` (NaN-safe)."""
    sel = np.asarray(field, dtype=float)[:, cells]
    with np.errstate(all="ignore"):
        return np.nanmean(sel, axis=1)


def summarize_results(case, mesh, fields, *, times: NDArray[np.float64] | None = None) -> dict[str, Any]:
    """Per-probe / per-well reconstruction summary (JSON-serialisable).

    ``case`` is a :class:`~src.core.lab_case.LabCase`, ``mesh`` a
    :class:`~src.programs.mesh.MeshResult`, and ``fields`` a
    :class:`~src.programs.pipeline.ProgramFields`. ``times`` overrides the time
    axis (e.g. the field-scaled times for the ``field`` UDP stream) without
    touching the index-aligned observed/reconstructed values.
    """
    times = np.asarray(fields.times if times is None else times, dtype=float)
    probes_out: list[dict[str, Any]] = []
    for n, pid in enumerate(case.probe_ids):
        cell = int(mesh.probes.cell[n])
        obs = {
            "pressure_pa": case.pressure[:, n],
            "sw": case.sw[:, n],
            "so": case.so[:, n],
            "sg": case.sg[:, n],
        }
        rec = {
            "pressure_pa": fields.p[:, cell],
            "sw": fields.sw[:, cell],
            "so": fields.so[:, cell],
            "sg": fields.sg[:, cell],
        }
        probes_out.append(
            {
                "id": str(pid),
                "cell": cell,
                "observed": {k: _series(v) for k, v in obs.items()},
                "reconstructed": {k: _series(v) for k, v in rec.items()},
                "rmse": {k: _finite_rmse(obs[k], rec[k]) for k in obs},
            }
        )

    wells_out: list[dict[str, Any]] = []
    for n, wid in enumerate(case.well_ids):
        cells = np.asarray(mesh.wells.cells[n], dtype=np.int64)
        wells_out.append(
            {
                "id": str(wid),
                "kind": str(case.wells[n].kind),
                "n_cells": int(cells.size),
                "bhp_reconstructed_pa": _series(_mean_over_cells(fields.p, cells)),
                "sw": _series(_mean_over_cells(fields.sw, cells)),
                "so": _series(_mean_over_cells(fields.so, cells)),
                "sg": _series(_mean_over_cells(fields.sg, cells)),
                "control": {
                    "pw_pa": _series(case.well_pw[:, n]),
                    "q_m3s": _series(case.well_q[:, n]),
                    "qw_m3s": _series(case.well_qw[:, n]),
                    "qo_m3s": _series(case.well_qo[:, n]),
                    "qg_m3s": _series(case.well_qg[:, n]),
                },
            }
        )

    diagnostics = dict(fields.diagnostics) if getattr(fields, "diagnostics", None) else {}
    return {
        "n_times": int(times.size),
        "times_s": _series(times),
        "probes": probes_out,
        "wells": wells_out,
        "diagnostics": diagnostics,
    }
