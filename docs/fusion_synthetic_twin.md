# Fusion Synthetic Twin

## Purpose

F4-04 adds a lightweight synthetic twin data layer for parameter-field fusion.
It combines static property fields, dynamic simulation/observation fields, and
production or water-cut time series into one reportable structure.

This stage is a data and diagnostics layer. It is not history matching, not an
ensemble data-assimilation workflow, and not a closed-loop digital twin.

## Data Structures

- `SyntheticTwinMetadata`: twin id, case id, run id, grid shape, time steps,
  source name, and run metadata.
- `StaticFieldRecord`: static fields such as permeability and porosity with
  source, confidence, variance, mask, provenance, and optional synthetic truth.
- `DynamicFieldRecord`: time-indexed fields such as pressure and saturation
  with time steps, confidence, mask, provenance, and optional synthetic truth.
- `ProductionSeriesRecord`: production or water-cut time series with source,
  confidence, mask, provenance, and optional synthetic truth.
- `DynamicFusionSummary`: JSON-serializable static, dynamic, production,
  diagnostics, provenance, warnings, and limitations output.

## Static Field Fusion

Static records are grouped by `field_name` and fused with the existing
uncertainty-aware weighting utility. Supported static examples are:

- permeability
- porosity

Bounds are checked for known physical quantities:

- permeability should be positive.
- porosity should stay within `[0, 1]`.

## Dynamic Field Fusion

Dynamic records are grouped by `field_name`. Each dynamic record must have
shape:

```text
(num_time_steps, nz, ny, nx)
```

The metadata time-step list must match each dynamic record. Pressure and
saturation records preserve source, confidence, mask, and provenance.
Saturation is checked against `[0, 1]`.

## Production Series

Production or water-cut series are fused as time series. Each record must have
matching 1D `time` and `values` arrays. Water cut is checked against `[0, 1]`.

## Diagnostics

If synthetic truth is provided, the report computes:

- RMSE
- MAE
- max absolute error
- number of compared samples

The report also records:

- shape / time-step consistency
- confidence weighting policy
- mask and NaN handling
- bounds violations
- provenance sources
- static / dynamic / production record counts

## Report

Run:

```bash
python -m reservoir_backend.fusion.synthetic_twin_report
```

Outputs:

```text
accuracy_reports/fusion_synthetic_twin_summary.json
accuracy_reports/fusion_synthetic_twin_summary.md
```

## Limitations

- No history matching is implemented.
- No EnKF / ES-MDA is implemented.
- No automatic geological model update is implemented.
- No closed-loop digital twin control is implemented.
- No frontend, UDP, or REST API is implemented.
- No solver, inversion, cross-scale, data, benchmark, reference, config, C++,
  CMake, or pybind11 code is modified by F4-04.
