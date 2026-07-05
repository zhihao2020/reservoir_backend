# Result Manifest

## Overview

The result manifest is the M7 boundary object between backend computation
artifacts and downstream consumers such as CLI reports, delivery bundles, or a
future frontend. It records what a result is, where it lives, and how it should
be interpreted. It does not run solvers or change numerical outputs.

## Manifest Schema

Required fields:

| Field | Meaning |
| --- | --- |
| `result_id` | Stable unique identifier inside a result catalog. |
| `case_id` | Case or dataset identifier. |
| `run_id` | Run identifier or report generation identifier. |
| `module` | Module owner, such as `M1`, `M3`, `M4`, `M5`, or `M8`. |
| `result_type` | Type such as `pressure_field`, `saturation_field`, `benchmark_registry`, or `experimental_data_qc`. |
| `field_name` | Field or report name, for example `pressure`, `sw`, or `qc_summary`. |
| `shape` | Array shape using the project convention. Scalars and reports use `[]`. |
| `dtype` | Data type, for example `float64` or `json`. |
| `unit` | Physical unit or `dimensionless`. |
| `path` | Relative or absolute path to the result artifact. |
| `format` | File format such as `json`, `md`, `csv`, `npz`, or `npy`. |
| `created_at` | ISO timestamp for the manifest entry. |
| `source_task` | Task that produced or registered the result. |
| `source_report` | Upstream report path when applicable. |
| `metadata` | JSON object for additional field context. |
| `warnings` | List of warning strings. |
| `limitations` | List of limitations attached to the result. |

## Result Types

Supported result type labels include:

- `experimental_data_qc`
- `pressure_field`
- `saturation_field`
- `capillary_gravity_report`
- `three_phase_report`
- `parameter_fusion_report`
- `benchmark_registry`
- `cross_scale_report`

## Path Conventions

Reports are indexed under `accuracy_reports/` when they are benchmark or QC
summaries. Case outputs remain under `results/<case_id>/`. Paths in manifests are
stored as strings and validated by the catalog without creating or mutating
files.

## JSON Example

```json
{
  "result_id": "pressure_field_demo",
  "case_id": "demo_case",
  "run_id": "result-contract-example",
  "module": "M3",
  "result_type": "pressure_field",
  "field_name": "pressure",
  "shape": [3, 4, 5],
  "dtype": "float64",
  "unit": "Pa",
  "path": "results/demo_case/pressure.npy",
  "format": "npy",
  "created_at": "2026-07-05T00:00:00+00:00",
  "source_task": "TASK-020",
  "source_report": "",
  "metadata": {"shape_convention": "(nz, ny, nx)"},
  "warnings": [],
  "limitations": ["example manifest only"]
}
```

## CSV Summary Behavior

CSV export is metadata-only. Large 3D arrays are not flattened into CSV. The CSV
summary contains manifest identifiers, field names, units, shapes, formats, and
paths.

## NPZ Field Array Behavior

NPZ export is available for field arrays when the caller explicitly provides
arrays. The manifest layer does not synthesize field data and does not alter
existing `.npy` or `.npz` results.

## Validation Rules

Validation rejects missing required keys, non-string identifier fields, invalid
shape entries, non-dict metadata, and non-list warnings or limitations.
