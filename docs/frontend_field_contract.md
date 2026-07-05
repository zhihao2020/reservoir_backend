# Frontend Field Contract

This document defines future frontend-facing field names and formats. It is a
field contract only.

No frontend implementation.
No UDP implementation.
No REST API implementation.

## Allowed Formats

- JSON for manifests, report summaries, warnings, and limitations.
- CSV for metadata summary tables.
- NPZ or NPY for numerical field arrays.
- Markdown for human-readable report indexes.

## Shape Conventions

- 3D cell-centered scalar fields use `(nz, ny, nx)`.
- X face fields use `(nz, ny, nx + 1)`.
- Y face fields use `(nz, ny + 1, nx)`.
- Z face fields use `(nz + 1, ny, nx)`.
- Report-only entries use `shape: []`.

## Pressure Field Fields

| Field | Unit | Shape |
| --- | --- | --- |
| `pressure` | `Pa` | `(nz, ny, nx)` |
| `flux_x` | `m3/s` or configured Darcy flux unit | `(nz, ny, nx + 1)` |
| `flux_y` | `m3/s` or configured Darcy flux unit | `(nz, ny + 1, nx)` |
| `flux_z` | `m3/s` or configured Darcy flux unit | `(nz + 1, ny, nx)` |

## Saturation Field Fields

| Field | Unit | Shape |
| --- | --- | --- |
| `sw` | `fraction` | `(nz, ny, nx)` |
| `so` | `fraction` | `(nz, ny, nx)` when present |
| `sg` | `fraction` | `(nz, ny, nx)` when present |
| `cfl` | `dimensionless` | scalar/report field |

## Fusion Field Fields

| Field | Unit | Shape |
| --- | --- | --- |
| `permeability` | `m2` | `(nz, ny, nx)` |
| `porosity` | `fraction` | `(nz, ny, nx)` |
| `confidence` | `fraction` | `(nz, ny, nx)` or report field |
| `variance` | field-specific squared unit | `(nz, ny, nx)` or report field |

## Benchmark Report Fields

Benchmark reports expose `success`, `num_cases`, `num_passed`, `num_failed`,
module-specific key metrics, `has_nan`, `has_inf`, `warnings`, and
`recommendations`.

## QC Report Fields

Experimental-data QC reports expose `success`, `fields_detected`,
`fields_missing`, `unit_warnings`, `num_nan`, `num_inf`, `num_missing`,
`bounds_violations`, `warnings`, and `recommendations`.

## Error And Warning Fields

All frontend-facing result entries may contain:

- `warnings`
- `limitations`
- `metadata`
- `source_report`
- `source_task`

## Limitations

This contract does not implement a frontend, UDP server changes, REST API,
database service, VTK large visualization export, Petrel-like workflow, solver
rewrite, or commercial data-management platform.
