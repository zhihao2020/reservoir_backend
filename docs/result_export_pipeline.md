# Result Export Pipeline

## Registration Flow

1. A module or report producer creates a result file.
2. A `ResultManifest` records field identity, shape, unit, path, format, source
   task, warnings, and limitations.
3. A `ResultCatalog` registers one or more manifests.
4. The catalog validates path consistency without creating missing result files.
5. Export helpers write JSON manifest summaries, CSV metadata summaries, NPZ
   arrays supplied by callers, and Markdown report indexes.

## Manifest Generation

The manifest layer is intentionally separate from numerical modules. TASK-020
adds example manifests for pressure, saturation, parameter fusion,
experimental-data QC, and benchmark registry outputs.

## Catalog Validation

The catalog supports:

- `add`
- `list`
- `find`
- `validate_paths`
- JSON serialization through `to_dict`

Duplicate `result_id` values are rejected. Missing result paths produce warnings.

## Export Formats

- JSON manifest: full machine-readable catalog or summary.
- CSV summary: metadata table only; large arrays are not flattened.
- NPZ field arrays: explicit caller-provided arrays.
- Markdown report index: human-readable result and report path listing.

## Report Path Index

The default report path index registers:

- `accuracy_reports/experimental_data_qc_summary.json`
- `accuracy_reports/experimental_data_qc_summary.md`
- `accuracy_reports/saturation_inversion_benchmark_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/capillary_gravity_benchmark_summary.json`
- `accuracy_reports/three_phase_benchmark_summary.json`
- `accuracy_reports/parameter_fusion_benchmark_summary.json`
- `accuracy_reports/benchmark_registry_summary.json`

Missing reports are not fabricated; they are recorded as missing warnings.

## Limitations

No database service.
No frontend integration.
No UDP implementation.
No VTK large visualization export.
No commercial data-management platform.
No Petrel-like workflow.
No solver rewrite.
