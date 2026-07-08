# Experimental Data Contract

## Overview

TASK-009 fixes the test and documentation contract for the experimental data
entry layer. Fixtures under `tests/fixtures/experimental_data/` describe which
inputs should pass, which should fail, and what QC summary each case should
produce.

Fixtures are synthetic and deterministic. They are not real field data and do
not represent commercial data-management workflows.

## Supported Formats

Current supported reader formats:

- CSV
- JSON
- NPZ

Excel is intentionally not included in this lightweight stage because the
project dependency set does not include an Excel reader. JSON is used as the
third text interchange format.

## Required Fields

The schema has no global hard-coded required fields. Required fields are
declared by each consumer or fixture metadata. Example required sets:

- inversion: `time`, `porosity`, `resistivity`
- pressure: `porosity`, `permeability`, `pressure`
- saturation transport: `time`, `porosity`, `saturation`
- fusion: any same-shape property or dynamic fields plus optional confidence

Fixture metadata files list `required_fields` explicitly.

## Optional Fields

Optional standard fields:

- `resistivity`
- `electromagnetic_response`
- `acoustic_response`
- `pressure`
- `saturation`
- `porosity`
- `permeability`
- `temperature`
- `time`
- `x`, `y`, `z`
- `confidence`
- `variance`

## Units and Normalization

Supported unit normalization:

- pressure: `Pa`, `kPa`, `MPa`, `bar` -> `Pa`
- permeability: `m2`, `mD`, `D` -> `m2`
- fractions: `fraction`, `decimal`, `percent` -> `fraction`
- time: `s`, `min`, `h`, `day` -> `s`
- coordinates: `m`, `cm`, `mm` -> `m`
- temperature: `K`, `C` -> `K`

Missing or unknown units are reported through `unit_warnings`. Unknown suffixes
such as `pressure_psi` are not silently treated as canonical units.

## Shape Conventions

- 1D arrays represent records or time series.
- 2D arrays represent tabular or simple spatial records.
- 3D arrays follow project convention `(nz, ny, nx)`.
- Fields in one dataset should share shape unless a specific alignment step is
  requested.
- `valid_npz_grid_fields` demonstrates shape alignment to `(1, 2, 3)`.

## Metadata Convention

Each fixture has a metadata file under:

```text
tests/fixtures/experimental_data/metadata/
```

Metadata contains:

- `fixture_id`
- `source_name`
- `required_fields`
- optional `target_shape`
- description

## QC Behavior

QC reports check:

- required fields;
- dtype conversion;
- shape consistency;
- unit availability;
- unit normalization;
- NaN / Inf / missing values;
- duplicate time values;
- duplicate coordinate tuples;
- physical bounds;
- outlier flags;
- warnings and recommendations.

Physical bounds:

- `0 <= porosity <= 1`
- `0 <= saturation <= 1`
- `permeability > 0`
- finite pressure
- `resistivity > 0`
- `0 <= confidence <= 1`
- `variance >= 0`

## Fixture Catalog

| Fixture ID | Format | Expected behavior | Purpose |
| --- | --- | --- | --- |
| `valid_csv_core_fields` | CSV | pass | Core fields for inversion, pressure, transport, and fusion |
| `valid_json_multimodal_fields` | JSON | pass | Resistivity / EM / acoustic / pressure / saturation multimodal data |
| `valid_npz_grid_fields` | NPZ | pass | Structured-grid property fields and shape alignment |
| `invalid_missing_required_fields` | CSV | fail | Missing required porosity and permeability |
| `invalid_units_and_bounds` | CSV | fail | Unknown unit suffix and physical bounds violations |
| `duplicate_time_or_coordinates` | CSV | pass with warnings | Duplicate time and coordinate tuple detection |
| `nan_inf_missing_values` | CSV | fail | NaN, Inf, and missing-value detection |

Manifest:

```text
tests/fixtures/experimental_data/manifest.json
```

Expected summaries:

```text
tests/fixtures/experimental_data/expected/*.json
```

## Expected Errors and Warnings

- Missing required fields produce `fields_missing` and `success=false`.
- Unknown units produce `unit_warnings`.
- Physical bounds violations are listed by field in `bounds_violations`.
- NaN / Inf / blank values are counted as `num_nan`, `num_inf`, and
  `num_missing`.
- Duplicate time or coordinate tuples are reported in dedicated counters.
- Outlier flags are diagnostic and do not modify input values.

## Limitations

No database service.
No frontend integration.
No UDP implementation.
No commercial data-management platform.
No Petrel-like workflow.
No solver rewrite.
No benchmark physics-case modification.
No history matching.
No automatic calibration.
No Bayesian inversion.
No EnKF / ES-MDA.
No kriging / Gaussian-process modeling.
No deep-learning surrogate.
