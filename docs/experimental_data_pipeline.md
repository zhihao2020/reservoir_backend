# Experimental Data Pipeline

The TASK-008 experimental data pipeline provides a lightweight input layer for
the backend. It converts CSV, JSON, and NPZ files into a standard
`ExperimentalDataset`, runs QC checks, optionally resamples or aligns fields,
and writes JSON / Markdown QC reports.

## Reader Flow

1. Detect input format from file extension.
2. Read CSV, JSON, or NPZ.
3. Map source field names to standard schema names.
4. Attach units, source name, input file, input format, and metadata.
5. Return a unified `ExperimentalDataset`.

Current supported formats:

- CSV
- JSON
- NPZ

Excel is not implemented in this stage because the project dependency set is
intentionally lightweight (`numpy`, `PyYAML`, `pytest`). JSON is used as the
third lightweight interchange format instead of adding an Excel dependency.

## QC Flow

The QC pipeline checks:

- required fields;
- numeric dtype compatibility;
- shape consistency;
- unit availability;
- unit normalization;
- NaN values;
- Inf values;
- missing values;
- duplicate time values;
- duplicate coordinate tuples;
- physical bounds;
- statistical outlier flags.

Physical bounds include:

- `0 <= porosity <= 1`
- `0 <= saturation <= 1`
- `permeability > 0`
- finite pressure
- `resistivity > 0`
- `0 <= confidence <= 1`
- `variance >= 0`

## Unit Normalization

Known units are normalized to canonical schema units:

- pressure to `Pa`
- permeability to `m2`
- porosity / saturation / confidence to `fraction`
- time to `s`
- coordinates to `m`
- temperature to `K`

Unsupported or missing units are recorded as warnings.

## NaN / Inf / Missing Handling

Blank CSV cells are read as `NaN`. NaN, Inf, and missing-value counts are
reported explicitly. The QC pipeline does not silently fill missing values.

## Outlier Flagging

The first implementation uses a z-score based outlier flag. Outlier flags are
diagnostic only; they do not modify data.

## Resampling / Alignment

Two lightweight operations are supported:

- `resample_time_series`: linear interpolation of 1D fields to a target time
  axis;
- `align_fields_to_grid_shape`: reshape compatible fields or scalar-fill fields
  into a target structured grid shape.

This is not complex geological regridding, not corner-point grid support, not
local grid refinement, and not kriging.

## QC Report Format

QC reports contain:

- `success`
- `input_file`
- `format`
- `num_rows`
- `shape`
- `fields_detected`
- `fields_missing`
- `unit_warnings`
- `num_nan`
- `num_inf`
- `num_missing`
- `num_outliers`
- `bounds_violations`
- `resample_summary`
- `warnings`
- `recommendations`

Reports can be written with:

```python
from reservoir_backend.data.report import write_qc_report
```

or via:

```bash
python -m reservoir_backend.data.report
```

Default outputs:

- `accuracy_reports/experimental_data_qc_summary.json`
- `accuracy_reports/experimental_data_qc_summary.md`

## Fixture Catalog

TASK-009 adds reusable fixtures and expected QC summaries:

- `tests/fixtures/experimental_data/manifest.json`
- `tests/fixtures/experimental_data/metadata/*.json`
- `tests/fixtures/experimental_data/expected/*.json`
- CSV / JSON / NPZ fixture inputs

The catalog covers valid CSV core fields, valid JSON multimodal fields, valid
NPZ grid fields, missing required fields, invalid units and bounds, duplicate
time/coordinates, and NaN / Inf / missing values. See `docs/data_contract.md`
for the full contract.

## Limitations

No database service.
No frontend integration.
No UDP implementation.
No commercial data-management platform.
No Petrel-like workflow.
No solver rewrite.
No history matching.
No automatic calibration.
No Bayesian inversion.
No EnKF / ES-MDA.
No kriging / Gaussian-process field modeling.
No black-oil model.
