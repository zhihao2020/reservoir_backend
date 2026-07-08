# Data Contract Redirect

The active data contract is [API_AND_DATA_CONTRACT.md](API_AND_DATA_CONTRACT.md).

## Required Fields

- `required_fields`
- time or coordinates depending on input type

## Optional Fields

- resistivity
- electromagnetic_response
- acoustic_response
- pressure
- saturation
- porosity
- permeability
- temperature
- confidence
- variance

## Units and Normalization

pressure, permeability, fraction/percent, time, coordinates, and temperature.

## Shape Conventions

Structured fields use `(nz, ny, nx)` where applicable.

## QC Behavior

Reports include `bounds_violations`, missing values, NaN, Inf, and unit warnings.

## Expected Errors and Warnings

Reports include `fields_missing`, invalid units, shape mismatch, and bounds violations.

## Fixture Catalog

- valid_csv_core_fields
- valid_json_multimodal_fields
- valid_npz_grid_fields
- invalid_missing_required_fields
- invalid_units_and_bounds
