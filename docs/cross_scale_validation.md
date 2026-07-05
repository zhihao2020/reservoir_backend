# Cross-Scale Validation

## Similarity Criteria

The runner reports:

- Reynolds number, `Re`
- Capillary number, `Ca`
- Peclet number, `Pe`
- mobility ratio
- gravity number when density contrast is available
- dimensionless pressure when pressure drop is available
- dimensionless time when elapsed time is available
- overall similarity score
- missing-parameter warnings

## Scale-Effect Metrics

The scale-effect report includes:

- length ratio
- time ratio
- pressure ratio
- permeability ratio
- velocity ratio
- flow-rate ratio
- porosity ratio
- regime classification
- regime shift warning

## Lab-Field Curve Validation Metrics

The validation report includes:

- curve names
- overlap interval
- RMSE
- MAE
- MAPE if valid
- R2
- NRMSE
- max absolute error
- number of matched samples
- warnings

## Benchmark Fixture Description

Fixtures live in `tests/fixtures/cross_scale/`:

- `valid_cross_scale_case.json`
- `valid_cross_scale_case.yaml`
- `no_overlap_case.json`
- `invalid_missing_field_case.json`

The valid fixtures exercise all three report paths. The no-overlap fixture
checks curve warning behavior. The invalid fixture checks configuration
rejection.

## Report Fields

The benchmark summary contains:

- `benchmark_name`
- `case_id`
- `success`
- `similarity_report`
- `scale_effect_report`
- `lab_field_validation_report`
- `result_manifest_entry`
- `output_paths`
- `warnings`
- `limitations`
- `has_nan`
- `has_inf`

TASK-017 adds `docs/cross_scale_upscaling_report.md` and
`accuracy_reports/cross_scale_upscaling_summary.json/md` for scale conversion,
upscaling assumptions, and fine-grid vs coarse-grid comparison reporting.

## Limitations

No history matching.
No automatic calibration.
No complex upscaling solver.
No front-end.
No UDP.
No commercial simulator equivalence.
No validation of black-oil models.
