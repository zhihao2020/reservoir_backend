# Cross-Scale Upscaling Report

## Purpose

TASK-017 adds an explanatory cross-scale report layer. It summarizes similarity
criteria, lab-to-field scale conversion, lightweight upscaling assumptions, and a
fine-grid vs coarse-grid comparison framework. It does not implement a complex
upscaling solver.

## Scale Conversion Fields

The scale conversion report includes:

- `length_scale_lab`
- `length_scale_field`
- `length_scale_ratio`
- `time_scale_lab`
- `time_scale_field`
- `time_scale_ratio`
- `pressure_scale_lab`
- `pressure_scale_field`
- `pressure_scale_ratio`
- `permeability_scale_lab`
- `permeability_scale_field`
- `permeability_scale_ratio`
- `velocity_scale_lab`
- `velocity_scale_field`
- `velocity_scale_ratio`
- `flow_rate_scale_lab`
- `flow_rate_scale_field`
- `flow_rate_scale_ratio`
- `porosity_lab`
- `porosity_field`
- `porosity_ratio`

Ratios are field value divided by lab value. They are diagnostics, not
deterministic equivalence rules.

## Similarity Criteria Used

The report embeds the existing similarity criteria:

- Reynolds number, `Re`
- Capillary number, `Ca`
- Peclet number, `Pe`
- mobility ratio
- gravity number
- dimensionless pressure
- dimensionless time
- similarity score
- missing parameter warnings

## Upscaling Assumptions

The assumption report states:

- which properties may be upscaled;
- which properties should not be upscaled directly;
- which diagnostic assumptions are used;
- what a regime shift means;
- what validation is required before using coarse-scale outputs.

Properties that may be summarized include permeability, porosity, and flow rate.
Capillary-pressure curves, relative-permeability curves, and history-matched
parameters should not be upscaled directly without validation.

## Lightweight Upscaling Diagnostics

The current diagnostics include:

- arithmetic mean permeability;
- harmonic mean permeability;
- porosity volume average;
- flow-rate scaling sanity;
- velocity scaling sanity;
- regime shift flag.

These values are diagnostic evidence only. They are not a multiscale simulator.

## Fine-Grid vs Coarse-Grid Comparison Framework

The fine/coarse comparison accepts supplied or synthetic curves and reports:

- pressure curve comparison metrics;
- saturation curve comparison metrics;
- production curve comparison metrics when available;
- RMSE;
- MAE;
- R2;
- NRMSE;
- max absolute error;
- warnings for no overlap or undefined metrics.

If real fine/coarse data are absent, the test fixtures use synthetic curves and
the report states that limitation.

## Report Schema

`accuracy_reports/cross_scale_upscaling_summary.json` contains:

- `report_name`
- `case_id`
- `success`
- `scale_conversion_report`
- `similarity_criteria_report`
- `upscaling_assumption_report`
- `fine_coarse_comparison_report`
- `result_manifest_entry`
- `warnings`
- `limitations`
- `non_claims`
- `has_nan`
- `has_inf`

The Markdown report is written to
`accuracy_reports/cross_scale_upscaling_summary.md`.

## Limitations

No complex upscaling solver.
No multiscale finite-volume implementation.
No history matching.
No automatic calibration.
No commercial simulator equivalence.
No validation of black-oil models.
No front-end.
No UDP.

## Non-Claims

The report is not a Petrel-like workflow, not OPM Flow equivalence, not MRST
integration, not full SPE1 / SPE10 reproduction, and not a commercial simulator
validation.
