# Benchmark Registry

Benchmark Registry: Done
Benchmark Registry：Done

The benchmark registry is the validation index layer for the reservoir digital
twin backend. It reads existing benchmark summary JSON files and reference
fixtures, then records which module each benchmark covers, what validation
level each case represents, what reference material is used, whether the case
is an exact reproduction, and where the detailed report is stored.

The registry does not run solvers, rewrite algorithms, parse upstream OPM/MRST
files, or create runtime dependencies on external simulators.

## Registered Benchmark Summaries

- `saturation_inversion_benchmark`
- `pressure_solver_benchmark`
- `saturation_transport_benchmark`
- `capillary_gravity_benchmark`
- `three_phase_benchmark`
- `parameter_fusion_benchmark`

Registry output:

- `accuracy_reports/benchmark_registry_summary.json`
- `accuracy_reports/benchmark_registry_summary.md`

## Registered Benchmark Cases

The registry expands every benchmark summary into case entries with:

- `case_name`
- `module_id`
- `validation_level`
- `reference_type`
- `source`
- `is_exact_reproduction`
- `success`
- `key_metrics`
- `limitations`
- `warnings`

For benchmark summaries that predate the common `cases` schema, such as
`saturation_inversion_benchmark`, the registry builds case entries from the
existing summary fields without modifying the original benchmark report.

## Module Coverage

The current registry covers:

- M2: saturation inversion
- M3: pressure field reconstruction
- M4: saturation, capillary/gravity, combined, and simplified WOG transport
- M5: parameter field fusion
- M8: validation, benchmark, and delivery reports

## Validation Level Taxonomy

Each case is classified as one of:

- `analytical`
- `manufactured_solution`
- `adapted_open_source_reference`
- `diagnostic_sanity`
- `property_metadata_sanity`
- `trend_validation`
- `stability_validation`

Examples:

- Archie analytical inversion: `analytical`
- 2D/3D manufactured pressure: `manufactured_solution`
- OPM SPE1CASE1 adapted property sanity: `property_metadata_sanity`
- Buckley-Leverett qualitative front movement: `trend_validation`
- three-phase closure: `diagnostic_sanity`
- explicit stability / boundedness checks: `stability_validation`

## Reference Type Taxonomy

Each case is also classified by reference posture:

- `exact reproduction`
- `adapted reference`
- `reference context only`
- `property metadata sanity only`
- `internal benchmark`

The current registry does not contain exact OPM/MRST reproductions. OPM and
MRST materials are used only as adapted metadata, reference context, or
qualitative benchmark inspiration.

## Report Paths

The registry records each source summary:

- `accuracy_reports/saturation_inversion_benchmark_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/capillary_gravity_benchmark_summary.json`
- `accuracy_reports/three_phase_benchmark_summary.json`
- `accuracy_reports/parameter_fusion_benchmark_summary.json`

## Known Limitations

- Registry reads existing summary files only.
- Missing summaries are reported as missing and make registry success false.
- The registry is not a new benchmark implementation.
- The registry is not a solver, not a calibration engine, and not a workflow
  manager.
- The registry checks for overclaim wording but cannot replace engineering
  review of all external communications.

## Non-Claims

No full SPE1 reproduction.
No full SPE10 reproduction.
No OPM Flow equivalence.
No MRST integration.
No commercial simulator equivalence.
No black-oil validation.
No history matching implemented.
No automatic calibration implemented.
