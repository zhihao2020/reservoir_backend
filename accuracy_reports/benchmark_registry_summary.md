# Benchmark Registry Summary

- success: True
- num_benchmark_summaries: 6
- num_benchmark_cases: 43
- num_passed_cases: 43
- num_failed_cases: 0
- num_missing_summaries: 0
- modules_covered: M2, M3, M4, M5, M8

## Summary Table

| Benchmark | Module | Task | Success | Cases | Validation levels | Reference types |
| --- | --- | --- | --- | --- | --- | --- |
| saturation_inversion_benchmark | M2 | TASK-046 | True | 6 | analytical, diagnostic_sanity | internal benchmark |
| pressure_solver_benchmark | M3 | TASK-047 | True | 8 | adapted_open_source_reference, diagnostic_sanity, manufactured_solution, property_metadata_sanity | adapted reference, internal benchmark, property metadata sanity only, reference context only |
| saturation_transport_benchmark | M4 | TASK-048 | True | 7 | diagnostic_sanity, property_metadata_sanity, stability_validation, trend_validation | adapted reference, internal benchmark, property metadata sanity only, reference context only |
| capillary_gravity_benchmark | M4 | TASK-049 | True | 8 | diagnostic_sanity, property_metadata_sanity, stability_validation, trend_validation | internal benchmark, property metadata sanity only |
| three_phase_benchmark | M4 | TASK-050 | True | 8 | diagnostic_sanity, property_metadata_sanity, stability_validation | internal benchmark, property metadata sanity only |
| parameter_fusion_benchmark | M5 | TASK-051 | True | 8 | diagnostic_sanity, property_metadata_sanity, trend_validation | internal benchmark, property metadata sanity only |

## Open-Source References

- opm_water_1ph_single_cell: OPM/opm-tests `water-1ph/WATER2F.DATA`; type=property metadata sanity only; runtime_dependency=False; exact_reproduction=False
- opm_spe1_case1_layered_subset: OPM/opm-tests `spe1/SPE1CASE1.DATA`; type=adapted reference; runtime_dependency=False; exact_reproduction=False
- mrst_simple_incomp_tpfa_reference: SINTEF-AppliedCompSci/MRST `modules/book/examples/1phase/src/simpleIncompTPFA.m`; type=reference context only; runtime_dependency=False; exact_reproduction=False
- mrst_buckley_leverett_1d_reference: SINTEF-AppliedCompSci/MRST `modules/book/examples/in2ph/buckleyLeverett1D.m`; type=adapted reference; runtime_dependency=False; exact_reproduction=False

## Limitations

- Registry reads existing benchmark summaries and reference fixtures only.
- OPM/MRST materials are adapted references or context only, with no runtime dependency.
- Registry does not claim full SPE1/SPE10 reproduction, OPM Flow equivalence, MRST integration, commercial simulator equivalence, or black-oil validation.

## Overclaim Warnings

- None
