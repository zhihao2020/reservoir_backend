# Parameter Fusion Benchmark Summary

- success: True
- num_cases: 8
- num_passed: 8
- overall_mae: 0.000000e+00
- overall_rmse: 0.000000e+00
- overall_max_abs_error: 0.000000e+00
- overall_num_bound_violations: 0
- overall_num_masked_cells: 2

## Cases

### equal_weight_field_fusion

- success: True
- source: internal equal-weight same-grid fusion benchmark
- success: True
- shape_consistent: True
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- num_compared: 6
- has_nan: False
- has_inf: False
- warnings: []
- fused_mean: 3.0
- field_count: 3
- num_masked_cells: 0
- num_bound_violations: 0

### explicit_weight_field_fusion

- success: True
- source: internal explicit-weight fusion benchmark
- success: True
- shape_consistent: True
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- num_compared: 6
- has_nan: False
- has_inf: False
- warnings: []
- used_weights: [1.0, 3.0]
- invalid_weight_rejected: True
- weight_min: 1.0
- weight_max: 3.0
- num_masked_cells: 0
- num_bound_violations: 0

### confidence_weighted_fusion

- success: True
- source: internal confidence-weighted fusion benchmark
- success: True
- shape_consistent: True
- mean_distance_to_low_confidence: 0.7200000000000001
- mean_distance_to_high_confidence: 0.07999999999999996
- closer_to_high_confidence: True
- high_confidence_influence_ratio: 9.000000000000005
- warnings: []
- fused_mean: 0.8200000000000002
- confidence_min: 0.5
- confidence_max: 0.5
- zero_confidence_source_does_not_dominate: True
- num_masked_cells: 0
- num_bound_violations: 0
- has_nan: False
- has_inf: False

### uncertainty_or_variance_weighted_fusion_if_supported

- success: True
- source: current fusion API capability survey
- uncertainty_fusion_supported: False
- inverse_variance_weighting_verified: False
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- num_masked_cells: 0
- num_bound_violations: 0
- has_nan: False
- has_inf: False

### nan_aware_fusion

- success: True
- source: internal NaN-aware fusion benchmark
- success: True
- shape_consistent: True
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- num_compared: 6
- has_nan: False
- has_inf: False
- warnings: []
- single_source_nan_ignored: True
- all_source_nan_masked: True
- num_masked_cells: 2
- nan_cells_count: 2
- expected_masked_output_has_nan: True
- num_bound_violations: 0

### bounds_and_clipping_report

- success: True
- source: internal physical-bounds fusion benchmark
- clipped_cells: 6
- saturation_num_bound_violations: 0
- porosity_num_bound_violations: 0
- permeability_positive: True
- num_bound_violations: 0
- num_masked_cells: 0
- has_nan: False
- has_inf: False

### shape_mismatch_rejection

- success: True
- source: internal shape-mismatch fusion benchmark
- shape_mismatch_rejected: True
- shape_consistent: False
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- num_masked_cells: 0
- num_bound_violations: 0
- has_nan: False
- has_inf: False

### multi_field_property_dynamic_fusion_sanity

- success: True
- source: internal multi-field property/dynamic fusion sanity benchmark
- all_outputs_finite: True
- shape_consistent: True
- permeability_positive: True
- porosity_num_bound_violations: 0
- saturation_num_bound_violations: 0
- updated_saturation_num_bound_violations: 0
- source_names: ['perm_a', 'perm_b', 'phi_a', 'phi_b', 'pressure_a', 'pressure_b', 'sw_sim', 'sw_obs']
- source_count: 8
- mask_shape_consistent: True
- clipped_cells: 0
- num_masked_cells: 0
- num_bound_violations: 0
- mae: 0.0
- rmse: 0.0
- max_abs_error: 0.0
- has_nan: False
- has_inf: False
