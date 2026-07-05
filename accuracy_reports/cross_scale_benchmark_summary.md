# Cross-Scale Benchmark Summary

- success: True
- case_id: cross_scale_default
- similarity_score: 0.16586000000000004
- regime_shift_detected: True
- num_curves: 1

## Similarity Criteria

| criterion | lab | field | score |
| --- | --- | --- | --- |
| reynolds | 1.0 | 100000.0 | 9.999999999999997e-06 |
| capillary | 3.3333333333333334e-08 | 3.3333333333333335e-05 | 0.0010000000000000002 |
| peclet | 999.9999999999999 | 100000000.0 | 9.999999999999997e-06 |
| mobility_ratio | 2.0 | 2.0 | 1.0 |
| gravity_number | 0.9806649999999999 | 0.009806650000000002 | 0.010000000000000004 |
| dimensionless_pressure | 0.1 | 0.005 | 0.04999999999999998 |
| dimensionless_time | 9.999999999999999e-05 | 0.001 | 0.09999999999999998 |

## Scale Effect

| metric | value |
| --- | --- |
| scale_ratio_length | 100.0 |
| scale_ratio_time | 100.0 |
| scale_ratio_pressure | 20.0 |
| scale_ratio_permeability | 0.2 |
| scale_ratio_velocity | 1000.0000000000001 |
| scale_ratio_flow_rate | 20000.0 |
| scale_ratio_porosity | 1.25 |
| scale_ratio_temperature | 1.0 |
| regime_lab | capillary_convection_dominated_laminar_viscous_flow |
| regime_field | mixed_or_uncertain_convection_dominated_inertial_effect_possible |
| regime_shift_detected | True |

## Lab-Field Validation

| metric | value |
| --- | --- |
| rmse | 0.061237243569579436 |
| mae | 0.04999999999999999 |
| mape | 11.874999999999996 |
| r2 | 0.9591836734693878 |
| nrmse | 0.0765465544619743 |
| max_absolute_error | 0.09999999999999998 |
| num_matched_samples | 4 |

## Warnings

- curve 0: mape used epsilon for 1 near-zero reference values

## Limitations

- No history matching.
- No automatic calibration.
- No complex upscaling solver.
- No frontend.
- No UDP.
- No commercial simulator equivalence.
- No black-oil validation.
