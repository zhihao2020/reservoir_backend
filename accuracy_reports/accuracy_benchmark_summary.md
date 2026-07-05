# Accuracy Benchmark Summary

- success: True
- num_benchmarks: 8
- num_passed: 8
- num_failed: 0
- has_nan: False
- has_inf: False

## Benchmarks

### pressure_linear_1d
- success: True
- has_nan: False
- has_inf: False
- max_pressure_error: 5.587935447692871e-09
- relative_pressure_error: 6.20881716410319e-16
- max_flux_variation: 7.453889935837843e-20
- mass_balance_error: 0.0

### pressure_manufactured_3d
- success: True
- has_nan: False
- has_inf: False
- l2_error: 7.566100363280588e-09
- linf_error: 1.4901161193847656e-08
- relative_l2_error: 9.546607301900083e-16

### buckley_leverett_1d
- success: True
- has_nan: False
- has_inf: False
- sw_min: 0.2
- sw_max: 0.4492752648635387
- front_position_estimate: 3
- material_balance_error: 0.0
- max_cfl: 0.02

### capillary_smoothing
- success: True
- has_nan: False
- has_inf: False
- initial_gradient_norm: 0.9000000000000001
- final_gradient_norm: 0.6832627215955484
- gradient_reduction: 0.21673727840445178
- max_abs_capillary_flux: 1.584936490538905e-05
- sw_min: 0.35
- sw_max: 0.65

### gravity_segregation
- success: True
- has_nan: False
- has_inf: False
- expected_gravity_flux_sign: -1.0
- observed_gravity_flux_sign: -1.0
- bottom_sw_change: 0.00040861041666662157
- top_sw_change: -0.0004086104166666771
- sw_min: 0.4995913895833333
- sw_max: 0.5004086104166666

### combined_transport_stability
- success: True
- has_nan: False
- has_inf: False
- max_abs_capillary_flux: 1.584936490538905e-08
- max_abs_gravity_flux: 7.880343749999997e-08
- max_total_water_flux: 7.880343749999997e-08
- max_effective_flux: 7.880343749999997e-08
- material_balance_error: 0.0
- max_cfl: 8.672811995269449e-05
- sw_min: 0.34996059828125
- sw_max: 0.6500119918274456

### three_phase_closure
- success: True
- has_nan: False
- has_inf: False
- closure_error_max: 0.0
- sw_min: 0.3
- sw_max: 0.3
- sg_min: 0.1
- sg_max: 0.1
- so_min: 0.6
- so_max: 0.6
- water_balance_error: 0.0
- gas_balance_error: 0.0
- oil_balance_error: 0.0

### cross_scale_formula_check
- success: True
- has_nan: False
- has_inf: False
- formula_checks_passed: 8
- num_formula_checks: 8
- max_formula_error: 0.0

## Recommendations

- current C++ recommendation: No C++ migration is recommended from this MVP accuracy suite; benchmark cases are small and stable.
- current numerical risk: Explicit transport remains the main risk for stronger capillary/gravity cases or finer grids.
- which module needs further validation: Pressure, saturation, capillary, gravity, combined, three-phase, and cross-scale formulas need larger experimental/golden-case validation before production use.
