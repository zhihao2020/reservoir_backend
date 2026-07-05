# Saturation Transport Benchmark Summary

- success: True
- num_cases: 7
- num_passed: 7
- overall_material_balance_error: 1.734723e-16
- overall_max_cfl: 2.000000e+00

## Open-source references used
- MRST buckleyLeverett1D
- MRST simpleIncompTPFA
- OPM SPE1CASE1 metadata

## Cases

### buckley_leverett_1d_qualitative

- success: True
- source: MRST buckleyLeverett1D.m
- initial_front_position: None
- final_front_position: 3.5
- front_moved_downstream: True
- inlet_sw_increase: 0.34837148888808217
- sw_min: 0.2
- sw_max: 0.5483714888880822
- material_balance_error: 1.387778780781445e-16
- max_cfl: 0.049999999999999996
- has_nan: False
- has_inf: False

### mrst_buckley_leverett_1d_reference

- success: True
- source: MRST buckleyLeverett1D.m
- metadata_loaded: True
- grid_shape: [100, 1]
- porosity: 0.2
- permeability_md: 100.0
- mentions_explicit_transport: True
- mentions_implicit_transport: True
- has_nan: False
- has_inf: False

### saturation_boundedness

- success: True
- source: internal boundedness regression set
- num_cases: 5
- num_bounded: 5
- sw_min_global: 0.2
- sw_max_global: 0.7999990000000015
- num_bound_violations: 0
- clipped_cells: 0
- bound_handling: solver clips to [Swi, 1-Sor] when needed
- has_nan: False
- has_inf: False

### cfl_stability

- success: True
- source: internal CFL stability regression
- stable_max_cfl: 0.09999999999999999
- near_limit_max_cfl: 0.9
- too_large_max_cfl: 2.0
- num_cfl_warnings: 1
- stability_flags: {'stable': 'success', 'near_limit': 'success', 'too_large': 'cfl_violation'}
- max_cfl: 2.0
- has_nan: False
- has_inf: False

### material_balance_1d

- success: True
- source: internal one-step material-balance case
- initial_water_volume: 1.5000000000000004
- final_water_volume: 1.5100000000000005
- injected_water_volume: 0.01
- produced_water_volume: 0.0
- material_balance_residual: -1.734723475976807e-18
- relative_material_balance_error: 1.7347234759768068e-16
- material_balance_error: 1.7347234759768068e-16
- has_nan: False
- has_inf: False

### areal_waterflood_2d_qualitative

- success: True
- source: internal areal-like x-direction waterflood
- injection_region_sw_initial: 0.20000000000000004
- injection_region_sw_final: 0.2239408333628254
- producer_region_sw_initial: 0.20000000000000004
- producer_region_sw_final: 0.20000000000000004
- front_direction_score: 1.0
- sw_min: 0.2
- sw_max: 0.2239408333628254
- has_nan: False
- has_inf: False

### opm_spe1case1_saturation_sanity_adapted

- success: True
- source: OPM/opm-tests spe1 SPE1CASE1.DATA
- porosity_min: 0.3
- porosity_max: 0.3
- permeability_min_md: 50.0
- permeability_max_md: 500.0
- permeability_contrast: 10.0
- sw_min: 0.2
- sw_max: 0.2
- bounded: True
- has_nan: False
- has_inf: False
