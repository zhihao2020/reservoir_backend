# Capillary / Gravity Benchmark Summary

- success: True
- num_cases: 8
- num_passed: 8
- overall_gradient_reduction: 5.761298e-02
- overall_max_capillary_flux: 1.205524e-05
- overall_max_gravity_flux: 8.172208e-08
- overall_material_balance_error: 0.000000e+00

## Cases

### capillary_pressure_monotonicity

- success: True
- source: internal capillary pressure trend case
- pc_min: 1008.4389681792214
- pc_max: 7745.966692414842
- pc_mean: 1819.8276411653333
- pc_monotonicity_score: 1.0
- num_nonfinite: 0
- model_convention: Brooks-Corey Pc=Po-Pw decreases as Sw increases
- has_nan: False
- has_inf: False

### capillary_no_gradient_zero_flux

- success: True
- source: internal uniform-Sw capillary flux case
- max_abs_capillary_flux: 0.0
- mean_abs_capillary_flux: 0.0
- flux_zero_tolerance: 1e-30
- success: True
- has_nan: False
- has_inf: False

### capillary_smoothing

- success: True
- source: internal 1D capillary smoothing case
- initial_gradient_norm: 0.39999999999999997
- final_gradient_norm: 0.3423870156320877
- gradient_reduction: 0.05761298436791229
- has_nan: False
- has_inf: False
- max_abs_capillary_flux: 1.2055240631880779e-05
- sw_min: 0.3
- sw_max: 0.7
- num_bound_violations: 0

### gravity_zero_density_difference

- success: True
- source: internal zero-density-difference gravity case
- max_abs_gravity_flux: 0.0
- mean_abs_gravity_flux: 0.0
- flux_zero_tolerance: 1e-30
- success: True
- has_nan: False
- has_inf: False

### gravity_segregation_direction

- success: True
- source: internal gravity segregation direction case
- expected_gravity_flux_sign: -1
- observed_gravity_flux_sign: -1
- sign_matches_expectation: True
- top_sw_change: 0.00040861041666662157
- bottom_sw_change: -0.0004086104166666771
- vertical_axis_convention: Grid3D arrays use (nz, ny, nx); positive flux_z is bottom-to-top; rho_w>rho_o gives negative internal gravity_flux_z
- positive_gravity_direction_convention: depth_positive=down
- sw_min: 0.4995913895833333
- sw_max: 0.5004086104166666
- num_bound_violations: 0
- max_abs_gravity_flux: 8.172208333333336e-08
- has_nan: False
- has_inf: False

### combined_capillary_gravity_stability

- success: True
- source: internal combined capillary + gravity stability case
- max_abs_capillary_flux: 1.584936490538905e-08
- max_abs_gravity_flux: 7.880343749999997e-08
- sw_min: 0.34998029914062495
- sw_max: 0.6500059959137229
- num_bound_violations: 0
- material_balance_error: 0.0
- max_cfl: 4.3364059976347246e-05
- has_nan: False
- has_inf: False

### water_flux_composer_consistency

- success: True
- source: internal water_flux_composer consistency case
- pressure_only_flux_norm: 12.68857754044952
- capillary_contribution_norm: 1.4949916387726052
- gravity_contribution_norm: 1.2000000000000002
- combined_flux_norm: 13.613045214058461
- shape_consistent: True
- has_nan: False
- has_inf: False

### opm_spe1case1_capillary_gravity_sanity_adapted

- success: True
- source: OPM/opm-tests spe1 SPE1CASE1.DATA
- porosity_min: 0.3
- porosity_max: 0.3
- permeability_min_md: 50.0
- permeability_max_md: 500.0
- permeability_contrast: 10.0
- sw_min: 0.5
- sw_max: 0.5
- num_bound_violations: 0
- metadata_loaded: True
- has_nan: False
- has_inf: False
