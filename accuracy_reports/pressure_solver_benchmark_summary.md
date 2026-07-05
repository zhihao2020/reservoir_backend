# Pressure Solver Benchmark Summary

- success: True
- num_cases: 8
- num_passed: 8
- overall_max_error: 2.980232e-08
- overall_mass_balance_error: 8.282100e-15
- overall_flux_conservation_error: 7.453890e-20

## Cases

### linear_1d_analytical

- success: True
- max_abs_pressure_error: 5.587935447692871e-09
- relative_l2_pressure_error: 5.016031324416748e-16
- max_flux_variation: 7.453889935837843e-20
- mass_balance_error: 8.28209992870868e-15
- pressure_min: 1225000.0000000005
- pressure_max: 9774999.999999998
- has_nan: False
- has_inf: False

### manufactured_2d_linear

- success: True
- max_abs_error: 2.9802322387695312e-08
- l2_error: 1.799911894388302e-07
- relative_l2_error: 3.805046953505174e-15
- linf_error: 2.9802322387695312e-08
- has_nan: False
- has_inf: False
- pressure_min: 2300000.000000004
- pressure_max: 7700000.000000009

### manufactured_3d_linear

- success: True
- max_abs_error: 5.587935447692871e-09
- l2_error: 2.929227883427693e-08
- relative_l2_error: 3.8338905702410085e-16
- linf_error: 5.587935447692871e-09
- has_nan: False
- has_inf: False
- pressure_min: 3499999.999999998
- pressure_max: 8500000.000000006

### opm_water_1ph_adapted

- success: True
- porosity: 0.1
- permeability_x_md: 1000.0
- permeability_y_md: 1000.0
- permeability_z_md: 100.0
- metadata_loaded: True
- pressure_case_mode: metadata_sanity_only
- has_nan: False
- has_inf: False

### opm_spe1case1_layered_adapted

- success: True
- porosity_min: 0.3
- porosity_max: 0.3
- permeability_min_md: 50.0
- permeability_max_md: 500.0
- permeability_contrast: 10.0
- pressure_min: 2399999.9999999995
- pressure_max: 9600000.000000007
- max_abs_flux: 0.0019738466000000137
- mass_balance_error: 2.774473359390406e-16
- has_nan: False
- has_inf: False

### mrst_simple_incomp_tpfa_reference

- success: True
- is_runtime_dependency: False
- mentions_tpfa: True
- mentions_boundary_conditions: True
- mentions_sources: True
- has_nan: False
- has_inf: False

### boundary_sanity

- success: True
- pressure_monotonicity_score: 1.0
- pressure_within_boundary_range: True
- boundary_pressure_left: 10000000.0
- boundary_pressure_right: 0.0
- pressure_min: 416666.666666663
- pressure_max: 9583333.333333327
- warnings: []
- has_nan: False
- has_inf: False

### source_sink_material_balance

- success: True
- total_source: 2e-05
- total_sink: -2e-05
- net_source: 0.0
- boundary_flux_balance: 0.0
- mass_balance_residual: 0.0
- pressure_min: -5938.206056965719
- pressure_max: 2729.8009262638698
- has_nan: False
- has_inf: False
- status: done
