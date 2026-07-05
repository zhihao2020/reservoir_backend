# Three-Phase WOG Benchmark Summary

- success: True
- num_cases: 8
- num_passed: 8
- overall_max_closure_error: 2.220446e-16
- overall_num_bound_violations: 0
- overall_fractional_flow_sum_error: 0.000000e+00
- overall_max_phase_flux: 9.765249e-06

## Cases

### three_phase_saturation_closure

- success: True
- source: internal simplified incompressible WOG benchmark
- sw_min: 0.28
- sw_max: 0.46
- so_min: 0.36000000000000004
- so_max: 0.64
- sg_min: 0.08
- sg_max: 0.18
- sw_mean: 0.36999999999999994
- so_mean: 0.5
- sg_mean: 0.13
- closure_max_abs_error: 2.220446049250313e-16
- closure_l2_error: 2.7194799110210365e-16
- has_nan: False
- has_inf: False
- warnings: []
- num_bound_violations: 0

### residual_saturation_bounds

- success: True
- source: internal simplified incompressible WOG benchmark
- sw_min: 0.2
- sw_max: 0.6
- so_min: 0.25
- so_max: 0.75
- sg_min: 0.05
- sg_max: 0.18
- sw_mean: 0.38749999999999996
- so_mean: 0.48750000000000004
- sg_mean: 0.125
- closure_max_abs_error: 1.1102230246251565e-16
- closure_l2_error: 1.1102230246251565e-16
- has_nan: False
- has_inf: False
- warnings: []
- num_bound_violations: 0
- residual_violations: 0
- swi: 0.2
- sor: 0.2
- sgc: 0.05

### three_phase_relperm_endpoint_sanity

- success: True
- source: internal simplified incompressible WOG benchmark
- krw_endpoint: 0.0
- krg_endpoint: 0.0
- kro_low_oil_endpoint: 0.10578512396694209
- krw_monotonicity_score: 1.0
- krg_monotonicity_score: 1.0
- kro_decreases_as_oil_saturation_decreases: True
- krw_min: 0.0
- krw_max: 0.2008264462809917
- kro_min: 0.10578512396694209
- kro_max: 0.8
- krg_min: 0.0
- krg_max: 0.31735537190082647
- has_nan: False
- has_inf: False

### phase_mobility_fractional_flow_consistency

- success: True
- source: internal simplified incompressible WOG benchmark
- krw_min: 0.0063471074380165304
- krw_max: 0.06704132231404958
- kro_min: 0.06770247933884299
- kro_max: 0.5119999999999999
- krg_min: 0.001785123966942148
- krg_max: 0.03352066115702479
- krw_nonnegative: True
- kro_nonnegative: True
- krg_nonnegative: True
- has_nan: False
- has_inf: False
- warnings: []
- lambda_w_min: 6.34710743801653
- lambda_w_max: 67.04132231404958
- lambda_o_min: 13.540495867768598
- lambda_o_max: 102.39999999999998
- lambda_g_min: 178.5123966942148
- lambda_g_max: 3352.066115702479
- lambda_total_min: 287.2595041322313
- lambda_total_max: 3432.647933884297
- lambda_total_positive: True
- fw_min: 0.019530497623211632
- fw_max: 0.022877279609688297
- fo_min: 0.003944621216206848
- fo_max: 0.35647210458479106
- fg_min: 0.6214325170318541
- fg_max: 0.9765248811605816
- fractional_flow_sum_error: 0.0

### phase_flux_finite_shape_consistency

- success: True
- source: internal simplified incompressible WOG benchmark
- max_abs_water_flux: 2.287590814657599e-07
- max_abs_oil_flux: 3.564721045847911e-06
- max_abs_gas_flux: 9.765248811605817e-06
- water_flux_shape: [98]
- oil_flux_shape: [98]
- gas_flux_shape: [98]
- has_nan: False
- has_inf: False
- warnings: []
- phase_flux_closure_error_max: 1.6940658945086007e-21
- flux_shape_x: [2, 3, 5]
- flux_shape_y: [2, 4, 4]
- flux_shape_z: [3, 3, 4]

### three_phase_1d_transport_boundedness

- success: True
- source: internal simplified incompressible WOG benchmark
- sw_min: 0.3
- sw_max: 0.3047877054987962
- so_min: 0.5994116226087066
- so_max: 0.6
- sg_min: 0.0958006718924972
- sg_max: 0.1
- sw_mean: 0.30015959018329325
- so_mean: 0.5999803874202903
- sg_mean: 0.0998600223964166
- closure_max_abs_error: 1.1102230246251565e-16
- closure_l2_error: 5.978733960281817e-16
- has_nan: False
- has_inf: False
- warnings: []
- num_bound_violations: 0
- saturation_change_l1: 0.009575410997592362
- saturation_change_l2: 0.006395519404415221
- max_cfl: 0.01
- water_balance_error: -2.8189256484623115e-18
- oil_balance_error: 1.1140177322288558e-17
- gas_balance_error: -1.0842021724855044e-19

### three_phase_3d_transport_closure

- success: True
- source: internal simplified incompressible WOG benchmark
- sw_min: 0.3
- sw_max: 0.3002393852749398
- so_min: 0.5999705811304352
- so_max: 0.6
- sg_min: 0.09979003359462486
- sg_max: 0.1
- sw_mean: 0.300059846318735
- so_mean: 0.5999926452826089
- sg_mean: 0.09994750839865624
- closure_max_abs_error: 1.1102230246251565e-16
- closure_l2_error: 5.768888059150692e-16
- has_nan: False
- has_inf: False
- warnings: []
- num_bound_violations: 0
- saturation_change_l1: 0.004308934948917556
- saturation_change_l2: 0.0009593279106623554
- max_cfl: 0.0005
- water_balance_error: 3.382710778154774e-17
- oil_balance_error: -1.5486472781239824e-16
- gas_balance_error: -3.7947076036992655e-18

### production_summary_consistency

- success: True
- source: internal simplified incompressible WOG benchmark
- success: True
- sw_min: 0.3
- sw_max: 0.3047877054987962
- so_min: 0.5994116226087066
- so_max: 0.6
- sg_min: 0.0958006718924972
- sg_max: 0.1
- closure_max_abs_error: 1.1102230246251565e-16
- closure_l2_error: 4.839349969133127e-16
- num_bound_violations: 0
- krw_min: 0.00991735537190082
- krw_max: 0.010889715593754484
- kro_min: 0.4218965799006146
- kro_max: 0.42314049586776853
- krg_min: 0.004160730338785156
- krg_max: 0.0049586776859504135
- lambda_total_min: 511.34206545239294
- lambda_total_max: 590.4132231404958
- fractional_flow_sum_error: 0.0
- max_abs_water_flux: 0.0
- max_abs_oil_flux: 0.0
- max_abs_gas_flux: 0.0
- has_nan: False
- has_inf: False
- warnings: []
- water_rate: 1.679731243001119e-07
- oil_rate: 1.433370660694289e-06
- gas_rate: 8.398656215005601e-06
- water_cumulative: 1.679731243001119e-05
- oil_cumulative: 0.0001433370660694289
- gas_cumulative: 0.0008398656215005601
- summary_json_serializable: True
- production_summary_type: pore-volume phase outflow diagnostic, not surface-volume black-oil rate
