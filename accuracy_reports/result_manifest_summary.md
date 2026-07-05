# Result Manifest Summary

- success: True
- num_results: 5
- num_reports: 9
- num_missing_reports: 0

## Result Catalog

| result_id | module | result_type | field_name | format | path |
| --- | --- | --- | --- | --- | --- |
| pressure_field_demo | M3 | pressure_field | pressure | npy | results/demo_case/pressure.npy |
| saturation_field_demo | M4 | saturation_field | sw | npy | results/demo_case/sw_simulated.npy |
| parameter_fusion_report | M5 | parameter_fusion_report | fusion_summary | json | accuracy_reports/parameter_fusion_benchmark_summary.json |
| experimental_data_qc_report | M1 | experimental_data_qc | qc_summary | json | accuracy_reports/experimental_data_qc_summary.json |
| benchmark_registry_report | M8 | benchmark_registry | registry_summary | json | accuracy_reports/benchmark_registry_summary.json |

## Report Path Index

| path | format | result_type | exists |
| --- | --- | --- | --- |
| accuracy_reports/experimental_data_qc_summary.json | json | experimental_data_qc | True |
| accuracy_reports/experimental_data_qc_summary.md | md | experimental_data_qc | True |
| accuracy_reports/saturation_inversion_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/pressure_solver_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/saturation_transport_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/capillary_gravity_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/three_phase_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/parameter_fusion_benchmark_summary.json | json | benchmark_summary | True |
| accuracy_reports/benchmark_registry_summary.json | json | benchmark_registry | True |
