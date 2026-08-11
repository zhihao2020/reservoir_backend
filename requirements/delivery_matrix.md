# Delivery Matrix

Last updated: 2026-07-03

Status values: `Backlog`, `Designing`, `Coding`, `Testing`, `Validated`,
`Deliverable`.

No requirement is marked `Deliverable` in this audit because final delivery
requires code, tests, documentation, validation reports, and acceptance-level
traceability.

| Requirement ID | Contract / software requirement | Module | Current status | Code | Tests | Validation report | Meets delivery standard | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REQ-001 | Experimental data reading, cleaning, unit normalization, interpolation/resampling, grid mapping | M1 | Coding | `reservoir_backend/io/config_loader.py`, `reservoir_backend/core/units.py`, `reservoir_backend/io/result_manager.py`; `reader.py` reserved | `tests/test_config_loader.py`, `tests/test_units.py`, `tests/test_result_manager.py` | None dedicated | No | Real lab-data readers, anomaly cleaning, resampling, and point-to-grid preprocessing are not implemented |
| REQ-002 | Saturation inversion from resistivity / EM / acoustic signals | M2 | Validated | `reservoir_backend/inversion/*` | `tests/test_archie_inversion.py`, `tests/test_electromagnetic_inversion.py`, `tests/test_acoustic_inversion.py`, `tests/test_saturation_inversion_hardening.py` | `accuracy_reports/saturation_inversion_benchmark_summary.md` | No | EM/acoustic are empirical; no real calibration dataset or final acceptance report |
| REQ-003 | 3D pressure field reconstruction | M3 | Validated | `reservoir_backend/solver/pressure_solver.py`, `transmissibility.py`, `velocity.py`, `pressure_diagnostics.py` | `tests/test_pressure_solver_*.py`, `tests/test_pressure_solver_benchmark_hardening.py` | `accuracy_reports/pressure_solver_benchmark_summary.md` | No | MVP Cartesian validation only; no final field-scale acceptance report |
| REQ-004 | 3D saturation field calculation and displacement simulation | M4 | Validated | `reservoir_backend/solver/saturation_solver.py`, `relperm.py`, `cfl.py`, capillary/gravity/three-phase solver files | `tests/test_saturation_solver_*.py`, `tests/test_saturation_transport_benchmark_hardening.py`, `tests/test_three_phase_*.py` | `accuracy_reports/saturation_transport_benchmark_summary.md`, `validation_reports/combined_validation_summary.md`, `validation_reports/three_phase_validation_summary.md` | No | Explicit transport and small benchmark scope; no final acceptance report |
| REQ-005 | Parameter field fusion for permeability, porosity, pressure, saturation, and inversion results | M5 | Testing | `reservoir_backend/fusion/*` | `tests/test_field_fusion.py`, `tests/test_field_mapper.py`, `tests/test_multisignal_pipeline.py` | `validation_reports/*/fusion_report.json` | No | Dedicated fusion benchmark hardening is still planned |
| REQ-006 | Cross-scale similarity, scale effect, and lab-field comparison | M6 | Testing | `reservoir_backend/cross_scale/*` | `tests/test_similarity_criteria.py`, `tests/test_scale_effect_analysis.py`, `tests/test_lab_field_validation.py` | Cross-scale formula section in `accuracy_reports/accuracy_benchmark_summary.md` | No | Not connected to CLI/YAML; no acceptance-grade experimental/field comparison report |
| REQ-007 | Python backend UDP communication with frontend | M7 | Coding | `reservoir_backend/api/udp_server.py` | `tests/numerical/test_io_and_udp_regression.py` | Regression fixture only | No | Minimal `ping` and `archie_compute` only; missing versioned protocol, request IDs, case execution, status and result commands |
| REQ-008 | Validation, benchmark, and acceptance reporting | M8 | Testing | `tests/test_pipeline_*.py`, `python -m reservoir_backend.pipeline.run` | pipeline tests | `results/sensor_run` 等四场输出 | No | 旧 harness 已移除；以四场 pipeline 测试为准 |
| REQ-009 | Result export and catalog | M8 | Testing | `reservoir_backend/io/result_manager.py`, `writer.py`; `hdf5_export.py` / `vtk_export.py` reserved | `tests/test_result_manager.py` | Result outputs under `results` / validation reports | No | VTK/HDF5 exporters and result catalog are reserved/planned |
| REQ-010 | Black-oil / PVT behavior | M4 | Backlog | Not implemented | Not applicable | Not applicable | No | Current simplified WOG transport is not black-oil |
| REQ-011 | Performance migration / C++ kernels | M8 | Backlog | `specs/09_cpp_migration_spec.md` | profiling tests | profiling reports | No | Current benchmark/profile data does not justify C++ migration |
