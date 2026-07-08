# Project Status

## Status Taxonomy

This file is the only maintained source for module status.

Allowed status values:

- **Validated**: implementation is present and has targeted tests plus
  benchmark or report evidence suitable for the current mainline scope.
- **Testing**: implementation is present, but the interface, examples, or
  validation coverage still needs broader cases before it should be treated as a
  mainline assumption.
- **Coding**: implementation work is active or the interface is not yet stable.
- **Deferred**: outside the current MVP scope, intentionally postponed, or only
  recorded as future direction.

## Module Status

| Area | Scope | Status | Evidence | Scope Notes |
|---|---|---:|---|---|
| Core grid and field model | Structured Cartesian grids, field containers, units, well records | Validated | `tests/test_core_grid.py`, `tests/test_core_field.py`, `tests/test_units.py`, `tests/test_wells.py` | Structured-grid scope only. |
| Experimental data entry | CSV, JSON, NPZ readers, schema checks, QC, fixtures | Testing | `tests/test_experimental_data_pipeline.py`, `tests/test_experimental_data_fixtures.py`, `accuracy_reports/experimental_data_qc_summary.*` | Needs more real lab datasets before broader claims. |
| Field data ingestion | Well table, production history, pressure history, schedule CSV, and property field input summaries | Testing | `tests/test_field_data_ingestion.py`, `accuracy_reports/field_data_ingestion_summary.*` | File-based inputs only; no database service or commercial data platform. |
| Saturation inversion | Archie and multi-signal inversion utilities | Validated | `tests/test_saturation_inversion_hardening.py`, `accuracy_reports/saturation_inversion_benchmark_summary.*` | Empirical EM/acoustic paths remain lightweight. |
| Pressure reconstruction | TPFA pressure solve, well source terms, boundary utilities, backend stats | Validated | `tests/test_pressure_solver_benchmark_hardening.py`, `tests/test_pressure_solver_enhancement.py`, `accuracy_reports/pressure_solver_*summary.*` | No finite-element solver and no industrial well model. |
| Darcy flux and velocity | Face fluxes, velocity diagnostics, transmissibility helpers | Validated | `tests/test_velocity.py`, `tests/test_transmissibility.py`, pressure benchmark reports | Structured-grid Darcy flow only. |
| Oil-water saturation transport | Explicit finite-volume transport, CFL diagnostics, optional TVD/MUSCL helpers | Validated | `tests/test_saturation_transport_benchmark_hardening.py`, `tests/test_saturation_transport_enhancement.py`, `accuracy_reports/saturation_transport_*summary.*` | Fully implicit simulator is not implemented. |
| Capillary and gravity transport utilities | Capillary pressure/flux, gravity flux, combined water-flux diagnostics | Validated | `tests/test_capillary_gravity_benchmark_hardening.py`, capillary/gravity report artifacts | Explicit diagnostics and benchmark cases only. |
| Simplified IMPES loop | Pressure to flux to saturation coupling for small synthetic waterflood cases | Testing | `tests/test_impes_loop.py`, `accuracy_reports/impes_loop_summary.*` | Synthetic examples only; not a full reservoir simulator. |
| Simplified WOG three-phase utilities | Relperm, phase flux, transport checks, production summaries | Testing | `tests/test_three_phase_benchmark_hardening.py`, `accuracy_reports/three_phase_benchmark_summary.*` | Incompressible WOG utilities, not black-oil PVT. |
| Parameter fusion and uncertainty | Field fusion, uncertainty weighting, lightweight spatial fallback, synthetic-twin summaries | Testing | `tests/test_parameter_fusion_benchmark_hardening.py`, `tests/test_parameter_fusion_uncertainty.py`, `tests/test_fusion_synthetic_twin.py` | No history matching or ensemble assimilation workflow. |
| Cross-scale analysis | Similarity criteria, scale effects, lab-field curve validation, upscaling report layer | Testing | `tests/test_cross_scale_benchmark_cli.py`, `tests/test_cross_scale_upscaling_report.py`, cross-scale report artifacts | Reporting layer only; no multiscale FV solver. |
| Result and project management | Result manifest, report index, project/case/run registries | Testing | `tests/test_result_export_contract.py`, `tests/test_project_case_management.py` | File-based registries, no database service. |
| Industrial case workflow v0 | Case config to Project/Case/Run to IMPES to production summary and engineering report | Testing | `tests/test_industrial_case_workflow.py`, `accuracy_reports/industrial_case_workflow_summary.*` | Synthetic structured-grid workflow only. |
| Benchmark registry and performance baseline | Report aggregation, benchmark index, runtime and memory summaries | Testing | `tests/test_benchmark_registry_hardening.py`, `tests/test_performance_baseline.py` | Registry reads existing reports; it is not a solver. |
| CLI and case configuration | Lightweight script entrypoints and YAML case examples | Testing | `tests/test_cli_run_case.py`, config loader tests | CLI surface is intentionally small. |
| UDP, REST API, and front-end integration | Product API and UI integration | Deferred | Front-end field contract documents expected fields | No service layer in current MVP. |
| Black-oil, PVT, history matching, full SPE reproduction, C++ kernels | Advanced simulator and acceleration directions | Deferred | Roadmap and limitation docs | Not part of the current validated Python backend scope. |
