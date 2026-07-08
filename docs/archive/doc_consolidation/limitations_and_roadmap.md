# Limitations and Roadmap

## Current Limitations

The release-candidate prototype does not support:

1. black-oil PVT
2. solution gas / vaporized oil (`Rs` / `Rv`)
3. bubble point or phase appearance / disappearance
4. commercial-grade well controls
5. corner-point grid
6. NNC
7. local grid refinement
8. fully implicit Newton
9. geomechanics
10. thermal model
11. reactive transport
12. real-time frontend communication
13. production-scale parallel simulation

It is also not a commercial reservoir simulator. Petrel-like workflow design,
full frontend protocol design, complex frontend interface, contract-level
business workflow, black-oil, and C++ are deferred.

Current priority is requirement-level function hardening and benchmark
validation. Function hardening first. Workflow design after contract
confirmation.

The three-phase flow design is completed in
`specs/12_three_phase_flow_design.md`, and the independent Corey-style
three-phase relperm / mobility / fractional-flow module and independent
advective phase-flux module are implemented. Independent 1D and 3D three-phase
transport and the YAML/CLI `three_phase_case.yaml` pipeline are implemented.
The three-phase path is simplified incompressible WOG and is not equivalent to black-oil modeling: black-oil PVT, solution gas,
vaporized oil, bubble point handling, and phase appearance / disappearance
remain future work. Three-phase validation/profiling is completed for the small
pipeline case; current small-case profiling does not justify C++ migration.
The three-phase WOG benchmark hardening stage is also completed. It validates
closure, residual bounds, relperm endpoints, mobility / fractional-flow
consistency, phase-flux consistency, 1D/3D explicit transport boundedness, and
production-summary consistency without rewriting the three-phase core modules.

The cross-scale analysis design is completed in
`specs/13_cross_scale_analysis_design.md`. It keeps the product as one backend
with two first-level modules: the computational module and the cross-scale
analysis module. The independent similarity criteria module is implemented for
dimensionless-number comparison and similarity scoring. The independent
scale-effect analysis module is implemented for scale ratios, dominant-force
classification, flow-regime classification, and regime-shift detection. The
lab-field validation module is implemented for curve-to-curve comparison, time
alignment, linear interpolation, and mismatch metrics. Cross-scale
implementation is not yet connected to CLI/YAML. The first implementation does
not perform history matching or automatic parameter calibration, and the
cross-scale modules are not connected to solver internals.

The numerical accuracy benchmark suite is implemented as an MVP accuracy gate.
It validates small deterministic pressure, saturation, capillary, gravity,
combined, three-phase closure, and cross-scale formula cases. This is not
commercial-grade validation; additional experimental and field validation
remains future work.

The saturation transport benchmark hardening stage is completed. It adds
qualitative Buckley-Leverett front movement, MRST buckleyLeverett1D adapted
metadata, boundedness, CFL stability, material balance, areal-like waterflood,
and OPM SPE1CASE1 saturation sanity checks. It does not rewrite the saturation
solver, does not implement a semi-implicit solver, does not implement
black-oil transport, and does not claim full MRST / SPE / OPM Flow equivalence.

The capillary / gravity benchmark hardening stage is completed. It adds
capillary pressure trend, no-gradient capillary zero-flux, capillary smoothing,
zero-density-difference gravity zero-flux, gravity segregation direction,
combined capillary + gravity stability, water-flux composer consistency, and
OPM SPE1CASE1 capillary/gravity property sanity checks. It does not rewrite
capillary, gravity, composer, saturation, or pressure solver internals, and it
does not implement semi-implicit capillary diffusion or black-oil transport.

The parameter fusion benchmark hardening stage is completed. It adds
equal-weight fusion, explicit-weight fusion, confidence-weighted fusion,
documented uncertainty/variance behavior, NaN-aware fusion, bounds/clipping
reports, shape mismatch rejection, and multi-field property/dynamic fusion
sanity checks. It does not implement history matching, automatic calibration,
Bayesian inversion, EnKF / ES-MDA, kriging, Gaussian-process fusion, or
commercial simulator equivalence.

The parameter fusion uncertainty enhancement stage is completed for TASK-016.
It adds variance/std/confidence weighting, lightweight Kriging / GP-style
prediction interface, IDW uncertainty fallback, uncertainty diagnostics,
dominant-source reporting, and EnKF / ES-MDA deferred warnings. It does not
implement history matching, automatic calibration, Bayesian inversion
workflows, complete EnKF, ES-MDA history matching, commercial geostatistical
modeling, or Petrel-like workflow.

The benchmark registry stage is completed. It indexes existing benchmark
summary files, validation levels, reference types, module coverage, open-source
adapted reference metadata, limitations, and overclaim warnings. It is not a
new solver, not a new benchmark algorithm, and not a runtime dependency on OPM
or MRST.

The pressure solver enhancement stage is completed for TASK-011. It adds
simplified rate-controlled well source/sink contribution utilities, boundary
matrix/RHS contribution diagnostics, optional direct/CG/GMRES/ILU/AMG backend
evaluation with graceful fallback, solver stats, and pressure mass-balance
reporting. It does not implement black-oil controls, PVT, a full Peaceman
industrial well model, a complex wellbore network, or a fully implicit
reservoir simulator.

The saturation transport enhancement stage is completed for TASK-014. It adds
CFL adaptive timestep diagnostics, optional 1D TVD/MUSCL benchmark transport,
minmod/van Leer/superbee limiters, front sharpness, total variation,
overshoot/undershoot diagnostics, boundedness checks, material-balance reports,
and implicit-deferred fallback warnings. It preserves the first-order upwind
baseline and does not implement a fully implicit saturation solver, black-oil
transport, PVT, or commercial simulator equivalence.

The performance baseline stage is completed for TASK-019. It measures the
current Python / NumPy / SciPy implementation on small, medium, and large
synthetic pressure, saturation transport, fusion, cross-scale, and benchmark
registry cases. It records runtime, peak Python allocation, array size,
slowest stage, report generation time, and deterministic numerical
equivalence. Current baseline conclusion: C++ and numba migration are not
recommended until larger profiling shows a concrete hotspot. It does not
implement C++, pybind11, numba kernels, or numerical algorithm changes.

The project / case management layer is completed for TASK-056. It adds
file-based Project metadata, Case metadata, RunHistory, report links, result
manifest links, path validation, and `accuracy_reports/project_case_management_summary.json/md`.
It is not a database service, not a frontend, not UDP or REST API, and not a
Petrel-like full workflow. It does not modify solver, inversion, fusion,
cross-scale, data, result, benchmark, reference, config, CLI, API, C++, CMake,
or pybind11 code.

`docs/interface_contract.md` records a future command-style JSON interface
contract. A minimal UDP JSON Archie prototype exists in
`reservoir_backend/api/udp_server.py`, but full UDP product development remains
deferred. CLI, YAML, result directories, and report files remain the primary
interface.

## Engineering Hardening

- packaging
- CI
- API placeholder
- logging
- error handling
- sample data
- release tagging
- reproducible benchmark artifacts
- function benchmark matrix completed
- 046_saturation_inversion_hardening completed
- 047_pressure_solver_benchmark_hardening completed
- 048_saturation_transport_benchmark_hardening completed
- 049_capillary_gravity_benchmark_hardening completed
- 050_three_phase_wog_benchmark_hardening completed
- 051_parameter_fusion_benchmark_hardening completed
- 052_benchmark_registry_hardening completed
- 020_result_export_frontend_field_contract completed
- 003_cross_scale_benchmark_hardening_and_cli_yaml completed
- 011_pressure_solver_wells_boundaries_backends completed
- 014_saturation_transport_tvd_cfl_fallback completed
- F3-04_impes_sequential_loop completed
- 016_parameter_fusion_uncertainty_completed
- F4-04_synthetic_twin_dynamic_field_fusion_completed
- 019_performance_baseline_completed
- 056_project_case_management_completed

## Physics Enhancement

- three-phase design completed
- three-phase relperm implementation completed
- three-phase phase-flux implementation completed
- three-phase 1D transport implementation completed
- three-phase 3D transport implementation completed
- three-phase pipeline case completed
- three-phase validation/profiling completed
- black-oil design
- gas phase
- well model
- pressure solver well source/sink diagnostics completed for simplified
  rate-control wells
- relative permeability tables
- PVT tables
- optional capillary semi-implicit path if needed
- optional TVD/MUSCL 1D saturation transport enhancement completed; 3D
  high-order transport and fully implicit fallback remain future work
- lightweight oil-water IMPES sequential loop completed for synthetic
  waterflood production-curve reporting; fully implicit simulation and
  black-oil behavior remain out of scope
- cross-scale analysis design completed
- similarity criteria module completed
- scale-effect analysis module completed
- lab-field validation module completed
- no history matching or automatic parameter calibration in the MVP
- numerical accuracy benchmark suite completed
- interface contract placeholder completed; minimal UDP Archie prototype exists;
  full UDP product development deferred

## Performance

- larger profiling cases
- sparse solver improvements
- vectorization review
- C++ kernel only if needed
- TASK-019 performance baseline completed; current synthetic cases do not
  justify C++ or numba migration

## Function Hardening Roadmap

- 045_function_benchmark_matrix: completed
- 046_saturation_inversion_hardening: completed
- 047_pressure_solver_benchmark_hardening: completed
- 048_saturation_transport_benchmark_hardening: completed
- 049_capillary_gravity_benchmark_hardening: completed
- 050_three_phase_wog_benchmark_hardening: completed
- 051_parameter_fusion_benchmark_hardening: completed
- 052_benchmark_registry_hardening: completed
- 020_result_export_frontend_field_contract: completed
- 003_cross_scale_benchmark_hardening_and_cli_yaml: completed
- 054_result_delivery_packaging: planned
- 055_black_oil_model_design: planned
- 056_pvt_table_module: planned
- 057_accuracy_acceptance_report: planned

## Interface

Full UDP product development is deferred because the frontend communication
protocol is unknown. The interface direction should be chosen after
requirements clarify whether the backend should expose UDP, TCP, REST, or a
file-based exchange.

TASK-020 adds the file-based result manifest and frontend field contract layer:
`reservoir_backend/results/*`, `docs/result_manifest.md`,
`docs/frontend_field_contract.md`, and `docs/result_export_pipeline.md`.
It supports JSON manifests, CSV metadata summaries, NPZ field-array export, and
Markdown report indexes without implementing a frontend, REST API, UDP changes,
database service, VTK large visualization export, Petrel-like workflow, or
solver rewrite.

TASK-003 adds a cross-scale runner and benchmark report path:
`reservoir_backend/cross_scale/runner.py`, `reservoir_backend/cross_scale/report.py`,
`docs/cross_scale_cli.md`, `docs/cross_scale_validation.md`, and
`accuracy_reports/cross_scale_benchmark_summary.md`. The runner is limited to
similarity criteria, scale-effect analysis, and lab-field curve validation. It
does not implement history matching, automatic calibration, complex upscaling,
frontend, UDP, commercial simulator equivalence, or validation of black-oil models.

TASK-017 adds the cross-scale upscaling report layer:
`reservoir_backend/cross_scale/scale_conversion.py`,
`reservoir_backend/cross_scale/comparison.py`,
`reservoir_backend/cross_scale/upscaling_report.py`, and
`accuracy_reports/cross_scale_upscaling_summary.md`. It documents scale
conversion, lightweight upscaling diagnostics, and fine/coarse comparison
metrics without implementing multiscale finite-volume, history matching,
automatic calibration, frontend, UDP, or black-oil model validation.

TASK-056 adds the project / case management layer under
`reservoir_backend/project/*`. It is limited to file-based metadata and report
indexing for projects, cases, and runs. It does not add database service,
frontend integration, UDP, REST API, or Petrel-like workflow.

F3-04 adds the lightweight IMPES-style sequential loop under
`reservoir_backend/simulation/*` and writes
`accuracy_reports/impes_loop_summary.md`. It reports pressure, flux, Sw, CFL,
material balance, production curve, water cut, and breakthrough time for a
small synthetic oil-water waterflood. It does not implement a fully implicit
simulator, black-oil PVT, complex well controls, frontend integration, UDP,
REST API, or solver-core rewrite.

F4-04 adds synthetic twin dynamic field fusion under
`reservoir_backend/fusion/synthetic_twin.py`,
`reservoir_backend/fusion/dynamic_field_fusion.py`, and
`reservoir_backend/fusion/synthetic_twin_report.py`. It fuses static
permeability/porosity, dynamic pressure/saturation, and production or water-cut
time series while preserving source, confidence, mask, time step, run metadata,
and provenance. It does not implement history matching, EnKF/ES-MDA, automatic
geological model updates, closed-loop digital twin control, frontend
integration, UDP, REST API, or commercial simulator equivalence.
