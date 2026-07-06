# reservoir_backend Release Candidate v2

## Project Overview

`reservoir_backend` is a lightweight reservoir digital twin backend prototype
for structured Cartesian grid workflows. It is a cell-centered finite-volume /
TPFA prototype for oil-water two-phase studies, inversion-to-simulation demos,
validation harnesses, and regression testing.

This project is not a commercial black-oil simulator and does not currently
implement Petrel workflow, UDP, C++, history matching, or automatic
calibration. Function hardening first. Workflow design after contract
confirmation. The immediate development focus is benchmark validation for each
requirement-level function.

## Current Capabilities

- Structured Cartesian `Grid3D` and `Field3D`
- Experimental data schema, CSV / JSON / NPZ readers, QC pipeline, reusable
  fixtures, and data contract documentation
- Archie resistivity saturation inversion
- Empirical electromagnetic and acoustic saturation inversion
- 1D / 2D / 3D steady pressure solve using finite-volume transmissibility
- Darcy face flux and cell-centered velocity
- Corey relative permeability and fractional flow
- Explicit oil-water saturation transport with CFL checks
- Optional capillary pressure and capillary face flux
- Optional gravity segregation flux
- Optional combined capillary + gravity transport
- Independent three-phase Corey-style relperm / mobility / fractional flow
- Independent three-phase advective phase flux
- Independent three-phase 1D explicit transport
- Independent three-phase 3D explicit transport
- YAML/CLI `three_phase_case.yaml` for simplified incompressible WOG transport
- Three-phase is still not black-oil: no PVT, Rs/Rv, bubble point, or phase
  appearance / disappearance
- cross-scale analysis design for one backend with two first-level modules:
  computational module and cross-scale module
- Similarity criteria are implemented for Reynolds / capillary / Peclet /
  mobility / gravity / dimensionless pressure / dimensionless time comparison
  and similarity scoring
- Scale-effect analysis is implemented for scale ratios, dominant-force
  classification, flow-regime classification, and regime-shift detection
- lab-field validation is implemented for curve-to-curve comparison, time
  alignment, interpolation, and mismatch metrics
- Numerical accuracy benchmarks are implemented for pressure, two-phase
  saturation, capillary, gravity, combined transport, three-phase closure, and
  cross-scale formulas
- An interface contract placeholder exists in `docs/interface_contract.md`;
  a minimal UDP JSON Archie prototype exists in
  `reservoir_backend/api/udp_server.py`, but full frontend protocol development
  is still deferred
- The cross-scale implementation is not yet connected to CLI / YAML
- The cross-scale implementation is not yet complete because CLI/YAML
  integration, UDP exchange, and final acceptance reporting remain planned
- Field fusion with confidence weighting
- CLI case runner, result export, validation, and profiling scripts
- Function benchmark matrix for requirement-level hardening and validation
- Saturation inversion benchmark and hardening for Archie, EM, acoustic, fusion,
  and sensitivity reporting
- Pressure solver benchmark hardening for analytical/manufactured pressure,
  heterogeneous stability, flux conservation, and material balance diagnostics
- Pressure solver enhancement for simplified rate-controlled well source/sink
  terms, boundary matrix/RHS contribution diagnostics, optional linear solver
  backend stats, and pressure mass-balance reporting
- Saturation transport benchmark hardening for Buckley-Leverett qualitative
  front movement, CFL stability, boundedness, material balance, 2D waterflood
  trend checks, and OPM/MRST adapted reference metadata
- Saturation transport enhancement for CFL adaptive timestep diagnostics,
  optional 1D TVD/MUSCL transport benchmarks, front sharpness, boundedness,
  fallback warnings, and mass-balance reporting
- IMPES-style pressure-saturation sequential loop for small oil-water
  synthetic waterfloods, including pressure, flux, Sw, CFL, material balance,
  production curve, water cut, and breakthrough-time reporting
- Capillary / gravity benchmark hardening for capillary pressure trend,
  capillary smoothing, gravity segregation direction, combined stability,
  water-flux composer consistency, and OPM SPE1 adapted property sanity
- Three-phase WOG benchmark hardening for saturation closure, residual bounds,
  relperm endpoints, mobility / fractional-flow consistency, phase-flux
  consistency, 1D/3D transport boundedness, and production summary consistency
- Parameter fusion benchmark hardening for shape consistency, weighted fusion,
  confidence weighting, NaN/mask handling, bounds, provenance, and fusion
  error reports
- Parameter fusion uncertainty enhancement for variance/std/confidence
  weighting, lightweight Kriging/GP-style interface, IDW uncertainty fallback,
  diagnostics, and EnKF/ES-MDA deferred warnings
- Synthetic twin dynamic field fusion for static permeability/porosity,
  dynamic pressure/saturation, production or water-cut time series,
  confidence/mask/provenance tracking, and truth-error diagnostics
- Unified benchmark registry for saturation inversion, pressure, saturation
  transport, capillary/gravity, three-phase, and parameter fusion benchmark
  summaries
- Workflow design is deferred until contract details are confirmed

## Installation / Environment

```bash
cd reservoir_backend
pip install -e .
```

Python dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Quick Start

Run the default small case:

```bash
python scripts/run_case.py --config config/demo_case.yaml
```

Run the combined capillary + gravity case:

```bash
python scripts/run_case.py --config config/combined_case.yaml
```

Run the simplified three-phase WOG case:

```bash
python scripts/run_case.py --config config/three_phase_case.yaml
```

Dry-run a case without writing simulation results:

```bash
python scripts/run_case.py --config config/combined_case.yaml --dry-run
```

## CLI Usage

Supported entry points:

```bash
python scripts/run_case.py --config config/demo_case.yaml
python -m reservoir_backend.cli.run_case --config config/demo_case.yaml
```

Supported arguments:

- `--config`
- `--output-dir`
- `--case-id`
- `--mode`
- `--dry-run`
- `--verbose`

See `docs/cli_usage.md`.

## Available Cases

- `config/demo_case.yaml`: Archie-only base pipeline
- `config/multisignal_case.yaml`: resistivity / EM / acoustic signal fusion
- `config/capillary_case.yaml`: capillary transport enabled
- `config/capillary_gradient_case.yaml`: nonuniform Sw capillary validation
- `config/gravity_case.yaml`: gravity transport enabled
- `config/combined_case.yaml`: combined capillary + gravity transport enabled
- `config/three_phase_case.yaml`: simplified incompressible water-oil-gas
  advective transport

See `docs/case_configuration.md`.

## Output Files

Typical output directories are under `results/<case_id>/`. The full pipeline
can write pressure, saturation, velocity, face fluxes, production curves,
material-balance reports, fusion reports, solver reports, capillary reports,
gravity reports, combined reports, and case summaries.

## Validation

```bash
pytest -q
python harness/run_validation.py
python scripts/validate_combined_pipeline.py
python scripts/validate_three_phase_pipeline.py
```

Current release-candidate result:

- `pytest -q`: 1137 passed
- combined validation success: true
- material_balance_error: 0.0

See `docs/validation_and_profiling.md`.

## Profiling

```bash
python scripts/profile_full_pipeline.py
python scripts/profile_capillary_pipeline.py
python scripts/profile_combined_pipeline.py
python scripts/profile_three_phase_pipeline.py
python scripts/run_accuracy_benchmarks.py
```

Current combined profiling result:

- combined_case runtime approximately 0.07 s
- combined/demo runtime ratio approximately 1.23x
- base max_cfl approximately 0.163

Current recommendation: no C++ yet. C++ is planned only after larger-scale
profiling shows a concrete bottleneck.

Three-phase validation/profiling is also available for `three_phase_case.yaml`.
It checks `Sw + So + Sg = 1`, saturation bounds, CFL, material balance, dt
sensitivity, and records runtime for demo / combined / three-phase cases.
Current small-case recommendation: no C++ and no black-oil escalation yet.

Numerical accuracy benchmark reports are written to
`accuracy_reports/accuracy_benchmark_summary.json` and
`accuracy_reports/accuracy_benchmark_summary.md`. The suite is an MVP accuracy
gate, not commercial-grade validation. Additional experimental validation
remains future work.

The saturation inversion benchmark is available through:

```bash
python benchmarks/saturation_inversion_benchmark.py
```

It validates Archie analytical recovery, noise sensitivity, empirical EM /
acoustic clipping behavior, and uncertainty-weighted multi-signal fusion. It
does not implement Bayesian inversion, automatic calibration, machine-learning
inversion, or commercial petrophysical interpretation.

The pressure solver benchmark is available through:

```bash
python benchmarks/pressure_solver_benchmark.py
```

It validates 1D analytical pressure, 2D/3D manufactured linear pressure,
OPM water-1ph adapted metadata sanity, OPM SPE1CASE1 layered adapted pressure,
MRST simpleIncompTPFA reference-note coverage, source/sink material balance,
and boundary sanity. It is not full SPE1/SPE10 reproduction and does not claim
OPM Flow equivalence, MRST integration, or runtime dependency on OPM/MRST.

The saturation transport benchmark is available through:

```bash
python benchmarks/saturation_transport_benchmark.py
```

It validates a qualitative Buckley-Leverett 1D waterflood, MRST
buckleyLeverett1D adapted metadata, saturation boundedness, CFL stability,
material balance, an areal-like waterflood trend case, and OPM SPE1CASE1
saturation sanity metadata. It is not full MRST reproduction, not full SPE1 or
SPE10 reproduction, not OPM Flow equivalence, not black-oil transport, and has
no runtime dependency on OPM or MRST.

The lightweight IMPES sequential loop report is available through:

```bash
python -m reservoir_backend.simulation.impes_report
```

It writes `accuracy_reports/impes_loop_summary.json` and `.md` for a small
synthetic oil-water waterflood. The loop follows
`pressure -> flux -> saturation -> mobility -> pressure`, reports production
curve, water cut, breakthrough time, CFL, and material balance, and does not
implement a fully implicit simulator, black-oil model, complex well controls,
frontend, UDP, or REST API.

The capillary / gravity benchmark is available through:

```bash
python benchmarks/capillary_gravity_benchmark.py
```

It validates capillary pressure monotonicity, no-gradient capillary zero flux,
capillary smoothing, zero-density-difference gravity zero flux, gravity
segregation direction, combined capillary + gravity stability, water-flux
composer consistency, and OPM SPE1CASE1 property-driven sanity. It does not
implement a semi-implicit capillary solver, black-oil transport, full
SPE1/SPE10 reproduction, OPM Flow equivalence, MRST integration, or runtime
dependency on OPM/MRST.

The simplified three-phase WOG benchmark is available through:

```bash
python benchmarks/three_phase_benchmark.py
```

It validates WOG closure, residual bounds, Corey-style relperm endpoint
sanity, phase mobility, fractional-flow closure, phase-flux shape/finite
consistency, 1D/3D explicit transport boundedness, and simplified production
summary consistency. It is not black-oil, does not implement PVT tables,
Bo/Bw/Bg, Rs/Rv, bubble point, or phase appearance / disappearance, and does
not claim OPM Flow or commercial simulator equivalence.

The parameter fusion benchmark is available through:

```bash
python benchmarks/parameter_fusion_benchmark.py
```

It validates equal-weight and explicit-weight field fusion,
confidence-weighted fusion, NaN-aware masking behavior, saturation clipping,
shape mismatch rejection, and multi-field property/dynamic fusion sanity. It
does not implement history matching, automatic calibration, Bayesian inversion,
EnKF / ES-MDA, kriging, Gaussian-process fusion, black-oil, or commercial
simulator equivalence.

The synthetic twin dynamic field fusion report is available through:

```bash
python -m reservoir_backend.fusion.synthetic_twin_report
```

It writes `accuracy_reports/fusion_synthetic_twin_summary.json` and `.md` for
a deterministic synthetic twin fixture. The report fuses static permeability
and porosity fields, dynamic pressure and saturation fields, production /
water-cut time series, source metadata, confidence, masks, provenance, and
optional synthetic truth error metrics. It does not implement history matching,
EnKF / ES-MDA, automatic geological model update, closed-loop digital twin
control, frontend, UDP, or REST API.

The unified benchmark registry is available through:

```bash
python benchmarks/benchmark_registry.py
```

It indexes existing benchmark summaries, validation levels, reference types,
module coverage, report paths, limitations, and open-source adapted reference
metadata. It does not rerun solvers, parse upstream OPM/MRST files, introduce
runtime dependencies, or claim full SPE1/SPE10 reproduction, OPM Flow
equivalence, MRST integration, commercial simulator equivalence, or black-oil
validation.

The performance baseline is available through:

```bash
python -m reservoir_backend.performance.performance_report
```

It writes `accuracy_reports/performance_baseline_summary.json` and
`accuracy_reports/performance_baseline_summary.md` with small / medium / large
synthetic runtime, memory, slowest-stage, report-generation, and numerical
equivalence metrics for pressure, saturation transport, parameter fusion,
cross-scale reports, and benchmark registry aggregation. Current TASK-019
baseline recommendation: no C++ and no numba migration yet.

Experimental data fixtures are available under:

```text
tests/fixtures/experimental_data/
```

The fixture catalog covers valid CSV core fields, valid JSON multimodal fields,
valid NPZ grid fields, missing required fields, invalid units and bounds,
duplicate time/coordinates, and NaN / Inf / missing values. See
`docs/data_contract.md` for supported formats, required/optional fields, units,
shape conventions, expected QC behavior, and limitations.

Cross-scale analysis now includes independent similarity criteria and
scale-effect analysis modules. The design keeps Requirements 1 and 2 in one
Reservoir Digital Twin Backend rather than splitting them into two software
products. The `cross_scale` package currently computes dimensionless criteria,
similarity reports, scale ratios, flow-regime classifications, regime-shift
reports, and lab-field curve validation reports as pure functions. It will not
perform history matching or automatic parameter calibration in the MVP, and it
is not yet connected to CLI / YAML.

The interface contract is documented, and a minimal UDP JSON Archie prototype
is regression-tested. Full frontend/backoffice exchange should continue through
CLI, YAML, result directories, and reports until the final frontend protocol is
known.

The result manifest and frontend field contract are documented in
`docs/result_manifest.md`, `docs/frontend_field_contract.md`, and
`docs/result_export_pipeline.md`. The `reservoir_backend.results` package
provides a lightweight result manifest, catalog, report path index, JSON
manifest export, CSV metadata summary export, NPZ field-array export, and
Markdown report index. This is a file-based contract layer only; it does not
implement a frontend, UDP changes, REST API, database service, or solver
rewrite.

The project / case management layer is documented in
`docs/project_case_management.md` and implemented in
`reservoir_backend.project`. It provides lightweight project metadata, case
metadata, run history, report path validation, and generated
`accuracy_reports/project_case_management_summary.json/md` evidence. It is not
a database service, frontend, UDP / REST API, or Petrel-like workflow.

Cross-scale benchmark hardening is available through
`reservoir_backend.cross_scale.runner`. It reads dict / JSON / YAML
configuration, runs similarity criteria, scale-effect analysis, and lab-field
curve validation, writes `accuracy_reports/cross_scale_benchmark_summary.json`
and `.md`, and records a result manifest entry for the cross-scale report. It is
not history matching, automatic calibration, a complex upscaling solver, UDP, or
frontend integration.

The cross-scale upscaling report is available through
`reservoir_backend.cross_scale.upscaling_report`. It writes
`accuracy_reports/cross_scale_upscaling_summary.json/md` with scale conversion,
similarity criteria, lightweight upscaling assumptions, and synthetic
fine-grid/coarse-grid comparison metrics. It is not a multiscale finite-volume
solver and does not perform history matching or automatic calibration.

The current priority is requirement-level function hardening. The function
benchmark matrix maps each requirement function to its algorithm, inputs,
outputs, candidate benchmark, validation metric, and next hardening task.
Petrel-style workflow design and complex frontend/business process design are
deferred until contract details are confirmed.

## Numerical Method Summary

The prototype uses structured Cartesian grids, cell-centered unknowns,
face-centered flux arrays, TPFA-style transmissibility, finite-volume pressure
balance, upwind fractional flow, explicit saturation updates, CFL checks, and
material-balance reporting.

Optional combined transport uses:

```text
Fw_total = Fw_adv + Fw_cap + Fw_grav
```

Current recommendation: no semi-implicit capillary diffusion yet. It becomes a
candidate only if strong capillary pressure, fine grids, or dt sensitivity show
explicit-step instability or impractically small time steps.

See `docs/numerical_methods.md`.

## Limitations

The current CLI pipeline supports a simplified incompressible WOG three-phase
case, but it does not support black-oil PVT, solution gas / vaporized oil,
bubble point, phase appearance / disappearance, commercial-grade well controls, corner-point grids, NNC, local grid
refinement, fully implicit Newton coupling, geomechanics, thermal models,
reactive transport, production-scale parallel simulation, completed
cross-scale analysis implementation, history matching, automatic parameter
calibration, or real-time frontend communication. UDP is deferred because the
frontend protocol is unknown.

See `docs/limitations_and_roadmap.md`.

## Roadmap

Near-term work should continue function hardening. Completed:
`046_saturation_inversion_hardening`,
`047_pressure_solver_benchmark_hardening`,
`048_saturation_transport_benchmark_hardening`,
`049_capillary_gravity_benchmark_hardening`, and
`050_three_phase_wog_benchmark_hardening`, and
`051_parameter_fusion_benchmark_hardening`, and
`052_benchmark_registry_hardening`, and
`020_result_export_frontend_field_contract`. Next planned stages are cross-scale
delivery packaging, black-oil design, and PVT tables. C++ kernels should be
considered only when profiling justifies them.

## Documentation Index

- `docs/architecture.md`
- `docs/numerical_methods.md`
- `docs/data_schema.md`
- `docs/data_contract.md`
- `docs/experimental_data_pipeline.md`
- `docs/result_manifest.md`
- `docs/project_case_management.md`
- `docs/frontend_field_contract.md`
- `docs/result_export_pipeline.md`
- `docs/cross_scale_cli.md`
- `docs/cross_scale_validation.md`
- `docs/cross_scale_upscaling_report.md`
- `docs/case_configuration.md`
- `docs/cli_usage.md`
- `docs/validation_and_profiling.md`
- `docs/numerical_accuracy.md`
- `docs/three_phase_validation.md`
- `docs/parameter_fusion_validation.md`
- `docs/benchmark_registry.md`
- `docs/open_source_benchmark_references.md`
- `docs/performance_baseline.md`
- `docs/function_benchmark_matrix.md`
- `docs/benchmark_selection_policy.md`
- `docs/interface_contract.md`
- `docs/limitations_and_roadmap.md`
- `docs/module_matrix.md`
- `docs/release_checklist.md`
