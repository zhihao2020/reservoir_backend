# Function Benchmark Matrix

Current stage priority is requirement-level function hardening and benchmark
validation. Function hardening first. Workflow design after contract
confirmation.

The authoritative matrix is maintained in
`specs/14_function_benchmark_matrix.md`. It covers:

1. Saturation inversion module
2. Pressure field reconstruction module
3. Saturation transport module
4. Capillary / gravity enhancement module
5. Simplified three-phase WOG module
6. Parameter field fusion module
7. Cross-scale similarity module
8. Scale-effect analysis module
9. Lab-field validation module
10. Result reporting module
11. Future interface module
12. Future black-oil extension

Each function entry defines:

- Requirement source
- Functional objective
- Input data
- Output data
- Current algorithm
- Current implementation status
- Known limitations
- Candidate benchmark
- Validation metric
- Next hardening task
- Priority

This document deliberately does not define a Petrel workflow. Workflow design is
deferred until contract details, frontend behavior, data formats, and acceptance
paths are confirmed.

## 046 Saturation Inversion Hardening

The first hardening stage is complete. It adds:

- Archie analytical validation and stricter input checks
- empirical EM linear / polynomial inversion validation
- empirical acoustic linear / polynomial inversion validation
- uncertainty-weighted multi-signal saturation fusion
- Archie finite-difference sensitivity reports
- saturation inversion synthetic benchmark reports

The implementation does not add Bayesian inversion, automatic calibration,
machine-learning inversion, commercial petrophysical interpretation, or solver
changes.

## 047 Pressure Solver Benchmark Hardening

The pressure hardening stage is complete. It adds:

- 1D linear analytical pressure benchmark
- 2D manufactured linear pressure benchmark
- 3D manufactured linear pressure benchmark
- OPM water-1ph adapted reference metadata sanity
- OPM SPE1CASE1 layered adapted pressure benchmark
- MRST simpleIncompTPFA reference note
- source/sink material balance benchmark using existing well support
- boundary sanity benchmark
- pressure diagnostics for statistics, error metrics, flux conservation, and
  mass balance residuals

The implementation does not add finite element solving, black-oil pressure
modeling, full SPE1/SPE10 reproduction, OPM Flow equivalence, MRST integration,
runtime dependency on OPM/MRST, corner-point grids, or pressure solver core
rewrites.

## 048 Saturation Transport Benchmark Hardening

The saturation transport hardening stage is complete. It adds:

- 1D Buckley-Leverett qualitative waterflood benchmark
- MRST buckleyLeverett1D adapted reference metadata check
- saturation boundedness benchmark
- CFL stability benchmark with explicit too-large-dt diagnostics
- material balance benchmark
- 2D areal-like waterflood qualitative benchmark
- OPM SPE1CASE1 saturation sanity adapted metadata benchmark
- saturation diagnostics for statistics, bounds, front position, change norm,
  CFL statistics, and material-balance residuals

The implementation does not rewrite the saturation solver, does not implement
semi-implicit transport, does not implement black-oil transport, does not claim
full MRST reproduction, full SPE1/SPE10 reproduction, OPM Flow equivalence, or
MRST integration, and does not introduce OPM/MRST runtime dependencies.

## 049 Capillary / Gravity Benchmark Hardening

The capillary / gravity hardening stage is complete. It adds:

- capillary pressure monotonicity benchmark
- capillary no-gradient zero-flux benchmark
- capillary smoothing benchmark
- gravity zero-density-difference zero-flux benchmark
- gravity segregation direction benchmark
- combined capillary + gravity stability benchmark
- water flux composer consistency benchmark
- OPM SPE1 capillary / gravity sanity adapted benchmark
- diagnostics for gradient norm, flux statistics, expected flux signs,
  capillary smoothing, gravity segregation, and combined transport metrics

The implementation does not modify capillary pressure, capillary flux, gravity
flux, water flux composer, pressure solver, saturation solver, CLI/YAML, or
reference fixtures. It does not implement semi-implicit capillary diffusion,
black-oil transport, full SPE1/SPE10 reproduction, OPM Flow equivalence, MRST
integration, or OPM/MRST runtime dependency.

## 050 Three-Phase WOG Benchmark Hardening

The simplified three-phase WOG hardening stage is complete. It adds:

- three-phase saturation closure benchmark
- residual saturation boundedness benchmark
- Corey-style relperm endpoint sanity benchmark
- phase mobility and fractional-flow consistency benchmark
- phase flux finite / shape consistency benchmark
- 1D WOG transport boundedness benchmark
- 3D WOG transport closure benchmark
- production summary consistency benchmark
- diagnostics for closure, phase bounds, relperm, mobility, fractional flow,
  phase fluxes, transport change, and NaN / Inf flags

The implementation does not modify three-phase relperm, phase flux, transport,
pressure solver, saturation solver, CLI/YAML, or reference fixtures. It does
not implement black-oil, PVT tables, Bo/Bw/Bg, Rs/Rv, bubble point, phase
appearance / disappearance, OPM Flow equivalence, commercial simulator
equivalence, UDP, or C++.

## 051 Parameter Fusion Benchmark Hardening

The parameter fusion hardening stage is complete. It adds:

- equal-weight field fusion benchmark
- explicit-weight field fusion benchmark
- confidence-weighted fusion benchmark
- uncertainty / variance behavior documentation
- NaN-aware fusion benchmark
- bounds and clipping report benchmark
- shape mismatch rejection benchmark
- multi-field property / dynamic fusion sanity benchmark
- fusion diagnostics for statistics, finite-value checks, shape consistency,
  bounds, weight ranges, NaN / mask reporting, fusion error, and confidence
  influence

The implementation does not modify solver, inversion, cross-scale, IO, CLI,
YAML, API, script, or reference fixture behavior. It does not implement history
matching, automatic calibration, Bayesian inversion, EnKF / ES-MDA, kriging,
Gaussian-process fusion, black-oil, UDP, C++, or commercial simulator
equivalence.

Task anchor: `051_parameter_fusion_benchmark_hardening`.

## 052 Benchmark Registry Hardening

The unified benchmark registry stage is complete. It adds:

- registry loading for existing benchmark summary JSON files
- module / task / requirement mapping for completed hardening benchmarks
- validation level taxonomy:
  `analytical`, `manufactured_solution`, `adapted_open_source_reference`,
  `diagnostic_sanity`, `property_metadata_sanity`, `trend_validation`,
  `stability_validation`
- reference type taxonomy:
  `exact reproduction`, `adapted reference`, `reference context only`,
  `property metadata sanity only`, `internal benchmark`
- open-source reference metadata registration for OPM water-1ph, OPM
  SPE1CASE1, MRST simpleIncompTPFA, and MRST buckleyLeverett1D
- overclaim checks for full SPE reproduction, OPM/MRST equivalence, commercial
  simulator equivalence, black-oil overstatements, history matching, and
  automatic calibration claims
- JSON and Markdown registry reports

The implementation does not modify solver, inversion, fusion, cross-scale, IO,
CLI, YAML, API, script, or reference fixture behavior. It does not parse
upstream decks, introduce OPM/MRST runtime dependencies, or add new algorithms.

Task anchor: `052_benchmark_registry_hardening`.
