# Numerical Accuracy Benchmark Suite

The numerical accuracy benchmark suite is an MVP accuracy gate for the Python
backend. It is not commercial simulator validation and does not replace
comparison against real core-flood, pilot, or field data.

The current priority is requirement-level function hardening. Benchmark
validation is the next development driver. Workflow design is deferred until
contract details are confirmed.

## Benchmarks

- `pressure_linear_1d`: checks a homogeneous 1D Dirichlet pressure field,
  constant flux, and small mass-balance error.
- `pressure_manufactured_3d`: checks a 3D Cartesian pressure solve against a
  manufactured linear field.
- `pressure_solver_benchmark`: checks 1D analytical pressure, 2D/3D
  manufactured linear pressure, OPM water-1ph adapted metadata sanity, OPM
  SPE1CASE1 layered adapted pressure, MRST simpleIncompTPFA reference-note
  coverage, source/sink material balance, boundary sanity, and pressure
  diagnostics.
- `buckley_leverett_1d`: checks qualitative two-phase waterflood behavior:
  inlet Sw increase, downstream front movement, CFL, bounds, and material
  balance.
- `saturation_transport_benchmark`: checks qualitative Buckley-Leverett front
  movement, MRST buckleyLeverett1D adapted metadata, boundedness, CFL
  stability, material balance, areal-like waterflood trend behavior, and OPM
  SPE1CASE1 saturation sanity metadata.
- `capillary_smoothing`: checks that a capillary-enabled saturation step is
  smoothed and produces nonzero capillary flux.
- `gravity_segregation`: checks that water moves downward when water is denser
  than oil and that gravity flux has the expected sign.
- `combined_transport_stability`: checks combined capillary + gravity stability,
  nonzero capillary/gravity fluxes, CFL, bounds, and material balance.
- `capillary_gravity_benchmark`: checks capillary pressure monotonicity,
  capillary no-gradient zero flux, capillary smoothing, gravity zero-density
  zero flux, gravity segregation direction, combined capillary + gravity
  stability, water flux composer consistency, and OPM SPE1 property-driven
  sanity.
- `three_phase_closure`: checks simplified WOG closure, bounds, and phase
  material-balance errors.
- `three_phase_benchmark`: checks simplified WOG closure, residual bounds,
  Corey relperm endpoints, mobility, fractional-flow closure, phase-flux
  finite/shape consistency, 1D boundedness, 3D closure, and production summary
  consistency.
- `parameter_fusion_benchmark`: checks equal-weight fusion, explicit-weight
  fusion, confidence-weighted fusion, documented uncertainty/variance behavior,
  NaN-aware masking, bounds/clipping reports, shape mismatch rejection, and
  multi-field property/dynamic fusion sanity.
- `benchmark_registry`: indexes completed benchmark summaries, validation
  levels, reference types, module coverage, report paths, limitations, and
  adapted open-source reference metadata.
- `cross_scale_formula_check`: checks Re, Ca, Pe, similarity score, scale
  ratios, RMSE, MAE, and R2 against known values.
- `saturation_inversion_benchmark`: checks Archie analytical formula recovery,
  1% / 5% / 10% resistivity noise sensitivity, uncertainty-weighted
  multi-signal fusion, and empirical inversion clipping reports.

## How To Run

```bash
python scripts/run_accuracy_benchmarks.py
```

The runner writes:

- `accuracy_reports/accuracy_benchmark_summary.json`
- `accuracy_reports/accuracy_benchmark_summary.md`

The saturation inversion benchmark can also be run directly:

```bash
python benchmarks/saturation_inversion_benchmark.py
```

It writes:

- `accuracy_reports/saturation_inversion_benchmark_summary.json`
- `accuracy_reports/saturation_inversion_benchmark_summary.md`

The pressure solver benchmark can also be run directly:

```bash
python benchmarks/pressure_solver_benchmark.py
```

It writes:

- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.md`

This benchmark is analytical / manufactured / open-source-adapted only. It is
not full SPE1 or SPE10 reproduction, not OPM Flow equivalence, not MRST
integration, and not a runtime dependency on OPM or MRST.

The saturation transport benchmark can also be run directly:

```bash
python benchmarks/saturation_transport_benchmark.py
```

It writes:

- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.md`

This benchmark is qualitative / diagnostic / open-source-adapted only. It is
not full MRST reproduction, not full SPE1 or SPE10 reproduction, not OPM Flow
equivalence, not black-oil transport, and not a runtime dependency on OPM or
MRST.

The capillary / gravity benchmark can also be run directly:

```bash
python benchmarks/capillary_gravity_benchmark.py
```

It writes:

- `accuracy_reports/capillary_gravity_benchmark_summary.json`
- `accuracy_reports/capillary_gravity_benchmark_summary.md`

This benchmark is trend / stability / diagnostic only. It is not a
semi-implicit capillary solver, not black-oil transport, not full SPE1/SPE10
reproduction, not OPM Flow equivalence, not MRST integration, and not a runtime
dependency on OPM or MRST.

The three-phase WOG benchmark can also be run directly:

```bash
python benchmarks/three_phase_benchmark.py
```

It writes:

- `accuracy_reports/three_phase_benchmark_summary.json`
- `accuracy_reports/three_phase_benchmark_summary.md`

This benchmark is limited to simplified incompressible WOG diagnostics. It is
not black-oil, does not implement PVT tables, Bo/Bw/Bg, Rs/Rv, bubble point, or
phase appearance / disappearance, and does not claim OPM Flow or commercial
simulator equivalence.

The parameter fusion benchmark can also be run directly:

```bash
python benchmarks/parameter_fusion_benchmark.py
```

It writes:

- `accuracy_reports/parameter_fusion_benchmark_summary.json`
- `accuracy_reports/parameter_fusion_benchmark_summary.md`

This benchmark validates existing deterministic field fusion behavior. It is
not history matching, automatic calibration, Bayesian inversion, EnKF / ES-MDA,
kriging, Gaussian-process fusion, black-oil modeling, or commercial simulator
equivalence.

The unified benchmark registry can also be run directly:

```bash
python benchmarks/benchmark_registry.py
```

It writes:

- `accuracy_reports/benchmark_registry_summary.json`
- `accuracy_reports/benchmark_registry_summary.md`

This registry reads existing benchmark summaries and extracted reference
fixtures. It is not a new solver, not a new benchmark algorithm, not full SPE1
or SPE10 reproduction, not OPM Flow equivalence, not MRST integration, and not
a runtime dependency on OPM or MRST.

## Interpretation

These benchmarks validate small deterministic cases, formula implementations,
no NaN / Inf behavior, material balance, and repeatability. They are intended
to catch regressions before larger validation studies.

Additional experimental validation remains future work. More real laboratory
and field data are needed before claiming commercial-grade validation.

The current suite does not recommend C++ migration: the benchmark cases are
small and stable. C++ should remain gated by larger profiling evidence.

The current suite does not justify black-oil escalation. The executable model
remains a lightweight prototype with simplified oil-water and simplified WOG
paths, not a black-oil simulator.
