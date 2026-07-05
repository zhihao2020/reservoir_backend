# Pressure Solver Validation

This document records the 047 pressure solver benchmark hardening stage.

## Status

- Pressure solver benchmark hardening: Done
- 1D linear analytical benchmark: Done
- 2D manufactured benchmark: Done
- 3D manufactured benchmark: Done
- OPM water-1ph adapted reference: Done
- OPM SPE1CASE1 layered adapted benchmark: Done
- MRST simpleIncompTPFA reference note: Done
- boundary sanity benchmark: Done
- source/sink material balance benchmark: Done
- flux / mass balance diagnostics: Done
- pressure solver enhancement / TASK-011: Done

## Implemented Scope

The benchmark suite validates the current structured Cartesian finite-volume /
TPFA pressure reconstruction path. It does not add a new pressure solver and it
does not rewrite the existing solver implementation.

The benchmark is implemented in `benchmarks/pressure_solver_benchmark.py` and
checks:

- 1D linear pressure against an analytical Dirichlet solution
- 2D manufactured linear pressure field
- 3D manufactured linear pressure field
- OPM water-1ph adapted metadata sanity from extracted fixture
- OPM SPE1CASE1 layered adapted pressure benchmark from extracted fixture
- MRST simpleIncompTPFA method reference note from extracted fixture
- source/sink material balance using existing balanced well support
- boundary sanity for left-to-right pressure decline

The diagnostics helper `reservoir_backend.solver.pressure_diagnostics` computes
statistics, pressure error metrics, flux conservation metrics, scalar mass
balance residuals, and JSON-serializable reports. It does not modify pressure
or flux arrays.

## Open Benchmark Boundary

This stage uses analytical, manufactured, and open-source-adapted reference
benchmarks. It reads already extracted fixtures in `references/fixtures`; it
does not parse OPM decks directly and does not modify `references/upstream` or
`references/fixtures`. It is not:

- No finite element solver implemented.
- No black-oil pressure model implemented.
- No full SPE1 or SPE10 reproduction.
- No OPM Flow equivalence.
- No MRST integration.
- No runtime dependency on OPM or MRST.
- No solver core rewrite.

The OPM water-1ph case is metadata sanity only because the 1x1x1 extracted
reference cannot form an internal pressure gradient in this benchmark mode. The
OPM SPE1CASE1 case is a layered Cartesian adapted pressure benchmark, not an
exact SPE1 simulation reproduction. MRST simpleIncompTPFA is recorded as a
method reference note only; MATLAB/MRST is not executed.

## Manual Run

```bash
python benchmarks/pressure_solver_benchmark.py
```

Outputs:

- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.md`

## TASK-011 Pressure Enhancement

TASK-011 adds an engineering enhancement layer without replacing the existing
pressure solver:

- simplified rate-controlled injector / producer source-sink contribution
  vectors
- Dirichlet / Neumann / no-flow boundary matrix and RHS contribution
  diagnostics
- direct / CG / GMRES / ILU / AMG backend evaluation with graceful fallback
- solver stats, residual norm, fallback status, and mass-balance diagnostics

Manual run:

```bash
python -m reservoir_backend.solver.pressure_enhancement_report
```

Outputs:

- `accuracy_reports/pressure_solver_enhancement_summary.json`
- `accuracy_reports/pressure_solver_enhancement_summary.md`

This enhancement does not implement black-oil controls, PVT, a full Peaceman
industrial well model, complex wellbore networks, a fully implicit simulator,
front-end integration, or UDP.
