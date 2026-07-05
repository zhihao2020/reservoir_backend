# Pressure Solver Enhancement

## Purpose

TASK-011 adds pressure-solver engineering utilities for simplified well
source/sink terms, boundary contribution diagnostics, optional linear-solver
backend evaluation, solver statistics, and mass-balance reporting.

This stage does not rewrite the existing pressure solver. The existing
finite-volume / TPFA pressure path remains the baseline.

## Well Source / Sink Model

The enhancement layer defines a single-cell rate-controlled well model in
`reservoir_backend.solver.well_source`.

Supported fields:

- `well_id`
- `cell_index` or `i`, `j`, `k`
- `well_type`: `injector` or `producer`
- `control_type`: `rate`
- `rate`
- `unit`
- `phase`
- `metadata`

Only rate control is implemented. BHP controls, multi-segment wells, wellbore
networks, and full Peaceman industrial well modeling are outside this stage.

## Sign Convention

Positive source terms inject into the reservoir. Injector rates are assembled as
positive RHS/source contributions. Producer rates are assembled as negative
RHS/source contributions.

The well contribution vector uses the repository's x-fastest Cartesian cell
indexing:

```text
index = k * ny * nx + j * nx + i
```

## Boundary Matrix Contribution

`reservoir_backend.solver.boundary_matrix` provides standalone diagnostic
contribution arrays for:

- Dirichlet boundary: adds transmissibility to the adjacent-cell diagonal and
  `T * p_boundary` to the RHS.
- Neumann boundary: adds the configured flux value to the RHS; positive values
  inject into the domain.
- No-flow boundary: adds no matrix or RHS contribution.
- Source / sink terms: added to RHS by an explicit source vector.

These helpers are diagnostic utilities. They do not replace the pressure
solver's existing matrix assembly.

## Solver Backend and Fallback Policy

`reservoir_backend.solver.linear_solver_backend` exposes a lightweight wrapper
for:

- `direct`
- `cg`
- `gmres`
- `ilu`
- `amg`

`direct` is the baseline. ILU uses SciPy's `spilu` when available. AMG attempts
to use `pyamg` only if installed. Optional backend failures fall back to the
direct solver and record `fallback_used=true` with a warning. No hard AMG
dependency is introduced.

## Solver Stats

Solver stats include:

- `backend`
- `requested_backend`
- `success`
- `num_iterations`
- `residual_norm`
- `mass_balance_error`
- `flux_conservation_error`
- `warnings`
- `fallback_used`

All stats are JSON serializable.

## Mass-Balance Diagnostics

The enhancement report records:

- total injection rate
- total production rate
- net source rate
- source-vector mass-balance residual
- boundary RHS contribution diagnostics
- backend residual diagnostics

These are engineering diagnostics and do not add new pressure physics.

## Report Schema

The report runner is:

```bash
python -m reservoir_backend.solver.pressure_enhancement_report
```

Outputs:

- `accuracy_reports/pressure_solver_enhancement_summary.json`
- `accuracy_reports/pressure_solver_enhancement_summary.md`

The JSON report contains:

- `report_name`
- `success`
- `num_cases`
- `well_cases`
- `boundary_cases`
- `solver_backend_cases`
- `mass_balance_residuals`
- `solver_stats`
- `limitations`
- `warnings`

## Limitations

- No black-oil.
- No PVT.
- No full Peaceman industrial well model.
- No complex wellbore network.
- No fully implicit reservoir simulator.
- No history matching.
- No front-end.
- No UDP.

## Non-Claims

This stage does not claim commercial simulator equivalence, industrial well
control parity, OPM Flow equivalence, MRST integration, or a new pressure
solution algorithm.
