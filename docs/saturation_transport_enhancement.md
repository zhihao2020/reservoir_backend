# Saturation Transport Enhancement

## Purpose

TASK-014 adds optional saturation-transport enhancement utilities for CFL
adaptive timestep diagnostics, TVD/MUSCL 1D benchmark paths, boundedness/front
diagnostics, mass-balance reporting, and fallback policy validation.

## Upwind Baseline

The existing first-order upwind finite-volume saturation solver remains the
baseline. `method="upwind"` delegates to `advance_saturation_1d` and preserves
the existing baseline behavior.

## CFL Calculation

The enhancement layer wraps the existing `reservoir_backend.solver.cfl`
calculation with:

- `compute_cfl`
- `suggest_stable_timestep`
- `adapt_timestep`

Reports include `max_cfl`, `target_cfl`, `dt_original`, `dt_adapted`, and
`num_limited_cells`.

## Adaptive Timestep

When a proposed explicit timestep exceeds the target CFL, the helper returns a
smaller `dt_adapted`. The default behavior is report-and-adapt rather than
raising. Existing `check_cfl_condition` behavior and the default CFL module API
are unchanged.

## TVD / MUSCL Optional Path

`reservoir_backend.solver.tvd_transport` implements an opt-in 1D high-resolution
path for benchmark hardening:

- `method="tvd"`
- `method="muscl"`
- `method="upwind"`
- `method="implicit"` with scope-warning fallback behavior

Three-dimensional high-order transport remains future work.

## Limiter Choice

`reservoir_backend.solver.limiters` provides:

- minmod
- van Leer
- superbee

The default limiter for the TVD/MUSCL benchmark path is minmod.

## Diagnostics

`reservoir_backend.solver.transport_diagnostics` reports:

- overshoot
- undershoot
- front position
- front sharpness
- total variation
- material balance error
- max CFL
- num clipped cells
- boundedness status

## Fallback Policy

- If CFL is too large, adaptive timestep is preferred.
- If TVD/MUSCL produces nonphysical values, the path can fall back to upwind or
  clip with a warning.
- If implicit transport is requested, the report returns a scope warning and
  uses upwind fallback when configured. It does not claim an implicit solver is
  implemented.

## Report Schema

Run:

```bash
python -m reservoir_backend.solver.saturation_transport_enhancement_report
```

Outputs:

- `accuracy_reports/saturation_transport_enhancement_summary.json`
- `accuracy_reports/saturation_transport_enhancement_summary.md`

The report contains CFL cases, upwind-vs-TVD comparison cases, fallback cases,
boundedness diagnostics, front-sharpness metrics, material-balance metrics,
warnings, and limitations.

## Limitations

- Upwind baseline is preserved.
- TVD/MUSCL is optional.
- TVD/MUSCL support is currently limited to 1D benchmark scenarios.
- Implicit solver is deferred and remains outside the current implementation scope.
- No black-oil.
- No PVT.
- No commercial simulator equivalence.
- No history matching.
- No front-end.
- No UDP.

## Non-Claims

This stage does not implement a fully implicit reservoir simulator, black-oil
transport, PVT, commercial simulator equivalence, frontend integration, or UDP.
