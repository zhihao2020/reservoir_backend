# Three-Phase WOG Validation

Three-phase WOG benchmark hardening: Done

This page documents the dedicated benchmark hardening stage for the simplified
incompressible water-oil-gas path. It validates the existing WOG relperm,
mobility, fractional-flow, phase-flux, and explicit transport modules without
rewriting their algorithms.

## Benchmark Coverage

- three-phase saturation closure benchmark: Done
- residual saturation bounds benchmark: Done
- three-phase relperm endpoint sanity benchmark: Done
- phase mobility / fractional flow consistency benchmark: Done
- phase flux finite / shape consistency benchmark: Done
- 1D WOG transport boundedness benchmark: Done
- 3D WOG transport closure benchmark: Done
- production summary consistency benchmark: Done

The benchmark runner is:

```bash
python benchmarks/three_phase_benchmark.py
```

It writes:

- `accuracy_reports/three_phase_benchmark_summary.json`
- `accuracy_reports/three_phase_benchmark_summary.md`

## Diagnostics

The diagnostics module is `reservoir_backend/solver/three_phase_diagnostics.py`.
It reports:

- `Sw`, `So`, and `Sg` min / max
- closure errors for `Sw + So + Sg = 1`
- bound violations
- Corey relperm ranges
- phase mobility ranges
- fractional-flow closure
- phase-flux magnitudes and shapes
- NaN / Inf flags

## Current Results

The current small deterministic benchmark suite passes all eight WOG cases.
Representative results:

- closure error is near machine precision
- bound violations are zero
- fractional-flow sum error is zero
- phase fluxes are finite
- 1D and 3D explicit transport remain bounded under stable dt
- production summary is JSON serializable and nonnegative under the current
  outflow convention

## Explicit Non-Claims

Current model is simplified incompressible WOG, not black-oil.
No PVT table implemented.
No Bo / Bw / Bg implemented.
No Rs / Rv implemented.
No bubble point implemented.
No phase appearance / disappearance implemented.
No OPM Flow equivalence.
No commercial simulator equivalence.
No solver core rewrite.
No UDP development.
No C++ development.

The production summary is a pore-volume phase outflow diagnostic. It is not a
surface-volume black-oil production report.
