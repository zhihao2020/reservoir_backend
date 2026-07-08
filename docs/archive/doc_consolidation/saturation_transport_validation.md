# Saturation Transport Validation

## Status

Saturation transport benchmark hardening: Done

- 1D Buckley-Leverett qualitative benchmark: Done
- MRST buckleyLeverett1D adapted reference: Done
- saturation boundedness benchmark: Done
- CFL stability benchmark: Done
- material balance benchmark: Done
- 2D areal waterflood qualitative benchmark: Done
- OPM SPE1CASE1 saturation sanity adapted benchmark: Done
- Saturation transport enhancement / TASK-014: Done

## Scope

The benchmark hardens the current oil-water explicit saturation transport path:

- Corey relative permeability and fractional flow
- upwind finite-volume water flux
- explicit time stepping
- CFL diagnostics
- material balance diagnostics
- boundedness diagnostics

The benchmark does not change `saturation_solver.py`. It only calls existing
solver APIs and records diagnostics.

## Benchmark Cases

| Case | Purpose | Source |
| ---- | ------- | ------ |
| `buckley_leverett_1d_qualitative` | Checks downstream front movement, inlet Sw increase, bounds, CFL, and material balance. | MRST `buckleyLeverett1D.m` benchmark idea |
| `mrst_buckley_leverett_1d_reference` | Loads extracted grid, permeability, and porosity metadata. | MRST `buckleyLeverett1D.m` fixture |
| `saturation_boundedness` | Runs uniform, step, random, near-residual, and near-maximum Sw cases. | Internal deterministic cases |
| `cfl_stability` | Records stable, near-limit, and too-large dt behavior. | Internal CFL diagnostic case |
| `material_balance_1d` | Checks injected minus produced water against storage change. | Internal 1D transport case |
| `areal_waterflood_2d_qualitative` | Checks areal-like injection-region Sw increase and boundedness. | Internal thin Cartesian case |
| `opm_spe1case1_saturation_sanity_adapted` | Loads layered SPE1 property fixture and checks saturation bounds. | OPM `SPE1CASE1.DATA` adapted metadata |

## Open-Source Reference Policy

No black-oil transport implemented.
No full MRST reproduction.
No OPM Flow equivalence.
No full SPE1 or SPE10 reproduction.
No solver core rewrite.
No runtime dependency on OPM or MRST.
No semi-implicit solver implemented.

## TASK-014 Saturation Transport Enhancement

TASK-014 adds an optional enhancement layer without replacing the validated
upwind baseline:

- CFL adaptive timestep diagnostics
- minmod / van Leer / superbee limiter utilities
- optional 1D TVD / MUSCL benchmark transport path
- overshoot / undershoot, front position, front sharpness, total variation,
  boundedness, and material-balance diagnostics
- implicit-request deferred warning with upwind fallback when configured

Manual run:

```bash
python -m reservoir_backend.solver.saturation_transport_enhancement_report
```

Outputs:

- `accuracy_reports/saturation_transport_enhancement_summary.json`
- `accuracy_reports/saturation_transport_enhancement_summary.md`

The upwind baseline is preserved. TVD/MUSCL is optional and currently limited
to 1D benchmark scenarios. This stage does not implement a fully implicit
saturation solver, black-oil transport, PVT, frontend integration, or UDP.

The MRST and OPM files are reference materials only. The benchmark reads the
already extracted fixtures under `references/fixtures/` and does not modify or
regenerate upstream reference files.

## Outputs

Run:

```bash
python benchmarks/saturation_transport_benchmark.py
```

The runner writes:

- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.md`

The summary contains case-level front movement, CFL, boundedness, material
balance, NaN/Inf, and reference-policy metadata.

## Interpretation

These cases are small deterministic regression benchmarks. They show that the
current supported oil-water transport path is finite, bounded, CFL-controlled,
and materially balanced for simple cases. They do not prove commercial-grade
history matching, black-oil behavior, or exact reproduction of MRST / OPM Flow.
