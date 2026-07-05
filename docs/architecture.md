# Architecture

The current product direction is requirement-level function hardening and
benchmark validation. Workflow design is deferred until contract details are
confirmed.

## Project Structure

- `reservoir_backend/core`: grid, field, state, well, unit, and exception types.
- `reservoir_backend/inversion`: Archie, electromagnetic, and acoustic
  saturation inversion interfaces.
- `reservoir_backend/solver`: transmissibility, pressure, velocity, relperm,
  CFL, saturation, capillary, gravity, and water-flux composition modules.
- `reservoir_backend/fusion`: confidence handling, field mapping, and weighted
  field fusion.
- `reservoir_backend/io`: configuration loading and result export helpers.
- `reservoir_backend/cli`: YAML-driven command line runner.
- `examples`: deterministic runnable demo pipelines.
- `scripts`: user-facing run, validation, and profiling scripts.
- `config`: small YAML cases.
- `tests`: unit, regression, numerical, CLI, and harness tests.
- `specs`: staged requirement and migration specifications.
- `docs`: release-candidate user and developer documentation.

## Main Backend Chain

```text
Rt / EM / acoustic signals
-> saturation inversion
-> pressure solve
-> face flux
-> saturation transport
-> field fusion
-> result export
-> reports
```

## Current Hardening Boundary

The backend currently has computational, fusion, cross-scale, reporting,
validation, and benchmark modules. It does not implement a Petrel workflow,
business process layer, or complex frontend interface. CLI / YAML / reports
remain the primary interface until the contract workflow is confirmed.

## Combined Transport Chain

```text
pressure flux
-> advective water flux
-> capillary flux
-> gravity flux
-> water_flux_composer
-> saturation update
-> material balance
```

## Runtime Boundary

The CLI and examples orchestrate existing modules. They should not duplicate
solver logic. Result output is file-based under `results/<case_id>/`; generated
outputs are intentionally ignored by Git except for placeholder `.gitkeep`
files.
