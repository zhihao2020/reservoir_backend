# Project Overview

Last updated: 2026-07-03

## Project Goal

This repository implements a Python backend prototype for reservoir digital twin
and reservoir experiment data processing workflows. The current implementation
focuses on structured Cartesian grid computation, saturation inversion,
pressure reconstruction, saturation transport, field fusion, cross-scale
analysis utilities, CLI/YAML execution, and validation/benchmark reporting.

The repository is the engineering source of truth. Notion is used only as a
project dashboard and planning surface.

## Repository Scan

- Repository root used for this audit: `D:\Code\oil\reservoir_backend`
- Git branch / commit during audit: `main` / `70757f9`
- Remote: `https://github.com/zhihao2020/reservoir_backend.git`
- Python package: `reservoir_backend`
- Entry scripts: `main.py`, `scripts/run_case.py`, `python -m reservoir_backend.cli.run_case`
- Source directories:
  - `reservoir_backend/core`: grid, field, state, wells, units, exceptions
  - `reservoir_backend/io`: config loader, result manager, reserved readers/exporters
  - `reservoir_backend/inversion`: Archie, electromagnetic, acoustic, multi-signal saturation fusion
  - `reservoir_backend/solver`: pressure, velocity, transmissibility, CFL, saturation, capillary, gravity, three-phase helpers
  - `reservoir_backend/fusion`: confidence weighting, field mapping, dynamic state fusion
  - `reservoir_backend/cross_scale`: scale descriptors, similarity criteria, scale-effect and curve validation utilities
  - `reservoir_backend/api`: minimal UDP Archie server plus reserved API facade files
  - `reservoir_backend/cli`: YAML-driven case runner
- Test directories:
  - `tests`: unit, pipeline, validation, benchmark hardening tests
  - `tests/numerical`: numerical and UDP regression tests
  - `tests/regression/references`: reference `.json` / `.npz` fixtures
- Benchmark/report directories:
  - `benchmarks`: pressure, saturation inversion, saturation transport, cross-scale formula and related benchmark scripts
  - `accuracy_reports`: current benchmark summaries
  - `validation_reports`: validation summaries and generated arrays/reports
  - `profiling_reports`: generated profiling outputs
- Documentation/spec directories:
  - `docs`: architecture, numerical methods, case config, CLI, validation, benchmark policy, release checklist
  - `specs`: staged implementation and traceability specs
  - `references`: adapted open-source reference metadata and fixtures

## Current Development Status

The codebase is beyond a pure demo: core numerical modules have tests and
benchmark reports. However, it is still a prototype, not a commercial reservoir
simulator or contract-ready product. Current status by area:

- Saturation inversion: `Validated` for MVP synthetic/analytical checks; EM and acoustic paths remain empirical, not full physics inversion.
- Pressure reconstruction: `Validated` for Cartesian TPFA finite-volume benchmarks and adapted reference metadata checks.
- Saturation transport: `Validated` for explicit oil-water, capillary, gravity, combined, and simplified three-phase smoke/benchmark cases.
- Parameter field fusion: `Testing`; unit tests exist, but dedicated benchmark hardening is still planned.
- Cross-scale analysis: `Testing`; formula and curve utilities exist, but CLI/YAML integration and acceptance reports are not complete.
- UDP interface: `Coding`; a minimal JSON UDP Archie server and regression test exist, but full request/response protocol, status querying, and result transfer workflow are not implemented.
- Experimental data processing: `Coding`; YAML config, units, and result IO exist, but real experimental readers/cleaning/resampling pipelines are not implemented.
- Validation/benchmark reporting: `Testing`; benchmark summaries exist, but final acceptance report and delivery matrix closure are still planned.

Current test result from this audit:

```text
python -m pytest -q --basetemp=.pytest-tmp
1137 passed in 48.17s
```

## Overall Architecture

```text
YAML / synthetic inputs / small signal arrays
-> config and unit normalization
-> saturation inversion
-> pressure reconstruction
-> Darcy flux and velocity
-> saturation transport
-> field fusion
-> reports, result arrays, validation summaries
```

Cross-scale modules currently run as independent utility functions:

```text
lab descriptor + field descriptor + curves
-> dimensionless criteria
-> similarity score
-> scale-effect report
-> curve mismatch metrics
```

The current UDP path is separate and minimal:

```text
UDP JSON request
-> ping or Archie saturation compute
-> UDP JSON response
```

## Current Risks

- README and older docs had stale statements: README still recorded `585 passed`, while the audit run found `1137 passed`; `docs/interface_contract.md` said no UDP server existed while `reservoir_backend/api/udp_server.py` does exist.
- Several high-level documents use `Done`, but delivery-grade status should remain below `Deliverable` unless code, tests, docs, and benchmark/validation reports all exist.
- EM/acoustic saturation inversion is empirical and should not be presented as full Maxwell, Gassmann, or waveform inversion.
- Pressure and transport benchmarks are MVP-scale and adapted-reference based; they do not prove OPM Flow/MRST equivalence or commercial simulator accuracy.
- Explicit transport is the main numerical stability risk for stronger capillary/gravity cases or finer grids.
- Cross-scale utilities are not connected to CLI/YAML and do not perform history matching or automatic calibration.
- UDP protocol lacks request ID, protocol version, retry semantics, large-result handling, and frontend workflow integration.
- Real experimental data import, anomaly cleaning, resampling, and lab-signal-to-grid mapping remain underdeveloped.

## Next Stage Focus

1. Harden parameter fusion benchmarks and reports (`049_parameter_fusion_benchmark_hardening`).
2. Harden cross-scale benchmarks and connect cross-scale reports into CLI/YAML.
3. Replace UDP placeholder contract with a versioned request/response protocol and case execution commands.
4. Implement experimental data reader/cleaning/resampling pipeline for real lab data files.
5. Produce an acceptance-oriented numerical accuracy report that links requirements, code, tests, benchmark cases, and known limitations.
