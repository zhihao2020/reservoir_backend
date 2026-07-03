# reservoir_backend Release Candidate v2

## Project Overview

`reservoir_backend` is a lightweight Python reservoir-backend prototype for
structured Cartesian grid workflows. It is a cell-centered finite-volume / TPFA
prototype for oil-water two-phase studies, inversion-to-simulation demos,
validation harnesses, and regression testing.

This project is not a commercial black-oil simulator, not a Petrel replacement,
and not a CMG replacement. It is intended as a transparent numerical backend
prototype with small, reproducible cases and pytest coverage.

## Current Capabilities

- Structured Cartesian `Grid3D` and `Field3D`
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
- The cross-scale implementation is not yet complete; similarity criteria,
  scale-effect analysis, and lab-field validation remain planned
- Field fusion with confidence weighting
- CLI case runner, result export, validation, and profiling scripts

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

- `pytest -q`: 585 passed
- combined validation success: true
- material_balance_error: 0.0

See `docs/validation_and_profiling.md`.

## Profiling

```bash
python scripts/profile_full_pipeline.py
python scripts/profile_capillary_pipeline.py
python scripts/profile_combined_pipeline.py
python scripts/profile_three_phase_pipeline.py
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

Cross-scale analysis is currently design-only. The design keeps Requirements 1
and 2 in one Reservoir Digital Twin Backend rather than splitting them into two
software products. The future `cross_scale` module will read computational
outputs and produce similarity, scale-effect, mapping, validation, and
cross-scale reports. It will not perform history matching or automatic
parameter calibration in the MVP.

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

Near-term work should focus on engineering hardening, CI, packaging, API
stabilization, larger profiling, and clearer sample datasets. Physics expansion
such as three-phase validation/profiling, black-oil design, well controls, PVT
tables, cross-scale similarity criteria, scale-effect analysis, lab-field
validation, and relative-permeability tables should remain separate design stages.
C++ kernels should be considered only when profiling justifies them.

## Documentation Index

- `docs/architecture.md`
- `docs/numerical_methods.md`
- `docs/case_configuration.md`
- `docs/cli_usage.md`
- `docs/validation_and_profiling.md`
- `docs/limitations_and_roadmap.md`
- `docs/module_matrix.md`
- `docs/release_checklist.md`
