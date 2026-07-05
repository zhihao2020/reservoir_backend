# Numerical Methods

This document summarizes the numerical and physical methods currently visible
in the repository. UDP communication is not a numerical method; see
`docs/udp_protocol.md`.

## Grid and Field Representation

The backend uses structured Cartesian grids with cell-centered unknowns.
`reservoir_backend/core/grid.py` defines `Grid3D`, and
`reservoir_backend/core/field.py` defines `Field3D`.

Face flux arrays follow finite-volume shapes:

- `flux_x`: `(nz, ny, nx + 1)`
- `flux_y`: `(nz, ny + 1, nx)`
- `flux_z`: `(nz + 1, ny, nx)`

## Saturation Inversion

### Archie resistivity inversion

Implemented in `reservoir_backend/inversion/resistivity_archie.py`.

The MVP Archie path uses the relationship:

```text
Rt = a * Rw / (phi^m * Sw^n)
Sw = (a * Rw / (Rt * phi^m))^(1/n)
```

Current implementation supports scalar values, NumPy arrays, and `Field3D`.
It validates positive input values, clips to residual saturation limits, and
can return confidence and sensitivity reports.

Validation evidence:

- `tests/test_archie_inversion.py`
- `tests/test_saturation_inversion_hardening.py`
- `accuracy_reports/saturation_inversion_benchmark_summary.md`

### Electromagnetic and acoustic empirical inversion

Implemented in:

- `reservoir_backend/inversion/electromagnetic.py`
- `reservoir_backend/inversion/acoustic.py`

These modules currently use empirical linear or polynomial calibration. They
are not Maxwell-equation inversion, Gassmann inversion, or full-waveform
inversion. Current benchmark coverage validates MVP behavior, clipping, and
fusion consistency, not full physical inversion.

### Multi-signal fusion

Implemented in `reservoir_backend/inversion/saturation_fusion.py` and
`reservoir_backend/fusion`.

The current approach combines estimates using confidence or uncertainty
weights, then clips saturation to physical bounds. This is a deterministic
weighted fusion method, not Bayesian history matching.

## Pressure Field Reconstruction

Implemented in:

- `reservoir_backend/solver/pressure_solver.py`
- `reservoir_backend/solver/transmissibility.py`
- `reservoir_backend/solver/velocity.py`
- `reservoir_backend/solver/pressure_diagnostics.py`

The pressure module solves steady Cartesian single-phase pressure equations
using a cell-centered finite-volume / TPFA-style discretization. For a cell,
the discrete balance is:

```text
sum_faces T_face * (p_neighbor - p_cell) + q_cell = 0
```

The implementation supports 1D, 2D, and 3D Cartesian cases, Dirichlet
boundaries, no-flow boundaries, and source/sink wells. Face fluxes are derived
from Darcy's law:

```text
q_face = -T_face * (p_neighbor - p_cell)
```

Transmissibility uses harmonic averaging for adjacent-cell permeability.

Validation evidence:

- `tests/test_pressure_solver_1d.py`
- `tests/test_pressure_solver_2d.py`
- `tests/test_pressure_solver_3d.py`
- `tests/test_pressure_solver_benchmark_hardening.py`
- `accuracy_reports/pressure_solver_benchmark_summary.md`

Known scope limit: this is not a finite-element solver, not a fully implicit
black-oil pressure solve, and not OPM/MRST equivalence.

## Saturation Field Calculation

Implemented mainly in `reservoir_backend/solver/saturation_solver.py`.

The oil-water path uses explicit finite-volume updates:

```text
Sw_new = Sw_old - dt / (phi * V) * net_water_flux_out
```

Advective water flux uses upwind fractional flow:

```text
Fw = fw_upwind * Ft
fw = lambda_w / (lambda_w + lambda_o)
lambda_phase = kr_phase / mu_phase
```

Corey-style relative permeability is implemented in
`reservoir_backend/solver/relperm.py`.

Stability is monitored through CFL utilities in
`reservoir_backend/solver/cfl.py`. The explicit solver clips saturation to
configured residual bounds where needed.

Validation evidence:

- `tests/test_saturation_solver_1d.py`
- `tests/test_saturation_solver_3d.py`
- `tests/test_saturation_transport_benchmark_hardening.py`
- `accuracy_reports/saturation_transport_benchmark_summary.md`

### Capillary pressure and capillary flux

Implemented in:

- `reservoir_backend/solver/capillary_pressure.py`
- `reservoir_backend/solver/capillary_flux.py`

The convention is:

```text
Pc = Po - Pw
```

Implemented models include no-capillary, Brooks-Corey, and van Genuchten.
Capillary water flux is computed from face transmissibility, capillary
mobility, and pressure differences.

### Gravity flux

Implemented in `reservoir_backend/solver/gravity_flux.py`.

Depth is positive downward. With `rho_w > rho_o`, water segregation produces
downward movement under the repository's z-face sign convention.

### Combined transport

Implemented through `reservoir_backend/solver/water_flux_composer.py` and
combined saturation functions in `saturation_solver.py`.

```text
Fw_total = Fw_adv + Fw_cap + Fw_grav
```

The effective CFL flux uses the sum of absolute advective, capillary, and
gravity water flux components.

### Simplified three-phase transport

Implemented in:

- `reservoir_backend/solver/three_phase_relperm.py`
- `reservoir_backend/solver/three_phase_flux.py`
- `reservoir_backend/solver/three_phase_transport.py`
- `config/three_phase_case.yaml`

The current three-phase path is simplified incompressible water-oil-gas
transport with closure:

```text
Sw + So + Sg = 1
```

It is not a black-oil model. It does not include PVT tables, solution gas,
vaporized oil, bubble point, or phase appearance/disappearance.

Validation evidence:

- `tests/test_three_phase_relperm.py`
- `tests/test_three_phase_flux.py`
- `tests/test_three_phase_transport_1d.py`
- `tests/test_three_phase_transport_3d.py`
- `tests/test_three_phase_pipeline.py`
- `validation_reports/three_phase_validation_summary.md`

## Parameter Field Fusion

Implemented in:

- `reservoir_backend/fusion/confidence.py`
- `reservoir_backend/fusion/field_mapper.py`
- `reservoir_backend/fusion/field_fusion.py`

Current methods include confidence normalization, nearest-cell mapping, IDW
point-to-grid mapping, same-grid weighted averages, saturation clipping, and
fusion reports.

This module currently has unit and pipeline tests, but no dedicated benchmark
hardening report. It should remain `Testing` until
`049_parameter_fusion_benchmark_hardening` is complete.

## Cross-Scale Analysis

Implemented in:

- `reservoir_backend/cross_scale/descriptors.py`
- `reservoir_backend/cross_scale/similarity.py`
- `reservoir_backend/cross_scale/scale_effect.py`
- `reservoir_backend/cross_scale/validation.py`

Current methods include:

- Reynolds number
- Capillary number
- Peclet number
- Mobility ratio
- Gravity number
- Dimensionless pressure
- Dimensionless time
- weighted similarity score
- scale-ratio reporting
- dominant-force and flow-regime classification
- regime-shift detection
- lab/field curve alignment and mismatch metrics

Validation evidence:

- `tests/test_similarity_criteria.py`
- `tests/test_scale_effect_analysis.py`
- `tests/test_lab_field_validation.py`
- `accuracy_reports/accuracy_benchmark_summary.md`

This module is not yet connected to CLI/YAML and does not implement history
matching or automatic calibration.

## Numerical Validation Metrics

Current reports use:

- max pressure error
- relative pressure error
- L2 / Linf error
- flux conservation error
- mass-balance error
- CFL value
- saturation bounds
- breakthrough/front movement qualitative checks
- three-phase closure error
- formula check error for cross-scale criteria

The current benchmark suite is an MVP numerical gate. It is not yet a final
contract acceptance report.
