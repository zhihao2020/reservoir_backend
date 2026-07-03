# Combined Capillary + Gravity Transport Design

## 1. Scope

This design covers only oil-water two-phase saturation transport with combined
capillary and gravity water-flux terms on the existing Cartesian `Grid3D`
finite-volume path.

Out of scope:

- three-phase flow
- black-oil or compositional models
- PVT coupling
- well model changes
- fully implicit Newton coupling
- C++ kernels
- UDP, frontend, or protocol work

The function-level combined solver and the configuration-driven combined
pipeline are implemented. The pipeline entry point uses
`advance_saturation_3d_with_capillary_and_gravity(...)` when capillary and
gravity are both enabled.

## 2. Existing Solver Paths

Current saturation transport entry points are:

- `advance_saturation_3d(...)`
- `advance_saturation_3d_with_capillary(...)`
- `advance_saturation_3d_with_gravity(...)`
- `advance_saturation_3d_with_capillary_and_gravity(...)`

Current production behavior supports four explicit paths:

- capillary disabled, gravity disabled: `advance_saturation_3d(...)`
- capillary enabled, gravity disabled: `advance_saturation_3d_with_capillary(...)`
- capillary disabled, gravity enabled: `advance_saturation_3d_with_gravity(...)`
- capillary enabled, gravity enabled: `advance_saturation_3d_with_capillary_and_gravity(...)`

Configuration validation rejects inconsistent flags. For example,
`capillary_pressure.enabled=true` requires `saturation.use_capillary=true`, and
`gravity.enabled=true` requires `saturation.use_gravity=true`.

## 3. Proposed Combined Flux

Future combined transport should use:

```text
Fw_total = Fw_adv + Fw_cap + Fw_grav
```

where:

- `Fw_adv = fw_upwind * total_flux`
- `Fw_cap = capillary_flux`
- `Fw_grav = gravity_flux`

The combined update should reuse the existing explicit finite-volume
saturation update:

```text
Sw_new[cell] = Sw_old[cell] - dt / (phi[cell] * V[cell]) * net_water_flux_out
```

## 4. Sign Convention

Face flux directions follow the existing project convention:

- `flux_x > 0`: left -> right
- `flux_y > 0`: front -> back
- `flux_z > 0`: bottom -> top

Capillary pressure convention:

- `Pc = Po - Pw`
- `qcap_x = T_abs * Mcap * (Pc_right - Pc_left)`
- `qcap_y = T_abs * Mcap * (Pc_back - Pc_front)`
- `qcap_z = T_abs * Mcap * (Pc_top - Pc_bottom)`

Gravity convention:

- depth is positive downward
- if `rho_w > rho_o`, water moves downward
- `flux_z > 0` means bottom -> top
- therefore if `rho_w > rho_o`, internal `gravity_flux_z < 0`

## 5. CFL Strategy

The first combined path should keep the explicit method and use a conservative
CFL estimate:

```text
effective_flux_x = abs(total_flux_x) + abs(capillary_flux_x) + abs(gravity_flux_x)
effective_flux_y = abs(total_flux_y) + abs(capillary_flux_y) + abs(gravity_flux_y)
effective_flux_z = abs(total_flux_z) + abs(capillary_flux_z) + abs(gravity_flux_z)
```

Risks:

- strong capillary pressure can create diffusion-like fluxes that require much
  smaller explicit time steps
- fine grids reduce pore volume and tighten CFL limits
- strong density contrast can increase gravity segregation flux
- future high-capillary cases may need semi-implicit capillary diffusion instead
  of a purely explicit update

## 6. Material Balance

Material balance must be computed from the combined water flux, not just the
advective flux. The report must include at least:

- `injected_water_volume`
- `produced_water_volume`
- `storage_change`
- `material_balance_error`

## 7. Report Schema

`combined_report.json` includes at least:

- `capillary_enabled`
- `gravity_enabled`
- `capillary_model`
- `rho_w`
- `rho_o`
- `density_difference`
- `max_advective_flux`
- `max_capillary_flux`
- `max_gravity_flux`
- `max_total_water_flux`
- `max_effective_flux`
- `max_cfl`
- `material_balance_error`
- `capillary_flux_included`
- `gravity_flux_included`
- `has_nan`
- `has_inf`

## 8. YAML Behavior

Combined transport is enabled by `config/combined_case.yaml` or equivalent YAML:

```yaml
capillary_pressure:
  enabled: true

gravity:
  enabled: true

saturation:
  use_capillary: true
  use_gravity: true
```

Current behavior:

- enabling capillary alone is allowed
- enabling gravity alone is allowed
- enabling both capillary and gravity is allowed when `saturation.use_capillary`
  and `saturation.use_gravity` are both true
- mismatched `enabled` / `use_*` flags are rejected during config validation
- the combined pipeline outputs capillary flux, gravity flux, and
  `combined_report.json`

## 9. Future Implementation Plan

### 029_combined_flux_composer

`029_combined_flux_composer` is implemented as an independent water flux composer
for `Fw_adv`, `Fw_cap`, and `Fw_grav`. It does not modify the saturation solver
and does not advance `Sw`. Its role is to compose total water face fluxes, build
a conservative effective flux for CFL checks, and return a combined flux report.

### 030_combined_capillary_gravity_transport_3d

`030_combined_capillary_gravity_transport_3d` is implemented at the function
level. It uses `water_flux_composer` to combine `Fw_adv`, `Fw_cap`, and
`Fw_grav`, uses the conservative effective flux for CFL checks, and computes
material balance from the combined water flux. CLI / YAML integration is
available through `config/combined_case.yaml`.

### 031_combined_pipeline_case

`031_combined_pipeline_case` is implemented. The full pipeline dispatches to
`advance_saturation_3d_with_capillary_and_gravity(...)` when both physical terms
are enabled, saves `capillary_pressure.npy`, capillary face fluxes, gravity face
fluxes, and `combined_report.json`, and records combined transport fields in
`case_summary.json`. Demo, capillary-only, and gravity-only cases retain their
single-path behavior.

Planned follow-up stages:

### 032_combined_profiling_and_validation

`032_combined_profiling_and_validation` is implemented as an external harness
layer. It does not modify the solver kernels.

Combined profiling runs:

- `config/demo_case.yaml`
- `config/capillary_case.yaml`
- `config/gravity_case.yaml`
- `config/combined_case.yaml`

The profiling report records runtime, cell count, time-step count, capillary /
gravity / combined flags, CFL, material balance, max capillary flux, max gravity
flux, max total water flux, max effective flux, and success. The report files
are:

- `profiling_reports/combined_performance_summary.json`
- `profiling_reports/combined_performance_summary.md`

Combined validation runs `config/combined_case.yaml` and checks required output
files, `Sw` bounds, nonnegative `Pc`, finite `Pc` / flux / `Sw`, nonzero
capillary flux, nonzero internal gravity z flux, reasonable material balance,
`combined_transport_enabled=true`, and `success=true`. The report files are:

- `validation_reports/combined_validation_summary.json`
- `validation_reports/combined_validation_summary.md`

DT sensitivity uses the same combined case with:

- `dt = base_dt`
- `dt = base_dt / 2`
- `dt = base_dt / 4`

Each record includes `dt`, `max_cfl`, `material_balance_error`,
`sw_simulated_min`, `sw_simulated_max`, runtime, and success. Reducing `dt`
must not introduce NaN / Inf values, saturation-bound violations, or a clear
material-balance regression.

Explicit-format risk remains:

- strong capillary pressure may impose a strict diffusion-like time-step limit
- fine grids reduce pore volume and tighten CFL
- strong density contrast can increase segregation flux

C++ migration should not start from the combined path alone unless profiling
shows a concrete bottleneck, such as pressure assembly / solve, face flux
composition, or saturation update dominating runtime at engineering-scale case
sizes. Python remains the source of truth for configuration, IO, tests, and
reports.

Semi-implicit capillary diffusion remains a future option. It should be
triggered only if validation/profiling shows explicit combined transport needs
impractically small `dt`, produces sensitivity that does not improve with
smaller `dt`, or larger capillary-gradient cases become unstable despite CFL
checks.

## 11. Release Candidate Documentation

Release Candidate v2 documentation is organized under `docs/` and README:

- architecture and module boundaries
- numerical methods and sign conventions
- YAML case configuration
- CLI usage
- validation and profiling reproduction
- limitations and roadmap
- module capability matrix
- release and regression checklist

This stage is documentation-only. It does not modify pressure, velocity,
relative permeability, CFL, saturation, capillary, gravity, inversion, fusion,
UDP, or C++ implementation files.

## 10. Test Plan

Future combined solver tests should cover:

- disabled matches original
- capillary only matches `advance_saturation_3d_with_capillary`
- gravity only matches `advance_saturation_3d_with_gravity`
- capillary + gravity changes solution
- report keys
- `Sw` bounds
- no NaN / Inf
- CFL violation
- material balance
- repeatability
- capillary and gravity flux signs
- capillary gradient + gravity density contrast case
