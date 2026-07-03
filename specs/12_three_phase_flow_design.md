# Three-Phase Flow Design

## 1. Scope

This design covers a future simplified water-oil-gas three-phase transport path
for structured Cartesian grids. It is design-only. No three-phase solver is
implemented in this stage.

The first version is not a black-oil model and does not include:

- `Bo`, `Bw`, or `Bg`
- `Rs` or `Rv`
- bubble point handling
- phase appearance / disappearance
- PVT table interpolation
- compressibility
- well controls
- fully implicit Newton coupling
- commercial simulator-level black-oil behavior

The first target is:

- incompressible three-phase transport
- fixed density
- fixed viscosity
- Corey-style three-phase relative permeability
- explicit finite-volume transport
- structured Cartesian grid

## 2. State Variables

Three phase saturations are:

- `Sw`
- `So`
- `Sg`

The closure relation is:

```text
Sw + So + Sg = 1
```

Recommended primary variables are `Sw` and `Sg`. Oil saturation is derived:

```text
So = 1 - Sw - Sg
```

Physical constraints:

- `Sw >= Swi`
- `Sg >= Sgc`
- `So >= Sor`
- `Sw + Sg <= 1 - Sor`
- all saturations must be clipped or validated to physical ranges

Invalid state must raise a clear exception. The solver must not silently
generate negative oil saturation.

## 3. Saturation Bounds

Parameters:

- `Swi`
- `Sor`
- `Sgc`
- `Sw_max = 1 - Sor - Sg`
- `Sg_max = 1 - Sor - Sw`

The admissible region is a saturation triangle constrained by residual water,
residual oil, and critical gas saturation. Any update that makes `So < Sor` or
`Sw + Sg > 1 - Sor` is invalid unless a documented clipping policy is applied
and reported.

## 4. Relative Permeability Design

The first implementation should support a Corey three-phase relative
permeability model.

Water:

```text
krw = krw0 * Sew^nw
```

Gas:

```text
krg = krg0 * Seg^ng
```

Oil:

```text
kro = kro0 * Seo^no
```

Effective saturations must be defined under the three-phase constraints. A
simple first-pass design can normalize water and gas against their residual
values and derive oil mobility from the remaining oil saturation above `Sor`.

Complex Stone I, Stone II, and Baker models are not implemented in the first
version. They remain future work.

## 5. Mobility and Fractional Flow

Mobilities:

```text
lambda_w = krw / mu_w
lambda_o = kro / mu_o
lambda_g = krg / mu_g
lambda_t = lambda_w + lambda_o + lambda_g
```

Fractional flow:

```text
fw = lambda_w / lambda_t
fo = lambda_o / lambda_t
fg = lambda_g / lambda_t
```

The implementation must ensure:

```text
fw + fo + fg = 1
```

Endpoint cases must not produce NaN or Inf. If total mobility is zero, the
solver must use a clear policy such as raising an error or returning zero phase
fluxes with a report field.

## 6. Flux Design

Three-phase advective phase fluxes:

```text
Fw = fw_upwind * qt
Fo = fo_upwind * qt
Fg = fg_upwind * qt
```

where `qt` is total Darcy flux. Upwind selection follows the sign of `qt` on
each face.

The first three-phase transport path only considers advective flux. It does not
immediately merge capillary or gravity terms.

Future extensions may add:

- water-oil capillary pressure
- gas-oil capillary pressure
- gravity segregation
- combined three-phase transport

## 7. Transport Update

The first explicit update advances two independent variables:

```text
Sw_new = Sw_old - dt / (phi * V) * div(Fw)
Sg_new = Sg_old - dt / (phi * V) * div(Fg)
So_new = 1 - Sw_new - Sg_new
```

Post-update checks:

- `Sw >= Swi`
- `Sg >= Sgc`
- `So >= Sor`
- `Sw + Sg <= 1 - Sor`
- no NaN / Inf

The implementation must not allow `So < Sor` and must not allow
`Sw + Sg > 1 - Sor`. Material balance should be computed from phase fluxes and
storage changes.

## 8. CFL Strategy

The first design should use:

```text
effective_flux = abs(qt)
```

Reason: it is conservative for total advective transport, stays compatible with
the existing two-phase total-flux CFL logic, and avoids underestimating the time
step when water and gas phase fluxes move through the same face.

An alternative future diagnostic can also record:

```text
abs(Fw) + abs(Fg)
```

but the first implementation should check explicit stability against `abs(qt)`.

## 9. Material Balance

The three-phase material balance report should include:

- `water_injected_volume`
- `water_produced_volume`
- `water_storage_change`
- `water_balance_error`
- `gas_injected_volume`
- `gas_produced_volume`
- `gas_storage_change`
- `gas_balance_error`
- `oil_produced_volume`
- `oil_storage_change`
- `oil_balance_error`

The first implementation can primarily verify water and gas storage from their
explicit updates. Oil can be checked through closure:

```text
So = 1 - Sw - Sg
```

## 10. YAML Design

Future configuration:

```yaml
three_phase:
  enabled: true
  model: incompressible_wog
  primary_variables: [Sw, Sg]

fluid:
  mu_w: 1.0e-3
  mu_o: 5.0e-3
  mu_g: 1.0e-5
  rho_w: 1000.0
  rho_o: 800.0
  rho_g: 100.0

relperm_three_phase:
  swi: 0.2
  sor: 0.2
  sgc: 0.05
  krw0: 0.3
  kro0: 0.8
  krg0: 0.6
  nw: 2.0
  no: 2.0
  ng: 2.0

initial_saturation:
  sw: 0.2
  sg: 0.05
```

This YAML is a future design. The current executable pipeline remains oil-water
two-phase.

## 11. Future Implementation Plan

Suggested stages:

1. `035_three_phase_relperm`
2. `036_three_phase_fractional_flow`
3. `037_three_phase_transport_1d`
4. `038_three_phase_transport_3d`
5. `039_three_phase_pipeline_case`
6. `040_three_phase_validation_and_profiling`

## 12. Test Plan

Future three-phase flow tests should include:

1. saturation closure `Sw + So + Sg = 1`
2. bounds check
3. Corey `krw` / `kro` / `krg` endpoint behavior
4. finite mobility
5. `fw + fo + fg = 1`
6. 1D water-gas displacement
7. 3D transport shape
8. no NaN / Inf
9. CFL violation
10. material balance
11. repeatability
12. invalid saturation raises
13. invalid viscosity raises
14. three-phase disabled keeps two-phase behavior unchanged
