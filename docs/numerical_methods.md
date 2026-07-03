# Numerical Methods

## Grid and Unknown Placement

The prototype uses a structured Cartesian grid. Cell-centered variables include
pressure, saturation, porosity, permeability fields, capillary pressure, and
fused fields. Face flux arrays use finite-volume shapes:

- `flux_x`: `(nz, ny, nx + 1)`
- `flux_y`: `(nz, ny + 1, nx)`
- `flux_z`: `(nz + 1, ny, nx)`

## Discretization

The discretization is a cell-centered finite-volume method. Pressure uses
TPFA-style transmissibility with harmonic average permeability. Flux balance is
assembled over cell faces. Saturation transport uses upwind fractional flow for
advective water flux.

## Pressure Solver

The pressure solver is a steady-state 3D pressure prototype for Cartesian grids.
It supports Dirichlet boundaries, no-flow boundaries, source/sink style wells,
and sparse matrix solve. The finite-volume form balances transmissibility
weighted pressure differences and source terms.

## Saturation Solver

Saturation uses an explicit finite-volume update:

```text
Sw_new = Sw_old - dt / (phi * V) * net_water_flux_out
```

The update is water-flux based, not cell-velocity based. CFL checks use face
fluxes and pore volume. Reports include max CFL, water cut, storage change,
injected/produced water volume, and material-balance error.

## Capillary Pressure

The convention is:

```text
Pc = Po - Pw
```

Implemented independent models:

- no-capillary model
- Brooks-Corey
- van Genuchten

Capillary flux uses:

```text
qcap_x = T_abs * Mcap * (Pc_right - Pc_left)
qcap_y = T_abs * Mcap * (Pc_back - Pc_front)
qcap_z = T_abs * Mcap * (Pc_top - Pc_bottom)
```

Positive x/y/z flux means left-to-right, front-to-back, and bottom-to-top.

## Gravity

Depth is positive down. The z face convention is `flux_z > 0` for bottom to
top. If `rho_w > rho_o`, water segregates downward, so internal gravity water
flux in z is negative.

## Combined Transport

Combined transport uses:

```text
Fw_total = Fw_adv + Fw_cap + Fw_grav
```

where `Fw_adv = fw_upwind * total_flux`. The conservative effective flux for
CFL is:

```text
abs(Fw_adv) + abs(Fw_cap) + abs(Fw_grav)
```

The combined path remains explicit. Strong capillary pressure, fine grids, or
strong gravity segregation can require smaller time steps. Future work may add
semi-implicit capillary diffusion if validation shows explicit time-step limits
become too restrictive.

## Known Limitations

The current model has no black-oil PVT, no three-phase flow, no fully implicit
Newton solve, no commercial-grade well model, no corner-point grid, no NNC, no
local grid refinement, and no geomechanics.
