# Fully implicit: upstream concept → product names

Licensed adaptation from OPM / GEOS algorithm families is allowed.
**Product identifiers must not match upstream names.** This table is the
rename contract for FIM work under `reservoir_backend/solver/`.

| Upstream concept (docs only) | Product symbol |
|------------------------------|----------------|
| primary-variable switch (Sg ↔ Rs) | `switch_live_oil_unknown` |
| Appleyard saturation chop | `clip_saturation_increment` |
| CNV cell check | `cell_cnv_ok` |
| material-balance / MBE check | `global_mass_balance_ok` |
| Newton update scaling (GEOS-style) | `scale_newton_update` |
| adaptive Δt from Newton count | `dt_from_newton_iters` |
| quasi-IMPES pressure weights | `pressure_row_weights` (optional, later) |
| FI black-oil step | `solve_fi_step` |
| product flag | `PhysicsSpec.fully_implicit` |

Forbidden examples in product code (non-exhaustive):

- `SimulatorFullyImplicit`, `FIBlackoilModel`, `NonlinearSystemBlackOilReservoir`
- `CompositionalMultiphaseFVM`, `AccumulationKernel`, `SolutionScalingKernel`
- `AppleyardChop`, `adaptPrimaryVariables` (as public names)

Comments may say “adapted from an industrial FIM family” and point here.
Do not claim OPM Flow or GEOS API compatibility.

## Grep guard (review)

Before merge, product tree should not introduce these identifiers:

```text
SimulatorFullyImplicit|FIBlackoilModel|CompositionalMultiphaseFVM|AppleyardChop
```
