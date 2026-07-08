# Capillary / Gravity Validation

## Status

Capillary / gravity benchmark hardening: Done

- capillary pressure monotonicity benchmark: Done
- capillary no-gradient zero-flux benchmark: Done
- capillary smoothing benchmark: Done
- gravity zero-density-difference benchmark: Done
- gravity segregation direction benchmark: Done
- combined capillary + gravity stability benchmark: Done
- water flux composer consistency benchmark: Done
- OPM SPE1 capillary / gravity sanity adapted benchmark: Done

## Scope

This benchmark hardens the current optional oil-water capillary and gravity
paths:

- `capillary_pressure.py`
- `capillary_flux.py`
- `gravity_flux.py`
- `water_flux_composer.py`
- optional combined capillary + gravity saturation transport

The benchmark does not change solver behavior. It only calls the existing
modules and records diagnostics.

## Benchmark Cases

| Case | Purpose | Source |
| ---- | ------- | ------ |
| `capillary_pressure_monotonicity` | Checks Brooks-Corey Pc finite values and monotonic decrease with Sw. | Internal trend case |
| `capillary_no_gradient_zero_flux` | Checks uniform Sw gives approximately zero capillary flux. | Internal uniform field |
| `capillary_smoothing` | Checks a saturation step is smoothed by capillary transport with small stable dt. | Internal 1D step |
| `gravity_zero_density_difference` | Checks rho_w == rho_o gives approximately zero gravity flux. | Internal vertical grid |
| `gravity_segregation_direction` | Checks rho_w > rho_o gives negative internal z gravity flux under current convention. | Internal vertical grid |
| `combined_capillary_gravity_stability` | Checks combined transport is bounded and finite with nonzero capillary and gravity fluxes. | Internal 3D step |
| `water_flux_composer_consistency` | Checks advective, capillary, gravity, and combined switch behavior. | Internal composer arrays |
| `opm_spe1case1_capillary_gravity_sanity_adapted` | Loads SPE1 property metadata and checks bounded diagnostic setup. | OPM `SPE1CASE1.DATA` adapted metadata |

## Sign Convention

The project uses `Grid3D` arrays as `(nz, ny, nx)`. Positive `flux_z` is
bottom-to-top. With `depth_positive=down` and `rho_w > rho_o`, current gravity
flux convention gives negative internal `gravity_flux_z`, and the existing
gravity saturation path follows that convention.

## Explicit Transport Limitation

No semi-implicit capillary solver implemented.
No black-oil transport implemented.
No full SPE1 or SPE10 reproduction.
No OPM Flow equivalence.
No MRST integration.
No solver core rewrite.
No runtime dependency on OPM or MRST.

Strong capillary pressure, fine grids, or large density differences can make
explicit transport more restrictive. This benchmark uses small stable cases and
does not claim production-scale capillary/gravity validation.

## Outputs

Run:

```bash
python benchmarks/capillary_gravity_benchmark.py
```

The runner writes:

- `accuracy_reports/capillary_gravity_benchmark_summary.json`
- `accuracy_reports/capillary_gravity_benchmark_summary.md`

The summary contains capillary pressure trend, capillary and gravity flux
magnitudes, sign checks, smoothing metrics, combined boundedness, material
balance, NaN/Inf status, and adapted-reference policy metadata.
