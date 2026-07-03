# Case Configuration

## Available YAML Cases

| Case | Purpose | Enabled Physics | Initial Sw | Typical Command |
| --- | --- | --- | --- | --- |
| `demo_case.yaml` | Base Archie-to-simulation pipeline | pressure, flux, explicit saturation, fusion | uniform | `python scripts/run_case.py --config config/demo_case.yaml` |
| `multisignal_case.yaml` | Resistivity / EM / acoustic inversion fusion | multisignal inversion and field fusion | uniform | `python scripts/run_case.py --config config/multisignal_case.yaml` |
| `capillary_case.yaml` | Capillary pipeline smoke case | capillary pressure and capillary transport | uniform | `python scripts/run_case.py --config config/capillary_case.yaml` |
| `capillary_gradient_case.yaml` | Nonuniform capillary-gradient validation | capillary pressure and capillary transport | nonuniform `step_x` | `python scripts/run_case.py --config config/capillary_gradient_case.yaml` |
| `gravity_case.yaml` | Gravity segregation pipeline case | gravity transport | uniform mobile Sw | `python scripts/run_case.py --config config/gravity_case.yaml` |
| `combined_case.yaml` | Combined capillary + gravity case | capillary, gravity, combined transport | nonuniform `step_x` | `python scripts/run_case.py --config config/combined_case.yaml` |

## Key Configuration Sections

### `case`

Defines `case_id`, `output_dir`, and `mode`. `mode` is `archie_only` or
`multisignal`.

### `grid`

Defines `nx`, `ny`, `nz`, `dx`, `dy`, and `dz` for the structured Cartesian
grid.

### `rock`

Defines porosity and permeability. Permeability is configured in mD and
normalized to SI units by the loader.

### `fluid`

Defines water and oil viscosities.

### `archie`

Defines Archie constants, water resistivity, and residual saturation bounds.

### `electromagnetic`

Controls empirical EM inversion. It supports `enabled`, `model_type`,
`coefficients`, and `calibration_range`.

### `acoustic`

Controls empirical acoustic inversion. It supports `enabled`, `model_type`,
`coefficients`, and `calibration_range`.

### `pressure`

Defines boundary style and pressure values. MPa values are converted to Pa.

### `saturation`

Defines `dt`, `steps`, `max_cfl`, residual saturations, Corey endpoints and
exponents, injected saturation, and `use_capillary` / `use_gravity` switches.

### `capillary_pressure`

Defines optional capillary pressure. Supported models are `none`,
`brooks_corey`, and `van_genuchten`.

### `gravity`

Defines optional gravity segregation with `g`, `rho_w`, `rho_o`,
`depth_axis`, and `depth_positive`.

### `initial_saturation`

Defines uniform or nonuniform initial Sw. Supported types include `constant`,
`step_x`, `linear_x`, and `center_blob`.

### `fusion`

Defines signal weights and dynamic fusion alpha.

### `outputs`

Controls optional output families such as capillary pressure, capillary flux,
gravity flux, reports, and combined report.

## Expected Outputs by Physics

Capillary cases write `capillary_pressure.npy`, `capillary_flux_x/y/z.npy`, and
`capillary_report.json`. Gravity cases write `gravity_flux_x/y/z.npy` and
`gravity_report.json`. The combined case writes both families plus
`combined_report.json`.
