# reservoir_backend

Lightweight Python prototype for reservoir backend calculations.

Implemented backend modules include:

- `Grid3D`, `Field3D`, `Well`, units, and project exceptions
- Archie resistivity saturation inversion
- transmissibility, 1D/2D/3D pressure solve, Darcy face flux and velocity
- Corey relative permeability, fractional flow, CFL checks
- standalone Brooks-Corey / van Genuchten / no-capillary Pc(Sw) models and capillary face fluxes
- standalone oil-water gravity segregation face fluxes
- 1D/3D oil-water saturation transport, optional 1D/3D capillary water-flux coupling, and optional vertical 1D / 3D gravity water-flux coupling
- field mapping, confidence weighting, field fusion
- result management and a full pipeline demo

## Install

```bash
pip install -e .
```

## Test

```bash
pytest -q
```

## Full Pipeline Demo

```bash
python examples/run_full_pipeline_demo.py
```

This writes a complete small case to `results/demo_case`.

## Config-Driven Case Runner

Run a YAML-configured case through the backend pipeline:

```bash
python -m reservoir_backend.cli.run_case --config config/demo_case.yaml
python scripts/run_case.py --config config/multisignal_case.yaml
python scripts/run_case.py --config config/capillary_case.yaml
python scripts/run_case.py --config config/capillary_gradient_case.yaml
python scripts/run_case.py --config config/gravity_case.yaml
python scripts/run_case.py --config config/combined_case.yaml
```

Useful options:

- `--output-dir PATH` overrides `case.output_dir`
- `--case-id NAME` overrides `case.case_id`
- `--mode archie_only|multisignal` overrides `case.mode`
- `--dry-run` validates and prints the normalized core parameters without writing results
- `--verbose` prints formatted JSON output

Case YAML files may include an optional standalone capillary pressure section:

```yaml
capillary_pressure:
  enabled: false
  model: none  # none, brooks_corey, or van_genuchten
  entry_pressure_pa: 1000.0
  lambda_pc: 2.0
  p0_pa: 1000.0
  m: 0.5
  n: 2.0
```

Case YAML files may also include an optional standalone gravity section:

```yaml
gravity:
  enabled: false
  g: 9.80665
  rho_w: 1000.0
  rho_o: 800.0
  depth_axis: z
  depth_positive: down
```

Nonuniform initial saturation can be configured for capillary validation:

```yaml
initial_saturation:
  type: step_x
  low_sw: 0.2
  high_sw: 0.75
  split_fraction: 0.5
```

Capillary pressure can be evaluated independently from Pc(Sw). Capillary face
fluxes can be computed from Sw, Pc(Sw), relative permeability mobility, and
absolute permeability. The saturation transport path has opt-in
`advance_saturation_1d_with_capillary` and `advance_saturation_3d_with_capillary`
entry points. `config/capillary_case.yaml` enables 3D capillary transport in
the full pipeline. `config/capillary_gradient_case.yaml` also enables a step-x
initial Sw field and verifies nonzero Pc gradients and capillary flux. These
cases write:

- `capillary_pressure.npy`
- `capillary_flux_x.npy`
- `capillary_flux_y.npy`
- `capillary_flux_z.npy`
- `capillary_report.json`
- `initial_saturation.npy` for nonuniform initial saturation cases

The default demo and multisignal cases keep capillary transport disabled.

Gravity fluxes are available as an independent solver module. The current
convention is `gravity_flux_z > 0` for bottom-to-top water flux; when water is
denser than oil, gravity gives negative internal z flux, meaning downward water
segregation. `advance_saturation_1d_vertical_with_gravity` and
`advance_saturation_3d_with_gravity` can add that gravity water flux to
saturation transport. Gravity is disabled in existing YAML cases;
`config/gravity_case.yaml` enables 3D gravity transport in the full pipeline and
writes:

- `gravity_flux_x.npy`
- `gravity_flux_y.npy`
- `gravity_flux_z.npy`
- `gravity_report.json`

Capillary and gravity transport can be enabled together through
`config/combined_case.yaml`. In that mode the pipeline calls
`advance_saturation_3d_with_capillary_and_gravity`, uses `water_flux_composer`
to combine `Fw_adv`, `Fw_cap`, and `Fw_grav`, and writes:

- `capillary_pressure.npy`
- `capillary_flux_x.npy`, `capillary_flux_y.npy`, `capillary_flux_z.npy`
- `gravity_flux_x.npy`, `gravity_flux_y.npy`, `gravity_flux_z.npy`
- `combined_report.json`

The config loader requires `capillary_pressure.enabled` to match
`saturation.use_capillary`, and `gravity.enabled` to match
`saturation.use_gravity`; mismatched flags are rejected instead of silently
dropping a physical term.

## Multisignal Inversion Demo

```bash
python examples/run_multisignal_inversion_demo.py
```

This writes resistivity, electromagnetic, acoustic, and confidence-weighted
signal-fused saturation fields to `results/multisignal_demo`.

## Validation

```bash
python harness/run_validation.py
```

This runs tests, executes the full pipeline demo, checks output files and
physical ranges, and writes:

- `validation_reports/validation_summary.json`
- `validation_reports/validation_summary.md`

## Profiling

```bash
python scripts/profile_full_pipeline.py
python scripts/profile_capillary_pipeline.py
```

This profiles small, medium, and large-lite Python cases and writes:

- `profiling_reports/performance_summary.json`
- `profiling_reports/performance_summary.md`
- `profiling_reports/capillary_performance_summary.json`
- `profiling_reports/capillary_performance_summary.md`

## C++ Migration

C++ is not implemented now. See `specs/09_cpp_migration_spec.md`; C++ work only
starts after validation passes and profiling identifies a concrete bottleneck.
