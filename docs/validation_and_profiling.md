# Validation and Profiling

## Full Validation

```bash
pytest -q
python harness/run_validation.py
```

The full validation harness runs tests, executes the full demo pipeline, checks
required outputs, checks physical ranges, and writes validation reports.

## Combined Validation

```bash
python scripts/validate_combined_pipeline.py
```

The combined validation checks:

- required combined outputs
- `Sw` bounds
- `Pc >= 0`
- finite `Pc`, flux, and `Sw`
- nonzero capillary flux
- nonzero internal gravity z flux
- material balance
- `combined_transport_enabled=true`
- `success=true`

## Profiling

```bash
python scripts/profile_full_pipeline.py
python scripts/profile_capillary_pipeline.py
python scripts/profile_combined_pipeline.py
```

Combined profiling compares:

- `config/demo_case.yaml`
- `config/capillary_case.yaml`
- `config/gravity_case.yaml`
- `config/combined_case.yaml`

## Interpreting Validation Summary

`validation_reports/combined_validation_summary.json` records output checks,
finite-value checks, saturation bounds, material balance, CFL, nonzero physics
flux checks, and dt sensitivity records.

## Interpreting Profiling Summary

`profiling_reports/combined_performance_summary.json` records per-case runtime,
cell count, step count, enabled physics flags, max CFL, material balance, max
capillary flux, max gravity flux, max total water flux, max effective flux, and
success.

## DT Sensitivity

The combined validation runs:

- `dt = base_dt`
- `dt = base_dt / 2`
- `dt = base_dt / 4`

Reducing `dt` should not introduce NaN / Inf values, saturation-bound
violations, or material-balance degradation.

## Current Release-Candidate Results

- `pytest -q`: 585 passed
- combined validation success=true
- combined_case runtime approximately 0.07 s
- combined/demo runtime ratio approximately 1.23x
- base max_cfl approximately 0.163
- material_balance_error = 0.0
- current recommendation: no C++ yet
- current recommendation: no semi-implicit capillary diffusion yet

## C++ Decision Rule

C++ should start only after larger-scale profiling identifies a concrete
bottleneck. Candidate triggers include pressure assembly / solve, flux
composition, or saturation update dominating runtime for engineering-sized
cases. Python remains responsible for configuration, IO, reporting, and tests.

## Semi-Implicit Capillary Diffusion Trigger Rule

Semi-implicit capillary diffusion should be considered only if explicit
combined transport requires impractically small time steps, dt sensitivity does
not improve with smaller dt, or strong capillary-gradient cases become unstable
despite CFL checks.
