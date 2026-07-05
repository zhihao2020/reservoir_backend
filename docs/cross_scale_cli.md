# Cross-Scale Runner

## Overview

TASK-003 exposes the existing cross-scale library functions through a lightweight
runner. The runner reads a dict, JSON file, or YAML file, runs similarity
criteria, scale-effect analysis, and lab-field curve validation, then writes JSON
and Markdown reports.

## Configuration Schema

Required top-level sections:

- `case_id`
- `lab_case`
- `field_case`
- `curves`

`lab_case.descriptor` and `field_case.descriptor` use the
`ScaleDescriptor` fields:

- `length_scale_m`
- `time_scale_s`
- `pressure_scale_pa`
- `permeability_scale_m2`
- `porosity`
- `viscosity_pa_s`
- `density_kg_m3`
- `velocity_scale_m_s`
- `flow_rate_m3_s`
- optional `temperature_scale_k`
- optional `interfacial_tension_n_m`
- optional `diffusivity_m2_s`
- optional `delta_density_kg_m3`
- optional `gravity_m_s2`
- optional `pressure_drop_pa`
- optional `elapsed_time_s`
- optional `mobility_displacing`
- optional `mobility_displaced`

Each curve item contains `lab` and `field` curves with `name`, `time`, `values`,
and optional `unit`, `curve_type`, and `source`.

## JSON Example

See `tests/fixtures/cross_scale/valid_cross_scale_case.json`.

## YAML Example

See `tests/fixtures/cross_scale/valid_cross_scale_case.yaml`.

YAML support uses the existing `PyYAML` project dependency. Dict and JSON configs
remain supported for lightweight environments.

## Runner Flow

1. `load_config`
2. `run_similarity_report`
3. `run_scale_effect_report`
4. `run_lab_field_validation_report`
5. `run_cross_scale_benchmark`
6. `write_cross_scale_reports`

Run manually:

```bash
python -m reservoir_backend.cross_scale.runner
```

## Output Paths

Default outputs:

- `accuracy_reports/cross_scale_benchmark_summary.json`
- `accuracy_reports/cross_scale_benchmark_summary.md`

## Result Manifest Integration

The runner creates a result manifest entry when `reservoir_backend.results` is
available:

- `result_type`: `cross_scale_report`
- `module`: `M6`
- `source_task`: `TASK-003`
- `format`: `json`

TASK-017 adds a separate upscaling report entry point:

```bash
python -m reservoir_backend.cross_scale.upscaling_report
```

It writes `accuracy_reports/cross_scale_upscaling_summary.json` and `.md`.

## Limitations

No history matching.
No automatic calibration.
No complex upscaling solver.
No front-end.
No UDP.
No commercial simulator equivalence.
No validation of black-oil models.
