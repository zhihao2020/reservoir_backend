# API and Data Contract

This document consolidates the lightweight CLI, case configuration,
experimental data, result manifest, and front-end field contract notes. Module
status is tracked only in [../STATUS.md](../STATUS.md).

## CLI Entry Points

The current CLI surface is intentionally small. The shortest case check is:

```bash
python scripts/run_case.py --config config/demo_case.yaml --dry-run
```

Other report runners are Python modules, for example:

```bash
python -m reservoir_backend.simulation.impes_report
python -m reservoir_backend.performance.performance_report
python -m reservoir_backend.project.case_report
```

These entry points are developer tools, not a product API.

## Case Configuration

Case files are YAML documents under `config/`. A typical case describes:

- grid dimensions and cell size;
- rock and fluid properties;
- initial pressure or saturation fields;
- boundary conditions;
- optional wells or source terms;
- output paths and report options.

Configuration is intentionally limited to the existing Python backend. It does
not describe black-oil PVT, industrial well controls, or front-end workflows.

## Experimental Data Contract

The data pipeline reads CSV, JSON, and NPZ inputs into a standard internal
dataset. Supported field families include:

- `resistivity`
- `electromagnetic_response`
- `acoustic_response`
- `pressure`
- `saturation`
- `porosity`
- `permeability`
- `temperature`
- `time`
- `x`, `y`, `z`
- `confidence`
- `variance`
- `metadata`
- `unit`
- `source_name`

Physical checks include:

- porosity in `[0, 1]`;
- saturation in `[0, 1]`;
- permeability greater than zero;
- pressure finite;
- resistivity greater than zero;
- confidence in `[0, 1]` when present;
- variance nonnegative when present.

Unit normalization currently covers pressure, permeability, fraction/percent,
time, coordinates, and temperature. The QC pipeline reports missing units,
NaN/Inf values, missing values, duplicate time or coordinates, bounds
violations, and outlier flags.

## Result Manifest Contract

Each exported result manifest entry contains:

- `result_id`
- `case_id`
- `run_id`
- `module`
- `result_type`
- `field_name`
- `shape`
- `dtype`
- `unit`
- `path`
- `format`
- `created_at`
- `source_task`
- `source_report`
- `metadata`
- `warnings`
- `limitations`

Large arrays should be exported as NPZ. CSV exports should contain metadata and
summary rows, not full 3D field dumps.

## Report Index

The report index can register JSON and Markdown report paths such as:

- `accuracy_reports/experimental_data_qc_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/benchmark_registry_summary.json`
- `accuracy_reports/result_manifest_summary.json`

Missing paths should be reported as warnings rather than fabricated.

## Front-End Field Contract

The repository includes a field contract for future front-end or reporting
consumers. It defines:

- pressure fields;
- saturation fields;
- fusion fields;
- benchmark report fields;
- QC report fields;
- warning and error fields;
- units and shape conventions.

This is only a data contract. There is no front-end implementation, UDP
implementation, REST API, or database service in this repository.

## Limitations

- No commercial data-management platform.
- No Petrel-like workflow.
- No solver rewrite through this interface layer.
- No black-oil PVT contract.
- No guarantee that archived historical docs match the current contract.
