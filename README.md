# Reservoir Backend

## Project Positioning

Reservoir Backend is a Python backend for reservoir experiment data handling,
pressure-field reconstruction, saturation inversion, and small structured-grid
numerical verification.

The project is intended for:

- repeatable handling of lab or synthetic reservoir data;
- finite-volume pressure and transport experiments on structured grids;
- validation reports for pressure, saturation, fusion, cross-scale, and data
  pipeline utilities;
- future maintenance by developers and Codex agents.

This repository is not a commercial reservoir simulator, not a black-oil
simulator, and not a product front end. Current status is maintained only in
[STATUS.md](STATUS.md).

## Quick Start

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

Run the test suite:

```bash
pytest -q
```

Run a minimal case through the existing script entrypoint:

```bash
python scripts/run_case.py --config config/demo_case.yaml --dry-run
```

Read experimental data through the internal data pipeline:

```python
from reservoir_backend.data.reader import read_experimental_data
from reservoir_backend.data.qc import run_qc_pipeline

dataset = read_experimental_data("tests/fixtures/experimental_data/valid_csv_core_fields.csv")
qc_report = run_qc_pipeline(dataset)
print(qc_report["success"])
```

Generate an existing report runner when needed:

```bash
python -m reservoir_backend.simulation.impes_report
python -m reservoir_backend.fusion.synthetic_twin_report
```

Detailed CLI, configuration, data, and result contracts are consolidated in
[docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md).

## Current Capabilities

The current codebase contains tested support for:

- structured grid and field data structures;
- units, wells, source terms, and lightweight boundary contribution utilities;
- CSV, JSON, and NPZ experimental data ingestion with QC reports;
- field-data ingestion for well tables, production history, pressure history,
  schedule CSV, and property fields;
- multi-well schedule v0 metadata, rate/BHP control interfaces, and report
  steps;
- Archie-style and multi-signal saturation inversion utilities;
- finite-volume pressure reconstruction on structured Cartesian grids;
- Darcy velocity and face-flux calculations;
- oil-water saturation transport with CFL diagnostics and optional TVD/MUSCL
  helper paths;
- capillary, gravity, and combined water-flux diagnostics;
- combined capillary + gravity transport checks;
- simplified incompressible WOG utilities for three-phase relperm, phase flux,
  transport checks, and reports;
- simplified sequential pressure-saturation coupling for small waterflood
  examples, including an IMPES loop report;
- parameter fusion, uncertainty diagnostics, and synthetic-twin field summaries;
- synthetic-only history matching prototype for known-truth generated cases;
- cross-scale similarity, scale-effect, and curve-comparison reporting;
- result manifests, project/case/run registries, and report index utilities;
- industrial case workflow v0 for config to Project/Case/Run to IMPES to
  engineering report;
- benchmark registry and performance baseline reports.

Capability maturity and scope boundaries are listed only in
[STATUS.md](STATUS.md).

## Validation Summary

Primary validation command:

```bash
pytest -q
```

This documentation cleanup did not rerun the full test suite. Existing report
artifacts are kept under:

- `accuracy_reports/`
- `validation_reports/` if present in a local checkout

Validation organization, benchmark report locations, and rerun guidance are in
[docs/VALIDATION.md](docs/VALIDATION.md).

Useful runner and report anchors:

- `python benchmarks/three_phase_benchmark.py`: Three-phase WOG benchmark hardening.
- `python -m reservoir_backend.cross_scale.runner`: writes `cross_scale_benchmark_summary`.
- `python -m reservoir_backend.cross_scale.upscaling_report`: writes `cross_scale_upscaling_summary`.
- `python -m reservoir_backend.performance.performance_report`: writes `performance_baseline_summary`.
- project_case_management_summary and frontend field contract are indexed by the
  result manifest documentation.
- Function hardening first remains the benchmark validation principle.
- cross-scale analysis design describes one backend with two first-level modules;
  cross-scale implementation is not yet complete.
- pressure solver benchmark, pressure solver enhancement, saturation transport benchmark,
  saturation transport enhancement, capillary / gravity benchmark, parameter fusion benchmark,
  parameter fusion uncertainty, benchmark registry, project / case management, synthetic twin,
  saturation inversion benchmark, lab-field validation, curve-to-curve comparison,
  similarity criteria, scale-effect analysis, field data ingestion.

## Documentation Index

- [STATUS.md](STATUS.md): the only module status source.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): repository structure, data flow,
  and numerical workflow overview.
- CLI Usage and case configuration are summarized in
  [docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md).
- [docs/VALIDATION.md](docs/VALIDATION.md): tests, benchmark reports, and
  verification workflow.
- [docs/ROADMAP.md](docs/ROADMAP.md): current limitations, near-term work, and
  future scope.
- [docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md): CLI, case
  configuration, experimental data schema, result manifest, and front-end field
  contract.
- [docs/archive/](docs/archive/): historical documentation snapshots retained
  for traceability, not active status sources.
- Fixture samples live under `tests/fixtures/experimental_data`.
