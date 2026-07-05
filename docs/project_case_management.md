# Project / Case Management

## Purpose

TASK-056 adds a lightweight Project / Case management layer for organizing
backend evidence into projects, cases, runs, result manifests, and report
links. It is an engineering metadata layer. It does not run solvers and does
not replace the existing result manifest package.

## Project metadata

Project records contain:

- `project_id`
- `name`
- `description`
- `created_at`
- `metadata`

Projects group related cases and runs. The registry is in-memory and
JSON-serializable through `ProjectRegistry.to_dict()`.

## Case metadata

Case records contain:

- `case_id`
- `project_id`
- `case_name`
- `input_paths`
- `output_paths`
- `module_tags`
- `status`
- `metadata`

Supported status values are:

- `draft`
- `ready`
- `running`
- `completed`
- `failed`
- `validated`
- `archived`

The case registry supports add, list, find, status update, path validation, and
JSON serialization.

## Run history

Run records contain:

- `run_id`
- `case_id`
- `started_at`
- `finished_at`
- `status`
- `report_paths`
- `result_manifest_paths`
- `metrics`
- `warnings`

The run history is append-only in normal use and supports add, list, find,
report path validation, and JSON serialization.

## Report index alignment

The project management report uses the existing result report index contract
from `reservoir_backend.results.report_index`. It records whether expected
accuracy reports exist and exposes missing-path warnings without fabricating
files.

Generated reports:

- `accuracy_reports/project_case_management_summary.json`
- `accuracy_reports/project_case_management_summary.md`

Runner:

```bash
python -m reservoir_backend.project.case_report
```

## Example flow

1. Create a `ProjectMetadata` entry.
2. Register one or more `CaseMetadata` entries under the project.
3. Append `RunRecord` entries for completed or validated runs.
4. Validate input, output, report, and result manifest paths.
5. Write the summary report for delivery evidence.

## Limitations

- No database service.
- No frontend.
- No UDP or REST API.
- No Petrel-like full workflow.
- No complex permission system.
- No solver rewrite.
- No changes to inversion, fusion, cross-scale, data, result, benchmark,
  reference, config, CLI, API, C++, CMake, or pybind11 code.
