# API / Frontend Integration Roadmap

## Purpose

This document defines a future integration roadmap for REST API and frontend
consumers. It is documentation only. The repository does not implement a REST
service, frontend application, UDP service, authentication system, or deployment
gateway in this stage.

## REST API Roadmap

Future REST API work should expose file-backed project and result concepts
through stable endpoints. Proposed endpoint families:

- project endpoints;
- case endpoints;
- run endpoints;
- result manifest serving;
- report serving;
- health and version endpoints;
- authentication and authorization hooks.

## Project / Case / Run Endpoints

Suggested route shape:

```text
GET /projects
GET /projects/{project_id}
GET /projects/{project_id}/cases
GET /cases/{case_id}
GET /cases/{case_id}/runs
GET /runs/{run_id}
```

These routes should read from the existing project/case/run registry contracts.
They should not own numerical execution semantics.

## Result Manifest Serving

Future result endpoints should serve existing manifest entries:

```text
GET /results
GET /results/{result_id}
GET /results/{result_id}/metadata
GET /results/{result_id}/download
```

Large arrays should remain in NPZ or future visualization formats. Metadata and
summary tables can be returned as JSON.

## Report Serving

Report serving should expose JSON and Markdown artifacts under
`accuracy_reports/` and future case output directories:

```text
GET /reports
GET /reports/{report_id}
GET /reports/{report_id}.json
GET /reports/{report_id}.md
```

Missing report paths should return explicit missing-path errors rather than
fabricated summaries.

## Frontend Field Contract

The future frontend should consume the existing field contract:

- pressure fields;
- saturation fields;
- production curves;
- water cut curves;
- result manifests;
- QC reports;
- benchmark and engineering reports;
- warnings, limitations, and non-claims.

Shape conventions should follow structured-grid `(nz, ny, nx)` arrays and
time-series records with explicit `time` values.

## Auth / Security Placeholder

Future integration must define:

- authentication mode;
- authorization rules for project, case, run, and result access;
- file download restrictions;
- audit logging;
- input validation and upload limits;
- safe handling of generated reports.

No authentication or security service is implemented by this roadmap.

## UDP Deferred Rationale

UDP remains outside the current integration path because the industrial workflow
needs queryable project/case/run metadata, report serving, and result manifest
access. REST-style serving is a better future fit for those requirements than
command-style UDP messages.

## Limitations

- No REST service implementation.
- No frontend implementation.
- No UDP implementation.
- No auth/security implementation.
- No database service.
- No production deployment topology.

## Non-Claims

This document is not an API implementation claim and does not provide a frontend
or UDP runtime.
