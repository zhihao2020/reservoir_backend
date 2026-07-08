# Notion Sync Policy

Last updated: 2026-07-03

## Rules

1. Git is the engineering source of truth.
2. Notion is the project dashboard for progress, planning, and coordination.
3. Code status is determined by tests and benchmark/validation reports, not by optimistic labels.
4. Notion must not mark unverified functionality as `Deliverable`.
5. Every completed feature must synchronize:
   - GitHub Issue or task reference
   - test result
   - benchmark or validation report
   - Notion status
   - `requirements/delivery_matrix.md`

## Status Gate

| Status | Required evidence |
| --- | --- |
| Backlog | Requirement exists, no implementation started |
| Designing | Design/spec exists, code not complete |
| Coding | Code exists, tests or scope incomplete |
| Testing | Tests exist, benchmark/validation insufficient |
| Validated | Code, tests, and benchmark/validation report exist for declared MVP scope |
| Deliverable | Code, tests, docs, validation report, and delivery matrix closure exist |

## Prohibited Sync Behavior

- Do not copy API keys, passwords, database connection strings, server secrets, or personal private data into Notion.
- Do not mark demo-only functionality as `Deliverable`.
- Do not mark code-only functionality as `Validated`.
- Do not mark test-only numerical functionality as `Deliverable` without benchmark or validation reports.
- Do not hide known limitations in Notion summaries.

## Update Cadence

Update Git docs first, then Notion:

1. Commit or stage the relevant code and tests.
2. Run the relevant tests.
3. Regenerate benchmark/validation reports when numerical behavior changes.
4. Update `requirements/delivery_matrix.md`.
5. Update `docs/module_matrix.md`.
6. Sync Notion project overview and database rows.
