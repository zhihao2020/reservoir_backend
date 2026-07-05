# ADR 001: Git as Engineering Source

## Background

The project has code, tests, specs, benchmark scripts, and generated validation
reports in the repository. Project dashboards can drift from implementation
reality.

## Decision

Git is the engineering source of truth for implementation status, requirements,
tests, benchmark reports, and delivery evidence.

## Reasons

- Code, tests, and reports are versioned together.
- Reviewers can audit status claims against file paths.
- Notion pages can be regenerated or corrected from repository documents.

## Alternatives

- Use Notion as the authoritative source.
- Use ad hoc verbal status updates.

## Impact

Every status claim should point to repository files. Notion is a dashboard, not
the final engineering record.

## Risks

Repository documents may still become stale if not updated after code changes.

## Revision Conditions

Revisit if a formal requirements management system becomes mandatory and is
integrated with Git commits, CI, and test artifacts.
