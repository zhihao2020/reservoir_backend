# ADR 002: Notion as Project Dashboard

## Background

The project needs a visible planning surface for requirements, modules,
algorithms, validation, and tasks.

## Decision

Use Notion as a project dashboard and coordination board. Do not use it as the
engineering source of truth.

## Reasons

- Notion is convenient for status review and planning.
- Git remains better for code-linked evidence and version history.
- Conservative status gates reduce risk of overstating delivery readiness.

## Alternatives

- Keep all planning only in Markdown.
- Use GitHub Projects only.
- Use spreadsheets only.

## Impact

Notion pages and databases must mirror repository documents, especially
`STATUS.md` for module status and the active documents under `docs/`.

## Risks

Manual sync can drift. Status inflation is possible if Notion is updated
without tests and benchmark evidence.

## Revision Conditions

Revisit if Notion sync becomes automated through CI or if the project moves to
a different project management system.
