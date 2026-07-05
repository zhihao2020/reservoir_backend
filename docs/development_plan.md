# Development Plan

This plan is intentionally conservative. It does not mark any stage as
delivery-ready until code, tests, documentation, and validation evidence exist.

## M0: Project Skeleton and Documentation Standards

Goal: keep Git as the engineering source of truth and make status auditable.

Tasks:

- Maintain `docs/project_overview.md`, `docs/module_matrix.md`, and `requirements/delivery_matrix.md`.
- Keep `docs/notion_sync_policy.md` aligned with Notion dashboard usage.
- Require each numerical feature to link code, tests, benchmark report, and known limitations.

Acceptance criteria:

- Every requirement has a status and file references.
- No untested or unvalidated feature is marked `Deliverable`.

## M1: Pressure Field Solver Module

Goal: harden Cartesian pressure reconstruction.

Tasks:

- Extend heterogeneous and source/sink benchmark coverage.
- Add larger-grid profiling cases.
- Keep OPM/MRST adapted references clearly documented as adapted metadata, not equivalence claims.

Acceptance criteria:

- Pressure benchmark report includes analytical/manufactured/reference-adapted cases.
- Mass-balance and flux-conservation metrics remain within documented tolerance.

## M2: Saturation Inversion Module

Goal: make inversion status precise by signal type.

Tasks:

- Preserve Archie analytical validation.
- Add real or representative calibration datasets for EM and acoustic paths.
- Add report fields that separate empirical calibration quality from physical inversion claims.

Acceptance criteria:

- Archie, EM, acoustic, and fusion paths have tests and benchmark evidence.
- EM/acoustic documentation explicitly states empirical scope.

## M3: Saturation Field Calculation Module

Goal: harden explicit oil-water and simplified WOG transport.

Tasks:

- Extend dt/grid sensitivity for capillary, gravity, and combined cases.
- Track front movement, bounds, CFL, and material balance in one report.
- Consider semi-implicit capillary only if explicit stability becomes limiting.

Acceptance criteria:

- Saturation benchmark report covers bounds, CFL, material balance, and qualitative front movement.
- Three-phase reports preserve closure `Sw + So + Sg = 1`.

## M4: Parameter Field Fusion Module

Goal: move parameter fusion from `Testing` toward `Validated`.

Tasks:

- Implement `049_parameter_fusion_benchmark_hardening`.
- Benchmark same-grid fusion, IDW mapping, confidence weighting, NaN handling, and saturation bounds.
- Add fusion report summaries into validation output.

Acceptance criteria:

- Dedicated fusion benchmark summary exists under `accuracy_reports`.
- Tests verify fused-field shape, bounds, confidence behavior, and missing-data handling.

## M5: Cross-Scale Analysis Module

Goal: move independent cross-scale utilities toward integrated validation.

Tasks:

- Implement `050_cross_scale_benchmark_hardening`.
- Add CLI/YAML entry points for similarity and lab-field curve comparison.
- Create report templates for geometric, dynamic, material, and scale-effect analysis.

Acceptance criteria:

- Cross-scale benchmark summary exists.
- CLI/YAML can generate similarity and curve mismatch reports without solver modification.

## M6: UDP Frontend/Backend Communication

Goal: convert the minimal UDP Archie prototype into a versioned backend protocol.

Tasks:

- Add `protocol_version`, `request_id`, structured `payload`, and stable error codes.
- Add commands for `run_case`, `get_status`, and `get_result_summary`.
- Keep large arrays out of UDP datagrams; return file/report paths instead.
- Add frontend-facing protocol tests and timeout/retry documentation.

Acceptance criteria:

- UDP protocol roundtrips are regression-tested.
- `docs/udp_protocol.md` contains implemented request/response examples.

## M7: Validation Reports and Delivery Materials

Goal: produce acceptance-oriented evidence.

Tasks:

- Generate final numerical accuracy acceptance report.
- Link every requirement to code, tests, benchmark, and known gaps.
- Add result catalog and export plan for VTK/CSV/HDF5/slices/statistics.
- Keep Notion status synchronized with Git documents.

Acceptance criteria:

- `requirements/delivery_matrix.md` has no missing code/test/report references for deliverable items.
- Notion dashboard and Git docs agree on status.
