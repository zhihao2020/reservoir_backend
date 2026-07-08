# Roadmap

This document records limitations and development direction. Module status is
tracked only in [../STATUS.md](../STATUS.md).

## Current Limitations

The current backend is scoped to structured-grid Python workflows and small
validation cases. Known limits include:

- structured Cartesian grids are the main numerical target;
- explicit transport remains the primary transport path;
- simplified WOG utilities are incompressible and do not implement black-oil
  PVT;
- cross-scale modules provide reporting and diagnostics, not a multiscale
  finite-volume solver;
- fusion utilities do not perform history matching, automatic calibration, or
  ensemble data assimilation;
- result and project management are file-based and do not provide a database
  service;
- no front-end, UDP service, REST API, or C++ acceleration layer is part of the
  current backend.

## Near-Term Maintenance Priorities

Near-term work should focus on:

- keeping documentation entry points small and consistent;
- keeping `STATUS.md` as the only maintained status table;
- strengthening real experimental dataset coverage for the data pipeline;
- expanding pressure and transport regression cases only where they fit the
  existing structured-grid scope;
- improving report schemas when downstream consumers need stable fields;
- preserving benchmark reproducibility without adding runtime dependencies on
  external simulators.

## Future Scope

The following areas are outside the current MVP:

- black-oil and compositional PVT;
- broad industrial well controls and wellbore networks;
- fully implicit reservoir simulation;
- complex corner-point geological models;
- full SPE reproduction claims;
- OPM Flow or MRST equivalence;
- production front end, database service, UDP service, or REST API;
- C++ or pybind11 kernel migration without a measured bottleneck;
- history matching, automatic calibration, EnKF, ES-MDA, or Bayesian inversion
  workflows.

## Documentation Policy

Historical matrix and checklist documents have been moved to
`docs/archive/doc_consolidation/`. They are retained for traceability but are
not active status sources. New documentation should link to this roadmap and
to `STATUS.md` rather than creating another completion matrix.
