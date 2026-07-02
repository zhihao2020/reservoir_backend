# Numerical Regression Fixtures

This directory contains fixed reference outputs for small, fast numerical tests.

The cases are inspired by common reservoir-simulator regression patterns:
small Cartesian grids, simple boundary conditions, explicit physical checks, and
frozen reference values. They do not depend on OPM, MRST, PorePy, or external
data at runtime.

References live in `references/` as `.npz` field arrays and `.json` metadata.
