# C++ Migration Specification

## Current Decision

Do not implement C++ now. The current backend remains a Python prototype until validation and profiling identify a specific performance bottleneck.

## Migration Principles

1. C++ starts only after the Python full pipeline passes and profiling is complete.
2. C++ should migrate only performance bottleneck modules.
3. C++ does not replace the Python backend; it acts as an optional compute kernel.
4. Python keeps configuration, IO, tests, result management, and service/interface orchestration.
5. pybind11 is the recommended binding layer.
6. C++ implementations must reuse the existing pytest suite through the same Python API.
7. C++ must not change existing Python API behavior.

## C++ Migration Priority

1. Pressure matrix assembly.
2. Transmissibility / face flux.
3. 3D saturation update.
4. CFL calculation.
5. Material balance.

## Not Planned For C++ Migration

1. Archie inversion.
2. Field fusion.
3. Config loader.
4. Result manager.
5. UDP / API.
6. README / docs.

## Start Conditions

Only start C++ work when all of the following are true:

1. `pytest -q` passes.
2. `examples/run_full_pipeline_demo.py` runs reliably.
3. `validation_reports/validation_summary.json` has `success=true`.
4. Profiling report identifies a concrete bottleneck.
5. Python API is stable enough to bind against.
6. Golden tests or `allclose` numerical comparison tests already exist.
7. The target C++ module has clear inputs and outputs.

## Suggested Profiling Thresholds

Consider local C++ migration if the medium or large-lite case shows:

- Saturation update is more than 40% of total runtime, or
- Pressure assembly / solve is more than 40% of total runtime, or
- Face flux calculation is more than 20% of total runtime, or
- Total runtime exceeds the engineering threshold for interactive/backend use.

Until those conditions are met, continue improving the Python implementation and tests.
