# Performance Baseline

## Purpose

TASK-019 establishes a Python / NumPy / SciPy performance baseline for the
current backend. It measures existing implementations and writes a repeatable
report; it does not optimize, rewrite, or migrate numerical kernels.

## Scope

The baseline covers small / medium / large synthetic cases for:

- pressure solve
- oil-water saturation transport
- parameter field fusion
- cross-scale report calculations
- benchmark registry aggregation
- report generation

## Synthetic Cases

| Case | Pressure/fusion grid shape `(nz, ny, nx)` | Saturation cells |
| --- | --- | ---: |
| small | `(3, 4, 8)` | 64 |
| medium | `(4, 8, 12)` | 192 |
| large | `(5, 10, 16)` | 384 |

These cases are intentionally modest. They are designed to provide a stable
engineering baseline inside the test suite, not a production capacity claim.

## Metrics

Each stage records:

- `runtime_sec`
- `memory_peak_mb`
- `array_size_bytes`
- `success`
- stage-specific diagnostics such as pressure residual norm, saturation CFL,
  material balance, fusion ranges, cross-scale similarity, or registry counts

The report also includes:

- runtime summary by stage
- memory summary by stage
- slowest stage
- numerical equivalence check
- report generation time
- `numba_recommended`
- `cpp_recommended`

## Numerical Equivalence

The performance baseline repeats deterministic pressure, saturation, and fusion
operations and compares checksum-style outputs. The current equivalence check
requires the maximum checksum difference to be no larger than `1.0e-12`.

## Current Recommendation

No C++ kernels are implemented in this stage.
No numba kernels are introduced in this stage.

For the current synthetic baseline, numba is not recommended because no clear
Python kernel bottleneck is observed. C++ is also not recommended because the
measured cases do not justify pybind11 or compiled-kernel maintenance cost.

Larger field-like profiling should be run before any migration decision.

## Reports

Generated reports:

- `accuracy_reports/performance_baseline_summary.json`
- `accuracy_reports/performance_baseline_summary.md`

Runner:

```bash
python -m reservoir_backend.performance.performance_report
```

## Limitations

- No C++ kernels are implemented.
- No pybind11 integration is implemented.
- No numba kernels are introduced.
- No numerical algorithm is changed.
- No solver, inversion, fusion, cross-scale, data, result, benchmark, reference,
  config, CLI, API, UDP, CMake, or pybind11 code is modified by the baseline.
- Synthetic cases are not production-scale capacity tests.
- The report does not claim commercial simulator equivalence.
