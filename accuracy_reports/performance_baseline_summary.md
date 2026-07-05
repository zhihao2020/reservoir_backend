# Performance Baseline Summary

- success: True
- implementation: Python / NumPy / SciPy
- slowest_stage: saturation_transport
- slowest_stage_runtime_sec: 0.097747
- numba_recommended: False
- cpp_recommended: False
- numerical_equivalence_max_abs_error: 0.000000e+00

## Runtime Summary

| Stage | Runtime sec | Peak memory MB |
| --- | ---: | ---: |
| benchmark_registry | 0.089850 | 0.246010 |
| cross_scale | 0.008593 | 0.012409 |
| fusion | 0.002304 | 0.098600 |
| pressure | 0.149596 | 0.543477 |
| saturation_transport | 0.169261 | 0.044751 |

## Synthetic Cases

### small

- total_cells: 96
- total_runtime_sec: 0.072866
- total_memory_peak_mb: 0.246010
- pressure: runtime=0.015844s, memory=0.075054MB, success=True
- saturation_transport: runtime=0.017996s, memory=0.022374MB, success=True
- fusion: runtime=0.000777s, memory=0.016224MB, success=True
- cross_scale: runtime=0.003262s, memory=0.012409MB, success=True
- benchmark_registry: runtime=0.034987s, memory=0.246010MB, success=True

### medium

- total_cells: 384
- total_runtime_sec: 0.133587
- total_memory_peak_mb: 0.243530
- pressure: runtime=0.042073s, memory=0.230634MB, success=True
- saturation_transport: runtime=0.053517s, memory=0.026299MB, success=True
- fusion: runtime=0.000898s, memory=0.049593MB, success=True
- cross_scale: runtime=0.002971s, memory=0.009710MB, success=True
- benchmark_registry: runtime=0.034128s, memory=0.243530MB, success=True

### large

- total_cells: 800
- total_runtime_sec: 0.213150
- total_memory_peak_mb: 0.543477
- pressure: runtime=0.091679s, memory=0.543477MB, success=True
- saturation_transport: runtime=0.097747s, memory=0.044751MB, success=True
- fusion: runtime=0.000628s, memory=0.098600MB, success=True
- cross_scale: runtime=0.002360s, memory=0.009587MB, success=True
- benchmark_registry: runtime=0.020736s, memory=0.243507MB, success=True

## Recommendations

- numba: not recommended for the current synthetic baseline; no clear Python kernel bottleneck was observed
- C++: not recommended for the current synthetic baseline; keep C++ deferred until larger profiling proves need

## Limitations

- Synthetic small/medium/large cases are performance baselines, not production-scale capacity tests.
- The report does not implement C++, pybind11, numba kernels, or numerical algorithm changes.
- OPM/MRST and commercial simulator equivalence are not claimed.
- C++ or numba migration should start only after larger profiling proves a concrete hotspot.
