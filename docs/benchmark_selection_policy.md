# Benchmark Selection Policy

Benchmark validation is the next development driver. The benchmark set is split
into three classes.

## 1. Analytical / Manufactured Benchmark

Used to validate basic numerical formats and formulas:

- 1D linear pressure
- manufactured pressure field
- known formula checks
- known curve metrics

## 2. Qualitative Physical Benchmark

Used to validate physical trends:

- Buckley-Leverett front movement
- capillary smoothing
- gravity segregation
- three-phase closure

## 3. Open-Source Adapted Benchmark

Used later to improve credibility without adding heavy runtime dependencies:

- SPE10-like heterogeneity subset
- OPM-style mini waterflood
- MRST dataset idea adaptation
- Egg-style channelized synthetic model

Current limitations:

- No full SPE10 reproduction yet.
- No OPM deck parser yet.
- No MRST runtime dependency.
- No Egg full dataset import yet.
- No commercial simulator equivalence claim.
