# Saturation Inversion Validation

This document records the 046 saturation inversion hardening stage.

## Status

- saturation inversion hardening: Done
- Archie analytical benchmark: Done
- EM empirical inversion validation: Done
- Acoustic empirical inversion validation: Done
- multi-signal uncertainty-weighted fusion: Done
- sensitivity report: Done

## Implemented Scope

The hardened saturation inversion path covers:

- Archie resistivity inversion with strict input validation and clipping report
- empirical electromagnetic signal inversion using supplied linear or polynomial coefficients
- empirical acoustic signal inversion using supplied linear or polynomial coefficients
- multi-source saturation fusion using uncertainty, confidence, user, or equal weights
- inverse-variance weighting when uncertainty is supplied
- finite-difference Archie sensitivity report for `Rt`, `Rw`, `phi`, `m`, and `n`
- synthetic saturation inversion benchmark reports

## Benchmark

The benchmark is implemented in `benchmarks/saturation_inversion_benchmark.py`.
It checks:

- Archie analytical formula recovery
- noise sensitivity at 1%, 5%, and 10% resistivity noise
- uncertainty-weighted multi-signal fusion
- empirical inversion clipping and report fields

Manual run:

```bash
python benchmarks/saturation_inversion_benchmark.py
```

Outputs:

- `accuracy_reports/saturation_inversion_benchmark_summary.json`
- `accuracy_reports/saturation_inversion_benchmark_summary.md`

## Explicit Non-Goals

- No Bayesian inversion implemented.
- No automatic calibration implemented.
- No machine-learning inversion implemented.
- No commercial petrophysical interpretation claim.
- No solver modification.
