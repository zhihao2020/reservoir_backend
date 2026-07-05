# Parameter Fusion Uncertainty

## Purpose

TASK-016 adds uncertainty-aware output for parameter field fusion. The goal is
to expose variance, standard deviation, confidence, fallback policy,
lightweight Kriging / GP-style prediction, and diagnostics without replacing
the existing IDW / confidence-weighted fusion baseline.

## Baseline Fusion Policy

Existing same-grid weighted and confidence fusion remains the baseline. This
stage does not rewrite `field_fusion.py` and does not change established
parameter fusion benchmark behavior.

## Variance / Standard Deviation Weighting

When variance is supplied, weights are:

```text
weight = 1 / variance
```

When standard deviation is supplied, weights are:

```text
weight = 1 / std^2
```

Zero variance or zero standard deviation is floored for numerical stability.
Negative, NaN, or Inf uncertainty values are rejected.

## Confidence Weighting

When confidence is supplied and variance/std are absent, confidence is used as
the fusion weight. Confidence must be finite and within `[0, 1]`.

## Fallback Priority

Weighting priority:

1. variance
2. standard deviation
3. confidence
4. explicit source weights
5. equal weights

NaN source values are ignored cell-wise. Cells with no valid source values are
reported with warnings and masked output.

## Lightweight Kriging / GP Interface

`reservoir_backend.fusion.kriging` exposes a lightweight spatial prediction
interface. If optional `sklearn` Gaussian-process support is available and the
requested method allows it, it can be used. Otherwise the module falls back to
IDW prediction with a local weighted-variance uncertainty proxy.

This is an engineering interface and benchmarkable uncertainty output, not
commercial geostatistical modeling.

## Uncertainty Diagnostics

Diagnostics include:

- variance min / max / mean
- uncertainty nonnegative flag
- confidence range
- NaN / Inf counts
- masked cells
- bounds violations
- dominant source
- weighting policy
- fallback used

## Report Schema

Run:

```bash
python -m reservoir_backend.fusion.uncertainty_report
```

Outputs:

- `accuracy_reports/parameter_fusion_uncertainty_summary.json`
- `accuracy_reports/parameter_fusion_uncertainty_summary.md`

## Limitations

- No complete EnKF workflow.
- No ES-MDA history matching.
- No automatic calibration.
- No Bayesian inversion workflow.
- No commercial geostatistical modeling.
- No Petrel-like workflow.
- No front-end.
- No UDP.

## Non-Claims

The uncertainty output is not a closed-loop geological uncertainty workflow,
not automatic history matching, and not commercial geostatistical modeling.

## EnKF / ES-MDA Deferred Scope

EnKF and ES-MDA are explicitly deferred. Requests for these methods return a
deferred warning and do not update model states, pressure fields, saturation
fields, or production history.
