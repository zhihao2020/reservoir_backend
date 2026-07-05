# Cross-Scale Upscaling Summary

- success: True
- case_id: cross_scale_upscaling_default
- similarity_score: 0.17871714285714285
- regime_shift_flag: True

## Scale Conversion

| field | lab | field | ratio |
| --- | --- | --- | --- |
| length_scale | 1.0 | 100.0 | 100.0 |
| time_scale | 10.0 | 1000.0 | 100.0 |
| pressure_scale | 100000.0 | 2000000.0 | 20.0 |
| permeability_scale | 1e-12 | 2e-13 | 0.2 |
| velocity_scale | 1e-06 | 0.001 | 1000.0000000000001 |
| flow_rate_scale | 1e-09 | 2e-05 | 20000.0 |
| porosity | 0.2 | 0.25 | 1.25 |

## Upscaling Assumptions

- arithmetic mean permeability: 2.3333333333333334e-12
- harmonic mean permeability: 1.7142857142857142e-12
- porosity volume average: 0.21333333333333335
- flow-rate scaling sanity: 20000.0

## Fine-Grid vs Coarse-Grid Comparison

| metric | RMSE | MAE | R2 | NRMSE | max abs error |
| --- | --- | --- | --- | --- | --- |
| pressure | 866.0254037844386 | 750.0 | 0.994 | 0.028867513459481287 | 1000.0 |
| saturation | 0.017320508075688707 | 0.014999999999999944 | 0.9917241379310345 | 0.03464101615137742 | 0.019999999999999962 |
| production | 0.75 | 0.625 | 0.9967236985802694 | 0.02142857142857143 | 1.0 |

## Non-Claims

- No complex upscaling solver.
- No multiscale finite-volume implementation.
- No history matching.
- No automatic calibration.
- No commercial simulator equivalence.
- No validation of black-oil models.
- No front-end.
- No UDP.

## Warnings

- regime shift indicated by low capillary/Peclet/gravity similarity scores
- production: mape used epsilon for 1 near-zero reference values
