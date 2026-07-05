# Parameter Fusion Uncertainty Summary

- success: `True`
- num_cases: `5`
- num_passed: `5`

## Cases

| Case | Success | Key Metrics |
| --- | --- | --- |
| variance_weighted_uncertainty_fusion | True | `{"dominant_source": 1, "fallback_used": false, "fused_mean": 0.5882352941176471, "variance_mean": 0.23529411764705885, "weighting_policy": "variance"}` |
| confidence_weighted_uncertainty_fusion | True | `{"dominant_source": 1, "fallback_used": false, "fused_mean": 0.9, "variance_mean": 1.0, "weighting_policy": "confidence"}` |
| lightweight_kriging_gp_interface | True | `{"fallback_used": false, "method_used": "sklearn_gaussian_process", "prediction_mean": 1.9166662498133595, "variance_mean": 0.3645906283209424}` |
| explicit_weight_fallback_uncertainty | True | `{"fallback_used": true, "fused_mean": 2.5, "variance_mean": 0.25, "weighting_policy": "explicit_weight"}` |
| enkf_esmda_deferred_scope | True | `{"enkf_deferred": true, "esmda_deferred": true, "fallback_used": true}` |

## Limitations

- No complete EnKF workflow.
- No ES-MDA history matching implemented.
- No automatic calibration implemented.
- No Bayesian inversion workflow implemented.
- No commercial geostatistical modeling implemented.
- No Petrel-like workflow implemented.
- No front-end integration implemented.
- No UDP implementation.
