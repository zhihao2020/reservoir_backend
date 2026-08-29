# Shale-oil invert examples (field IMEX analog). See validation/shale_oil/README.md.

Run (requires IMEX `.out` under validation/shale_oil/):

```bash
reservoir invert examples/shale_oil/s1.yaml --output results/shale_s1
```

Outputs: `invert.json`, `check83.json`, `residuals.csv`, `k_mean.npy`, `k_std.npy` (when post_ensemble enabled).
