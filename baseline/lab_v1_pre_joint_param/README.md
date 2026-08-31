# Baseline before joint \(C_f,T_{mf}\) parameterization

Recorded against `6dc5cd8` (`main`). Physics results are not changed here.

| Item | Value |
|------|--------|
| git SHA | 6dc5cd89e965077cd2ffda4c44dcaf577f582042 |
| parameterization | scalar `log_conductivity` (\(n_\theta=1\)) |
| `shape_factor` | 40 (fixed) |
| 30³ scale gate | ~85.5 s (linear ~65 s), target `<60 s` not met |
| face-port noiseless recovery | failed: \(C_f\) rel error ≈ 77% vs prior 70% |

See `failed_face_recovery.json`.
