# DPDP performance profile

Target: 27k-cell (30³) Cartesian compositional DPDP, scalar \(C_f\).
Not a second OPM/GEOS. Default developer pytest is `-m "not slow"`.

## Cost model

```text
forward ≈ N_Newton × (flash + Jacobian FD + linear solve)
ES-MDA  ≈ N_a × N_e × forward
```

Jacobian is colored FD: 7 colours × 2 continua × \((N_c+1)\) residual evaluations
(Cartesian 7-point distance-2 coloring, verified by `verify_coloring_no_row_collision`).
Flash was the dominant term because `flash_tp` restarted from Wilson K and a stability test on every cell.

## Mitigations in tree

| Layer | What |
|-------|------|
| Tests | `slow` / `dpdp` / `assimilation` / `scalability`. Default: `not slow` |
| Sparse J | CSC colored FD + GMRES/ILU (`solver/dpdp_jacobian.py`, `solver/linear.py`) |
| Cache | `DPDPModelContext` topology and \(T(k=1)\) |
| Flash | reuse K on full residual (two-phase); Jacobian FD stays Wilson so J matches R |
| Δt | Newton count + \(\Delta p\), \(\Delta z\) chop |
| Ensemble | `ProcessPoolExecutor` when `n_cells ≥ 125` |
| Online | `TwinLoops`: 1 s frozen-λ pressure with live wells; Parameter EnKF on `slow_interval_s` |

## How to measure

```text
python -m pytest -m "not slow"          # Codex / day-to-day
python -m pytest -m dpdp                # dual solver
python -m pytest -m "slow and assimilation"
python scripts/dpdp_scale_bench.py --max-n 10 --t-end 0.5
```

Step reports include `jac_s`, `solve_s`, `resid_s` (flash lives in residual/Jacobian FD).

## Measured split

Step notes record `jac_s`, `solve_s`, `resid_s`, `flash_s`. Measured (`docs/bench/dpdp_scale.json`, one accepted step, 7 colours):

| grid | wall_s | jac_s | flash_s | solve_s |
|------|--------|-------|---------|---------|
| 4×3×2 | 18 | 8.9 | 6.9 | 0.003 |
| 5³ | 97 | 45 | 38 | 0.013 |
| 10³ | 757 | 359 | 278 | 0.47 |

Flash + colored FD dominate; sparse solve is small. 20³/30³ stay on `scripts/dpdp_scale_bench.py`, not default pytest.

ES-MDA wall time ≈ \(N_a N_e\) × forward. Smoke is `3×1×1`, Ne=4, Na=1 (`tests/inverse/test_esmda_smoke.py`).

30³ is a milestone, not a default test.
