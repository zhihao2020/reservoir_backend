# lab_v1 — 30 cm shale-oil laboratory digital twin

This is the **unique V1 product case**. `examples/lab/lab_cf.yaml` is a coarse
development fixture. `lab_apply.yaml` is a leftover two-region waterflood demo.

`make_lab_v1_face_twin()` (0.30×0.20×0.10 m, 4×2×1) is a **scientific
diagnostic fixture (M1a)**, not a coarsened 30 cm product model. M1b is
`case_dev.yaml` (0.30³ m, 4×4×2). Do not treat tiny recovery as product M1.

## What V1 does

Inputs: \(Q_{inj}(t)\), \(P_{prod}(t)\), \(P_{obs}\), \(S_{obs}(\sigma,x,y,z,t)\).

Outputs: reconstructed \(p\), \(S_w,S_o,S_g\), \(z_i\), and
\(\theta=(\log C_f,\log\beta_{mf})\) with \(T_{mf}=\beta_{mf}T_{mf}^{ref}\).

Saturation is already inverted upstream. Raw Archie / EM / acoustic inversion,
PINN, SRV, DFM/EDFM, AMR, thermal, zonal \(C_f\), per-cell \(K\), and fracture
half-length are out of scope.

## Layout

| File | Role |
|------|------|
| `case.yaml` | 30×30×30 product spec. Frozen until M3. Do **not** run ES-MDA on this in CI. |
| `case_dev.yaml` | M1b: 30 cm cube, 4×4×2, fracture-P / matrix-P / \(S_g\), Na=5. |
| `controls.csv` | Face inject rate + produce pressure. |
| `sensors.csv` | Product 30³ contract (bulk, 2 kPa, \(S_w\)). Not the M1b fixture. |
| `sensors_dev.csv` | M1b sensors: fracture-P 30 Pa (algorithmic), matrix-P 2 kPa, \(S_g\). |
| `experiment_design.yaml` | M1c lab envelope for `scripts/lab_v1_experiment_design.py`. |
| `cmg_gem/` | M2 CMG-GEM alignment deck + export contract. |
| `pvt.yaml` | Published C1–nC10 EXAMPLE card. |
| `pvt_lumped.yaml` | 4 pseudo-component characterization template. |
| `truth/` | Written by `scripts/lab_v1_generate_truth.py`. |

## Boundary conditions

- Left face (`xmin`): rate-controlled port. Total \(Q_{inj}\) is split by WI
  so \(\sum_i Q_i = Q_{inj}\) (900 cells on 30³).
- Right face (`xmax`): pressure-controlled port.
- Other faces: no-flow.

## Commands

```bash
python scripts/lab_v1_generate_truth.py --dev --case B
python scripts/lab_v1_offline.py --dev
python scripts/lab_v1_sensitivity.py --dev
python scripts/lab_v1_online_replay.py --dev
python scripts/lab_v1_gate.py --dev
python scripts/lab_v1_cmg_forward_gate.py --wiring
python scripts/lab_v1_cmg_forward_gate.py --export examples/lab_v1/cmg_gem/export
python scripts/lab_v1_cmg_invert.py --export examples/lab_v1/cmg_gem/export
reservoir validate examples/lab_v1/case_dev.yaml
reservoir replay experiments/EXP001 --output results/replay
```

Product 30³ forward (not ensemble):

```bash
python scripts/lab_v1_gate.py
```

## M1c experiment-design gate

Do **not** retune ES-MDA. The gate is joint \(H+R+u(t)\) under a laboratory
envelope, not “the solver ran”. Each candidate is a handful of deterministic
forwards.

```bash
python scripts/lab_v1_experiment_design.py --yaml examples/lab_v1/experiment_design.yaml
```

Exit code 1 with `n_identifiable=0` is the intended scientific result.

H must be labeled: `bulk_gauges` (real default), `tapped_channel` (assumption),
`dp_transducer` (write the bench spec; never invent 30 Pa). Two independent
2 kPa absolute gauges give \(\sigma_{\Delta P}\approx 2.83\,\mathrm{kPa}\);
subtracting them does not improve SNR. \(R\) is a covariance (bias +
\(\tau=5\,\mathrm{s}\) temporal correlation), not \(\mathrm{diag}(\sigma^2)\).

| design | H | \(D_{C_f}\) | \(D_{T_{mf}}\) | cond | ΔPmax | PV | feasible | \(D_{C_f}>2\) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| constant | bulk | 0.019 | 0.92 | 1555 | 2.1 kPa | 0.037 | yes | no |
| constant_tapped | tapped | 0.075 | 0.03 | 13320 | 2.1 kPa | 0.037 | yes | no |
| long_constant | bulk | bound \(<1\) | — | — | ~2.1 kPa | \(<3\) | yes (bound) | no |
| pulse_1 | bulk | bound \(\ll 2\) | — | — | ~2.1 kPa | low | yes (bound) | no |
| pulse_rest | bulk | 0.021 | 7.68 | 35259 | 2.1 kPa | 0.015 | yes | no |
| pulse_rest_tapped | tapped | 0.095 | 0.05 | 400 | 2.1 kPa | 0.015 | yes | no |
| pulse_rest_dp | DP 200 Pa | **0.14** | 7.81 | 819 | 2.1 kPa | 0.015 | yes | no |
| multistep | bulk | bound \(\ll 2\) | — | — | ~2.1 kPa | low | yes (bound) | no |
| legacy_m1b_rate | tapped | 0.075 | 0.03 | 10946 | 2.1 kPa | **6.7** | **no** | no |

At \(q_{\max}=100\,\mathrm{mL/min}\) the sample \(\Delta P\approx 2.1\,\mathrm{kPa}\),
so a 5% \(C_f\) change is ~100 Pa against a 2 kPa absolute gauge. Steady \(C_f\)
is one \(\Delta P\), not a longer time series of the same drop. Pulse helps
\(T_{mf}\), not \(C_f\). **Accepted conclusion:** 2 kPa instrument + 30 cm
apparatus cannot support 5% \(C_f\) inversion. Change the instrument, the
excitation hardware, or the \(C_f\) tolerance — not ES-MDA.

## M2 CMG-GEM benchmark

Online Parameter EnKF is paused. Product acceptance is now

`CMG-GEM → sparse gauges → ES-MDA → F_ours(θ̂) → hidden full-field`.

See `cmg_gem/README.md`. GEM 2024.20 ran the alignment deck. First M2a numbers:
NRMSE_P = 0.39, RMSE_Sg = 0.63 — **FAIL**. Do not invert until forward fields match.
Inversion never opens `hidden/`.
