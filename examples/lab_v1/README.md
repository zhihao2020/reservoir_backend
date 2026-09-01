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
| `case.yaml` | 30×30×30 product spec. Frozen until M1c. Do **not** run ES-MDA on this in CI. |
| `case_dev.yaml` | M1b: 30 cm cube, 4×4×2, fracture-P / matrix-P / \(S_g\), Na=5. |
| `controls.csv` | Face inject rate + produce pressure. |
| `sensors.csv` | Product 30³ contract (bulk, 2 kPa, \(S_w\)). Not the M1b fixture. |
| `sensors_dev.csv` | M1b sensors: fracture-P 30 Pa (algorithmic), matrix-P 2 kPa, \(S_g\). |
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
reservoir validate examples/lab_v1/case_dev.yaml
reservoir replay experiments/EXP001 --output results/replay
```

Product 30³ forward (not ensemble):

```bash
python scripts/lab_v1_gate.py
```
