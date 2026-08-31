# V1 laboratory digital twin — acceptance gates

V1 product case: `examples/lab_v1/`. Saturation observations arrive already inverted
(\(S_\alpha,\sigma,x,y,z,t\)). Raw electrical / electromagnetic / acoustic inversion,
PINN, SRV, DFM/EDFM, AMR, thermal, zonal \(C_f\), per-cell \(K\), and fracture
half-length are out of scope.

| Gate | Requirement | Evidence |
|------|-------------|----------|
| DPDP D0–D4 | pass | `tests/comp/test_dual_d0.py`, `test_dual_d1234.py` |
| Mass conservation | \(<10^{-4}\) | DPDP step reports; lab gate `mass_error` |
| FastPR parity | pass | `tests/physics/test_flash_backend.py`, `test_realfluid_flash.py` |
| 30³ scale gate | `<60 s` | `scripts/dpdp_scale_gate.py` — currently ~85.5 s (linear ~65 s). Not yet met. |
| Lab Gate | stable face BCs + sensors + composition | `scripts/lab_v1_gate.py` (distinct from `dpdp_scale_gate`) |
| Synthetic noiseless \(C_f\) | \(\lvert C_f^{P50}-C_f^{true}\rvert/C_f^{true}<5\%\) | `scripts/lab_v1_offline.py --dev` |
| Synthetic noisy \(C_f\) | recommended \(<10\)–\(15\%\) | same script `--noise` |
| Holdout RMSE | posterior / prior \(<0.7\) when noisy | offline `report.json` |
| Online old-observation reuse | 0 | `TwinRuntime.observe` + replay |
| Per-member state consistency | pass | `TwinLoops.from_posterior` requires DualState at \(t>0\) |
| Sensor dropout | recover | Observation QC; Case C |
| One failed ensemble member | recover | `replace_failed_members` including last posterior forward |
| Fast pressure reuse | `<1 s` | frozen-\(\lambda\) `fast_step` |
| Replay | complete | `reservoir replay experiments/EXP001` |
| UDP control update | functional | `TwinUDPProtocol` → `TwinRuntime.update_control` |
| Field snapshot | functional | `FieldStore` NPZ + `pressure_source` metadata |

Tag `v1.0-lab` only when every row is pass, including \(T_{30^3}<60\,\mathrm{s}\).

Default ensemble size is \(N_e=12\) (scan \(8,12,16,24,32\) in `scripts/esmda_ne_sweep.py`).
Do not swap ES-MDA for MCMC / adjoint / CMA-ES / PINN inversion.

PyAMG remains optional future work for the pressure Schur stage; V1 uses
Schur \(S_p\approx J_{pp}-J_{pn}D_{nn}^{-1}J_{np}\) + Jacobi.
