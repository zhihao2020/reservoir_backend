# CMG-GEM cross-simulator benchmark (M2)

Product acceptance is no longer self-consistent synthetic truth
(`F_ours(θ_true) → Inverse_ours → F_ours(θ̂)`). That path is inverse crime.

The closed loop is:

```text
CMG-GEM truth → sparse observations → ES-MDA → F_ours(θ̂) → full-field vs hidden CMG
```

Inversion **must not** see the CMG 3-D field. Only `Q_inj`, `P_prod`, `P_obs`, `S_obs`.

## Layout

| Path | Role |
|------|------|
| `spec.yaml` | Alignment contract with `case_dev.yaml` |
| `lab_v1_dev.dat` | GEM deck (verify keywords on the local GEM 2024 TPL) |
| `export/observations.csv` | Invert input (virtual gauges) |
| `export/controls.csv` | Invert input |
| `export/hidden/` | Scoring only |

GEM 2024.20 is at `D:\Tool\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe`.

```bash
python scripts/lab_v1_cmg_run_gem.py
python scripts/lab_v1_cmg_compare_plot.py
python scripts/lab_v1_cmg_pack_obs.py --hidden results/lab_v1/cmg_gem_run/hidden
```

`lab_v1_cmg_compare_plot.py` defaults to `results/lab_v1/cmg_gem_run/hidden`.
Do not omit `--hidden` if you intend a different export.

Init-flash gate (t=0, no wells needed). Published card, copied (not imported):

- Tc/Pc/ω/Mw: Reid/Prausnitz/Poling 5th ed. (`pvt.yaml`)
- \(k_{ij}=0.049\): Katz–Firoozabadi; GEM `*BIN` with `*HCFLAG 0 0`
  (GEM `*PVC3` overwrites HC–HC when both `HCFLAG=1`)
- \(V_c\): OPM `opm-tests/compositional/1D_COMP.DATA` METHANE/DECANE
- \(\Omega_a,\Omega_b\): GEM `*OMEGA`/`*OMEGB` PR defaults (same as GEOS
  `CubicEOSPhaseModel.hpp`)

`C1`/`NC10` are GEM **library** names. The deck uses `METHANE`/`DECANE` so
user `*PCRIT`/`*BIN` are not replaced by the database.

GEM `*PCRIT` under `INUNIT *SI` is **bar**, not kPa.

After this card, t=0 is `d_sg≈0.006` (ours \(S_g=0.356\), GEM \(S_g=0.362\)),
still PASS at `sg_tol=0.05`. The residual is GEM's cubic vs the textbook PR
used by GEOS/OPM-style codes, not a missing published \(V_c\) or \(k_{ij}\).
Do not retune `kij`/`PVC3` to swallow it.

Transfer: GEM `*TRANSFER 0` (usual dual-porosity, same as
`q = σ k_m V λ Δp`) and `*SIGMAMF 80` = Kazemi `σ=40 m⁻²` times `β_mf=2`.
Rate: YAML `3e-4 m³/s` is GEM `*STG` surface gas; `F_ours` converts to mol/s
at the GEM separator (101.3 kPa, 15.6 °C). `case_dev.yaml` is unchanged.

Gravity / viscosity / WI (60 s, this cut):

- GEM `*VARI` + `*DEPTH-TOP *KVAR 0 0` (both layers same depth) so
  `gravity: false`. Layers now match. GEM warns "vertical overlap > 100%"
  — expected.
- `*VISCOR *MIX` with 0.020 / 0.30 cP (C1 / nC10).
- `*PERF *WI` 608.15 md·m = our `half_cell_wi` at \(k_f=10^{-12}\,\mathrm{m}^2\).
- RMSE\(_P\) ≈ 294 Pa (max |ΔP| 0.43 kPa). NRMSE\(_P\) is large because
  GEM's (Pmax−Pmin) is 0.3 kPa (ASCII 0.1 kPa). Both fields sit on the
  11.8 MPa producer BHP.
- RMSE\(_{S_g}=0.024\) (provisional gate 0.05: **PASS**). Inlet GEM 0.459
  vs ours 0.411; outlet 0.367 vs 0.362.

Optimization (linear kr matching `*SGT` + face \(T_x\) WI 304.08 md·m):
RMSE\(_P\) 294→**23 Pa**, NRMSE_σ 0.147→**0.011** (PASS), raw NRMSE 0.075
(PASS vs 0.10). RMSE\(_{S_g}\)=**0.026** (PASS vs 0.05). `*SLIMTUBE` was
tried and reverted — it is a 1-D tube, not a 4×2 face.

M2a forward equivalence **PASS**. Do not start M2b until `export/` is packed
from `results/lab_v1/cmg_gem_run/hidden`.

## Commands

```bash
python scripts/lab_v1_cmg_forward_gate.py --wiring
python scripts/lab_v1_cmg_forward_gate.py --export examples/lab_v1/cmg_gem/export
python scripts/lab_v1_cmg_invert.py --export examples/lab_v1/cmg_gem/export
python scripts/lab_v1_cmg_invert.py --export examples/lab_v1/cmg_gem/export --score
```

`--wiring` checks spec ↔ `case_dev.yaml` and the no-hidden-truth contract. It is not M2a PASS.

`--score` is the only path allowed to open `hidden/`. Invert without `--score` never loads it.

## Hidden truth files

`export/hidden/meta.json`:

```json
{
  "nx": 4, "ny": 4, "nz": 2,
  "times_s": [0.5, 30.0, 60.0],
  "cell_order": "k_j_i",
  "components": ["C1", "nC10"],
  "pressure_unit": "Pa"
}
```

Arrays, shape `(n_times, n_cells)` unless noted:

- `pressure.npy`, `sg.npy`, `so.npy`, optional `sw.npy`
- `z.npy` shape `(n_times, n_cells, n_comp)`
- optional dual: `pressure_fracture.npy`, `pressure_matrix.npy`

## KPI order (field before parameters)

1. pressure field NRMSE (range denominator)
2. Sg / So / Sw field RMSE
3. injection / production curves
4. hold-out sensor error
5. component field RMSE
6. `C_f`, `β_mf` parameter error
