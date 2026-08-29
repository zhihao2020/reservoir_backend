# GEM ruler: Jiyang-pattern 1-inj 4-prod CO2 huff-n-puff

Offline CMG GEM deck for **well-history** invert checks. Product code does not call GEM.

| Item | Value |
|------|--------|
| Clone | `D:\Tool\CMG\GEM\2024.20\TPL\spr\gmspr003.dat` (CO2 pattern flood; `*SWT`/`*SGT` copied) |
| Fluid | EXAMPLE PR **CO2 + C1 + nC10** from `examples/compositional/fixtures/comp_c1c10co2.yaml` (OPM `1D_COMP.DATA`) |
| Not | Jiyang field `.gem`. Tc/Pc are published species, not invented shale oil |
| Grid | 21×21×5 Cartesian, 1260×1260×40 m (`examples/jiyang/jiyang_hnp.yaml`) |
| Wells | INJ J=11; P1 J=3; P2 J=7; P3 J=15; P4 J=19; I=4..18; k=3 |
| K | region map `jiyang_frac_regions.npy`; GEM uses 0.05 md (matrix) / 5 md (SRV). Product SI npy is too stiff for first Newton |
| Schedule | 90 d depletion, then **one** year-cycle: inj 1 mo / soak 1 mo / prod 10 mo |
| Gate | injector BHP + producer oil rate. Producer BHP is the control. **Not** field Dice |

```bash
python validation/jiyang/cmg_co2_hnp/build_deck.py
python validation/jiyang/cmg_co2_hnp/run_gem.py
python validation/jiyang/cmg_co2_hnp/extract_well_history.py
python validation/jiyang/cmg_co2_hnp/run_compare.py          # GEM vs F(m_true) BHP
python validation/jiyang/cmg_co2_hnp/run_compare.py --invert # LM; ~50 min per forward
```

Product case: `examples/jiyang/jiyang_co2_hnp.yaml` (EXAMPLE card + GEM well-history CSV). 21×21×5 正演约 50 min。Invert 是 contrast + LM。观测是 INJ BHP 和采井产油率。INJ BHP / \(q_o\) 未对齐 GEM 前不跑全井网 invert。

`run_gem.py` needs a local GEM 2024.20 license. Without it the `.dat` is still the deliverable.

Do not restore `cmg_harness`. Do not edit `solver/fi.py` or `examples/lab/lab_apply.yaml`.
