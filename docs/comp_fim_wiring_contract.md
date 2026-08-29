# Compositional ↔ black-oil FIM wiring contract

Status: **EXAMPLE path open**. Black-oil residual code stays frozen.
Do **not** edit `reservoir_backend/solver/fi.py`.
Standalone compositional kernel is `eos/` + `comp/` + `solver/fi_comp.py` (EXAMPLE C1–nC10).

Licensed adaptation of OPM / GEOS *concepts* is allowed for structure.
Product identifiers must still follow `docs/fim_name_map.md` (no upstream class names).
Do **not** `import references/`. Do **not** invent GEM Tc/Pc as a Jiyang card.

## 1. What is frozen (black-oil)

| Piece | Location | Rule |
|-------|----------|------|
| Unknowns | `(p, Sw, x)` with `x = Rs` or `Sg` | Keep OPM-style switch |
| Switch | `switch_live_oil_unknown` | Appear → Sg=0; disappear → Rs=RsSat |
| Residuals | surface-volume water / oil / gas | Do not rewrite for ladder 1 psi |
| Jacobian | residual-consistent coloring FD | Do not rewrite blindly |
| Flag | `PhysicsSpec.fully_implicit` | Already on after un-shrunk invert; leave alone |
| Timestep knobs | Newton-count Δt, `dt_max` fuse | Owned by fim-dt; not this page |

Forbidden while frozen: sequential flash / hybrid gravity into FIM, restoring harness,
chasing FIM≡sequential liberation gap, wiring `comp/` into `solve_fi_step`.

## 2. Compositional kernel today (wired as a new path)

`comp/` + `solver/fi_comp.py` Newton (EXAMPLE), **separate** from `fi.py`:

| Mode | Unknowns | Residuals |
|------|----------|-----------|
| Mass only (lagged p) | `n_i` per cell | Component molar conservation |
| Coupled pressure | `(n_i, p)` | Mass + volume `R_p = n_tot·v_mix − V_pore` |
| Rate control | `(n_i, p, p_wf)` | Above + `R_wf = Q_spec − Σ q_PI` |
| Specified BHP | `(n_i, p)` | `p_wf` Dirichlet; rate is outcome |
| Soak | drop `p_wf` | Well off |

Accumulation (SI): `n_i = V_pore · (ξ_L S_L x_i + ξ_V S_V y_i)` after `flash_tp`.
`T` is still prescribed. Immiscible water is opt-in (`CompSpec.has_water`): extra mole unknown \(n_w\), volume \(n_{\mathrm{hc}} v_{\mathrm{mix}}+n_w v_w=V_\varphi\). Still not `fi.py`.

## 3. Target wiring shape (when xuzh opens it)

Goal: one fully implicit compositional step, **new module path**, not a rewrite of
black-oil `switch_live_oil_unknown` into EOS.

Suggested product layout (names are proposals; rename if they collide):

| Concern | Suggested home | Notes |
|---------|----------------|-------|
| EOS / flash / stability | `eos/` (already) | Keep standalone; no GEM numbers invented |
| Comp accumulation / TPFA / wells | `comp/` (already) | Stay import-clean vs `references/` |
| Coupled FI driver | **new** e.g. `solver/fi_comp.py` | Prefer *add* file over editing `fi.py` |
| Black-oil FI | `solver/fi.py` | Remains the black-oil path |

### Unknown vector (isothermal first cut)

Per active cell (and wells as needed):

- Primary: component moles `n_i` (or overall `z` + total moles — pick one and stick)
- Pressure `p`
- Optional: well `p_wf` under rate control

Do **not** keep black-oil `(Sw, Rs/Sg)` as primaries on the compositional path.
Saturations come from flash + molar volumes, not from an Sg↔Rs switch.

### Residual blocks

1. **Component mass**: accumulation − divergence(TPFA molar flux) − well molar source = 0  
2. **Volume / pressure**: `n_tot · v_mix(T,p,z) − V_pore = 0` (same idea as `comp/implicit_p`)  
3. **Well control**: rate residual or Dirichlet BHP (same idea as `comp/implicit_bhp`)

Flash is an *inner* property evaluation at `(T,p,z)`, not a sequential operator-split
pass after Newton (that was the black-oil liberation gap lesson).

### Jacobian

Same rule as black-oil FI: residual-consistent. Prefer coloring FD of the *same*
residual used by Newton until an analytic J is proven matching. Do not port
black-oil grow/at_cap branches into compositional.

## 4. Hard boundaries (never cross without xuzh)

1. No edits to black-oil `switch_live_oil_unknown` for compositional CO2.  
2. No `import references/` (open-darts / GEOS / MRST are idea sources only).  
3. No invented Jiyang Tc/Pc; real card via `fluid.gem_deck` or refuse.  
4. No restore of `cmg_harness`.  
5. No merge of `comp/` into `solve_fi_step` until: three-phase cut agreed, real `.gem` or explicit EXAMPLE-only flag, and xuzh says wire.  
6. Product code must not introduce banned upstream IDs from `fim_name_map.md`.  
7. Liberation ladder / FIM≡sequential 1 psi is **not** a wiring acceptance gate.

## 5. Acceptance when wiring opens

Minimum (EXAMPLE first, then real card):

- [x] New driver path; black-oil `fi.py` still passes existing 26+1  
- [x] One-cell / small-grid isothermal `(n_i,p)` Newton closes (||R|| drop documented)  
- [x] Rate and BHP well modes match current `comp/` tests  
- [x] Import guard: no `references/`, no banned FIM names  
- [x] Public PR card loader (`io/eos_load.py`, OPM 1D_COMP numbers in `examples/compositional/fixtures/comp_c1c10co2.yaml`)  
- [x] Immiscible water on the EXAMPLE twin (F+H+LM, `tests/cases/test_comp_water.py`)  
- [ ] Optional: same case vs IMEX/GEM *well history* (rates/BHP) — field L2 is not the gate  
  EXAMPLE twin: `examples/compositional/comp_example.yaml` / `tests/cases/test_comp_twin.py`. Jiyang GEM still needs a real card.  

## 6. Owners

| Role | Duty |
|------|------|
| fim-resid | This contract; black-oil residual freeze; review wiring PRs against §4–5 |
| fim-dt | Timestep / invert driver; IMEX 1+4 well-history ruler (not GEM) |
| comp-eos | `eos/` + `comp/` kernel growth until wiring |
| lab-review | Read-only: did anything touch `fi.py` / invent criticals / skip tests? |
| xuzh | Opens wiring; supplies or approves real `.gem` |

Last updated: 2026-08-23. Residual freeze remains until xuzh opens wiring. EXAMPLE invert ~40 s; water invert ~30 s. Both 12-cell two-strip.
