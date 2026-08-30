# Reference implementation notes

Local trees under `references/` are gitignored. Product code must not import them.

## MRST (dual-porosity-permeability compositional)

Path (local): `references/MRST-2026a/modules/dual-porosity-permeability/ad_models/`
or `references/upstream/mrst/...`

- `OverallCompositionCompositionalModelDPDP.m`
- `equations/equationsCompositionalDPDP.m`

Ideas used: separate TPFA on fracture and matrix; antisymmetric matrix–fracture transfer; not dual-porosity-only.

## ERT

Path (local): `references/ert-main/src/ert/analysis/`

- `_es_update.py`
- `_update_commons.py`
- `_update_strategies/_global.py`

Ideas used: ensemble update of parameters only; observation QC before the smoother; no `np.linalg.inv`.
