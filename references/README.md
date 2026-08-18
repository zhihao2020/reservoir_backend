# Open-Source Reference Materials

This directory stores **read-only** upstream snapshots used to design adapted
benchmarks, structured-deck fixtures, and (where licensed) algorithm ports into
the product solver.

## Policy (compliance)

**Licensed adaptation is allowed.** Upstream OPM / GEOS (and related) trees may
be adapted into `reservoir_backend/` when company permission covers that use.

Hard rules:

1. **Never import upstream at runtime.** Python product code must not
   `import` modules from `references/**` (no `sys.path` injection, no linking
   Flow/GEOS as the forward operator \(F\)).
2. **Mandatory rename.** Adapted class names, function names, module names, and
   public APIs must **not** match upstream identifiers (e.g. do not ship
   `SimulatorFullyImplicit`, `FIBlackoilModel`, `CompositionalMultiphaseFVM`,
   or `AppleyardChop`). Use project names; see
   [`docs/fim_name_map.md`](../docs/fim_name_map.md).
3. **Do not claim equivalence.** Fixtures and ports do **not** claim full SPE10
   reproduction, OPM Flow equivalence, or commercial simulator equivalence.
4. **Keep snapshots read-only.** Do not copy upstream files into the product
   package under their original paths/names. Rewrite into project style
   (NumPy/SciPy) under local names.

Public APIs stay project-local (`load_structured_deck`, `StructuredDeckBundle`,
`solve_fi_step`, `PhysicsSpec.fully_implicit`, …).

## Submodules

```bash
git submodule update --init --depth 1
```

Configured in `.gitmodules`:

| Path | Remote |
|------|--------|
| `references/upstream/opm-tests` | https://github.com/OPM/opm-tests.git |
| MATLAB sequential-black-oil examples | local tree, offline reading only (see `.gitmodules`) |

Shallow clones (`--depth 1`) are recommended. The MATLAB tree is large; only a
few example paths are used as offline reading material for humans and for
optional path checks—never executed by this Python backend.

## Method references (ES-MDA / history matching)

See [methods/README.md](methods/README.md). Local shallow clones of equinor /
pyesmda / dass etc. Used only to **extract algorithms** into product modules
(self-contained; no runtime import of those trees).

## Extraction

```bash
python references/extract_reference_cases.py
```

Outputs:

- `references/fixtures/open_source_adapted_cases.json`
- `references/fixtures/open_source_adapted_arrays.npz`

The extractor only **reads file text** via `pathlib` / the project loader. It
does not import upstream packages.

## Project loader

```python
from reservoir_backend.io.structured_deck import load_structured_deck

bundle = load_structured_deck("references/upstream/opm-tests/spe1/SPE1CASE1.DATA")
grid = bundle.grid  # Grid3D with optional non-uniform spacing_k
```
