# Open-Source Reference Materials

This directory stores **read-only** upstream materials used to design adapted
benchmarks and to exercise the project-local structured deck loader.

## Policy (compliance)

- Upstream trees under `upstream/` are **not imported as runtime dependencies**.
- Python code in this repository must **never** `import` modules from
  `references/upstream/**` (no OPM, no MRST, no sys.path injection).
- Public APIs use project-local names (e.g. `load_structured_deck`,
  `StructuredDeckBundle`, `Grid3D`). Do not copy third-party class or
  function names into this package.
- Fixtures are **adapted** metadata/arrays only. They do **not** claim:
  full SPE10 reproduction, OPM Flow equivalence, MRST runtime integration, or
  commercial simulator equivalence.

## Submodules

```bash
git submodule update --init --depth 1
```

Configured in `.gitmodules`:

| Path | Remote |
|------|--------|
| `references/upstream/opm-tests` | https://github.com/OPM/opm-tests.git |
| `references/upstream/mrst` | https://github.com/SINTEF-AppliedCompSci/MRST.git |

Shallow clones (`--depth 1`) are recommended. MRST is large; only a few example
paths are used as offline reading material for humans and for optional path
checks—never executed by this Python backend.

## Method references (ES-MDA / history matching)

See [methods/README.md](methods/README.md). Local shallow clones of equinor /
pyesmda / dass etc. Used only to **extract algorithms** into
`reservoir_backend/pipeline/esmda.py` and `ensemble_math.py` (self-contained).

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
