# Open-Source Reference Materials

This directory stores small upstream reference materials used to design
open-source-adapted benchmarks.

## Sources

- OPM `opm-tests`: `water-1ph/WATER2F.DATA`
- OPM `opm-tests`: `spe1/SPE1CASE1.DATA`
- MRST: `modules/book/examples/1phase/src/simpleIncompTPFA.m`
- MRST: `modules/book/examples/in2ph/buckleyLeverett1D.m`

The files in `upstream/` are reference materials only. They are not imported as runtime dependencies and are not executed by the Python backend.

## Extraction

Run:

```bash
python references/extract_reference_cases.py
```

Outputs:

- `references/fixtures/open_source_adapted_cases.json`
- `references/fixtures/open_source_adapted_arrays.npz`

## Policy

These fixtures are adapted reference cases. They do not claim:

- full SPE10 reproduction
- OPM Flow equivalence
- MRST runtime integration
- Egg full dataset import
- commercial simulator equivalence
