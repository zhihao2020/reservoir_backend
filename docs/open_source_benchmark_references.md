# Open-Source Benchmark References

This document records how open-source reference materials are used by the
benchmark suite. The project stores extracted metadata and small adapted arrays
under `references/fixtures/`; benchmark runners read those fixtures only.

No runtime dependency on OPM or MRST.
No exact full reproduction claim.
No OPM Flow equivalence.
No MRST integration.
No commercial simulator equivalence.
No black-oil validation.

## OPM water-1ph

- Source: `OPM/opm-tests/water-1ph/WATER2F.DATA`
- Registry use: adapted metadata / simple sanity reference
- Extracted metadata:
  - grid: `1x1x1`
  - porosity: `0.1`
  - permeability: `kx=1000 mD`, `ky=1000 mD`, `kz=100 mD`
- Reference type: property metadata sanity only
- Exact reproduction: false
- Runtime dependency: false

This case is used to verify that OPM-style single-cell property metadata can be
loaded into internal pressure sanity checks. It is not a full OPM Flow run.

## OPM SPE1CASE1

- Source: `OPM/opm-tests/spe1/SPE1CASE1.DATA`
- Registry use: adapted property metadata / heterogeneous sanity reference
- Extracted metadata:
  - grid: `10x10x3`
  - porosity: `0.3`
  - permeability range: `50-500 mD`
- Reference type: property metadata sanity only / adapted reference
- Exact reproduction: false
- Runtime dependency: false

This case supports layered heterogeneous pressure, saturation, capillary/gravity
property sanity checks. It is not full SPE1 reproduction, not SPE10
reproduction, and not black-oil validation.

## MRST simpleIncompTPFA

- Source: `SINTEF-AppliedCompSci/MRST/modules/book/examples/1phase/src/simpleIncompTPFA.m`
- Registry use: reference context only
- Reference type: reference context only
- Exact reproduction: false
- Runtime dependency: false

The pressure benchmarks use this as TPFA method context only. The project does
not execute MATLAB, does not integrate MRST, and does not claim MRST
equivalence.

## MRST buckleyLeverett1D

- Source: `SINTEF-AppliedCompSci/MRST/modules/book/examples/in2ph/buckleyLeverett1D.m`
- Registry use: reference context / qualitative comparison only
- Extracted metadata:
  - grid: `100x1`
  - permeability: `100 mD`
  - porosity: `0.2`
- Reference type: adapted reference / trend validation context
- Exact reproduction: false
- Runtime dependency: false

The saturation benchmark uses this as qualitative Buckley-Leverett front
movement context only. It is not full MRST reproduction and does not introduce a
MATLAB or MRST dependency.

## Policy

- References are not runtime dependencies.
- `references/upstream/*` and `references/fixtures/*` are read-only evidence
  inputs for benchmark hardening stages.
- Benchmark reports must describe OPM/MRST usage as adapted metadata,
  reference context, or qualitative inspiration.
- Reports must not claim OPM Flow equivalence, MRST equivalence, full SPE1 /
  SPE10 reproduction, commercial simulator equivalence, or black-oil
  validation.
