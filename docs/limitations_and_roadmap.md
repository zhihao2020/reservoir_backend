# Limitations and Roadmap

## Current Limitations

The release-candidate prototype does not support:

1. black-oil PVT
2. solution gas / vaporized oil (`Rs` / `Rv`)
3. bubble point or phase appearance / disappearance
4. commercial-grade well controls
5. corner-point grid
6. NNC
7. local grid refinement
8. fully implicit Newton
9. geomechanics
10. thermal model
11. reactive transport
12. real-time frontend communication
13. production-scale parallel simulation

It is also not a Petrel replacement, not a CMG replacement, and not a
commercial reservoir simulator.

The three-phase flow design is completed in
`specs/12_three_phase_flow_design.md`, and the independent Corey-style
three-phase relperm / mobility / fractional-flow module and independent
advective phase-flux module are implemented. Independent 1D and 3D three-phase
transport and the YAML/CLI `three_phase_case.yaml` pipeline are implemented.
The three-phase path is simplified incompressible WOG and is not equivalent to black-oil modeling: black-oil PVT, solution gas,
vaporized oil, bubble point handling, and phase appearance / disappearance
remain future work. Three-phase validation/profiling is completed for the small
pipeline case; current small-case profiling does not justify C++ migration.

The cross-scale analysis design is completed in
`specs/13_cross_scale_analysis_design.md`. It keeps the product as one backend
with two first-level modules: the computational module and the cross-scale
analysis module. Cross-scale implementation is not yet complete. Similarity
criteria, scale-effect analysis, and the lab-field validation module remain
Planned. The first implementation will not perform history matching or
automatic parameter calibration.

## Engineering Hardening

- packaging
- CI
- API placeholder
- logging
- error handling
- sample data
- release tagging
- reproducible benchmark artifacts

## Physics Enhancement

- three-phase design completed
- three-phase relperm implementation completed
- three-phase phase-flux implementation completed
- three-phase 1D transport implementation completed
- three-phase 3D transport implementation completed
- three-phase pipeline case completed
- three-phase validation/profiling completed
- black-oil design
- gas phase
- well model
- relative permeability tables
- PVT tables
- optional capillary semi-implicit path if needed
- cross-scale analysis design completed
- similarity criteria module
- scale-effect analysis module
- lab-field validation module
- no history matching or automatic parameter calibration in the MVP

## Performance

- larger profiling cases
- sparse solver improvements
- vectorization review
- C++ kernel only if needed

## Interface

UDP is deferred because the frontend communication protocol is unknown. The
interface direction should be chosen after requirements clarify whether the
backend should expose UDP, TCP, REST, or a file-based exchange.
