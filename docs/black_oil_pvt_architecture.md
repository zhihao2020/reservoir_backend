# Black-Oil / PVT Architecture Roadmap

## Purpose

This document defines a future architecture direction for black-oil and PVT
support. It is a design document only. The current backend does not implement a
black-oil solver, PVT table parser, Eclipse deck parser, or commercial
simulator equivalent workflow.

## State Variables

A future black-oil formulation would require explicit state variables beyond
the current simplified incompressible oil-water and WOG utilities:

- pressure;
- water saturation;
- oil saturation;
- gas saturation;
- solution gas-oil ratio `Rs`;
- vaporized oil-gas ratio `Rv` if supported;
- phase presence flags for water, oil, and gas;
- optional component or surface-volume bookkeeping.

## PVT Tables

Future PVT input contracts would need table families such as:

- `Bo`, oil formation volume factor;
- `Bw`, water formation volume factor;
- `Bg`, gas formation volume factor;
- oil viscosity, water viscosity, and gas viscosity;
- oil density, water density, and gas density;
- `Rs` as a function of pressure;
- `Rv` as a function of pressure if vaporized oil is in scope.

This stage does not implement a PVT table parser.

## Bubble Point and Phase Behavior

A black-oil model must handle:

- bubble point pressure;
- saturated and undersaturated oil regions;
- gas phase appearance;
- gas phase disappearance;
- oil phase appearance or disappearance only if the future scope requires it;
- phase mobility changes when phase presence changes.

The current backend does not implement phase appearance/disappearance logic.

## Surface Rates

Industrial reporting requires conversion between reservoir conditions and
surface rates. Future report fields should separate:

- reservoir oil rate;
- reservoir water rate;
- reservoir gas rate;
- surface oil rate;
- surface water rate;
- surface gas rate;
- cumulative surface volumes.

The current production summaries are lightweight reservoir-condition estimates
for existing synthetic workflows.

## Well Controls

A future black-oil schedule layer would need well controls such as:

- rate controls for oil, water, gas, or liquid;
- BHP control;
- group controls;
- injection composition;
- producer and injector switching;
- shut/open status;
- production constraints.

The current well schedule v0 only validates metadata and control interfaces.

## Schedule, Restart, and Report Step

Future case orchestration should define:

- schedule parsing;
- report step generation;
- restart state serialization;
- restart loading;
- report vectors for pressure, saturations, rates, and cumulative quantities.

No restart system is implemented by this document.

## Architecture Layers

Recommended future layers:

1. PVT table schema and parser.
2. Black-oil state container.
3. Phase behavior evaluator.
4. Black-oil relative permeability and capillary pressure hooks.
5. Well-control evaluator.
6. Sequential or implicit black-oil transport solver.
7. Restart and report writer.
8. Validation cases against analytical, manufactured, or adapted references.

## Limitations

- No black-oil solver implemented.
- No PVT table parser implemented.
- No Bo/Bw/Bg runtime evaluator implemented.
- No Rs/Rv runtime evaluator implemented.
- No bubble point calculation implemented.
- No phase appearance/disappearance implementation.
- No surface-rate production accounting implemented.
- No Eclipse, CMG, OPM, or commercial simulator equivalence.

## Non-Claims

This document is not an implementation claim. It does not claim Eclipse, CMG,
OPM Flow, SPE deck, or commercial simulator equivalence.
