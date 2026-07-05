# ADR 004: Numerical Solver Scope

## Background

The repository contains pressure, saturation, capillary, gravity, simplified
three-phase, fusion, and cross-scale modules. It is not a commercial simulator.

## Decision

Keep current solver scope to structured Cartesian finite-volume/TPFA pressure,
explicit saturation transport, empirical/synthetic inversion validation,
field-fusion utilities, and independent cross-scale analysis utilities.

## Reasons

- Current tests and benchmarks support MVP numerical behavior.
- Larger commercial-simulator features would expand scope sharply.
- Explicit scope boundaries prevent overstating validation maturity.

## Alternatives

- Add black-oil PVT and fully implicit coupling immediately.
- Add finite-element pressure solver immediately.
- Add corner-point grids and production-grade well controls now.

## Impact

Black-oil, PVT tables, corner-point grids, NNC, LGR, fully implicit Newton,
history matching, automatic calibration, and commercial workflow features stay
outside the current solver scope.

## Risks

Stakeholders may expect field-scale simulator behavior from prototype module
names unless limitations remain visible.

## Revision Conditions

Revisit after acceptance requirements, field data formats, performance
targets, and validation benchmarks are formally agreed.
