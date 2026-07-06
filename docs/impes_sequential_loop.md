# IMPES Sequential Loop

## Purpose

F3-04 adds a lightweight pressure-saturation sequential coupling loop for the
existing structured Cartesian oil-water prototype. The loop composes already
validated pressure, face-flux, saturation-transport, CFL, and production
diagnostic utilities.

The update order is:

```text
pressure -> flux -> saturation -> mobility -> pressure
```

This is an IMPES-style sequential workflow: pressure and total Darcy face flux
are recomputed from the current saturation-dependent mobility, then water
saturation is advanced explicitly with the existing upwind finite-volume
transport routine.

## Coupling Flow

For each time step:

1. Compute Corey relative permeability and phase mobility from `Sw`.
2. Build an effective pressure transmissibility field proportional to
   `k * (lambda_w + lambda_o)`.
3. Solve steady 3D pressure using the existing pressure solver.
4. Compute internal Darcy face fluxes using the existing velocity helper.
5. Add simulation-layer Dirichlet boundary face fluxes for transport and
   production accounting.
6. Run CFL diagnostics.
7. Advance `Sw` with the existing explicit 3D saturation solver.
8. Record pressure, flux, saturation, CFL, material balance, production rate,
   water cut, and cumulative production.
9. Repeat with updated mobility.

## Report Schema

`python -m reservoir_backend.simulation.impes_report` writes:

```text
accuracy_reports/impes_loop_summary.json
accuracy_reports/impes_loop_summary.md
```

The report contains:

- `case_id`
- `num_steps`
- `grid_shape`
- `dt`
- `pressure_min`
- `pressure_max`
- `sw_min`
- `sw_max`
- `max_cfl`
- `max_flux`
- `max_mass_balance_error`
- `production_curve`
- `final_water_cut`
- `breakthrough_time`
- `warnings`
- `limitations`

## Production Summary

The current production summary is boundary-based. It records total liquid rate,
water rate, oil rate, water cut, cumulative water, and cumulative oil at a
configured producer boundary. Breakthrough time is the first reported time at
which water cut reaches the configured threshold.

## Limitations

- No fully implicit simulator is implemented.
- No black-oil model or PVT behavior is implemented.
- No complex well-control model is implemented.
- No Peaceman industrial well model is implemented in this loop.
- No front-end, UDP, REST API, or database service is implemented.
- Pressure and saturation solver cores are reused and not rewritten by F3-04.
