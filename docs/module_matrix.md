# Module Matrix

| Module | Capability | Status | Main Tests | Notes |
| ------ | ---------- | ------ | ---------- | ----- |
| Archie inversion | Resistivity to Sw inversion | Done | `tests/test_archie_inversion.py` | Supports scalar, ndarray, Field3D. |
| EM inversion | Empirical EM to Sw inversion | Partial | `tests/test_electromagnetic_inversion.py` | Linear and polynomial calibration, not Maxwell inversion. |
| Acoustic inversion | Empirical Vp to Sw inversion | Partial | `tests/test_acoustic_inversion.py` | Linear and polynomial calibration, not full Gassmann inversion. |
| Pressure solver | 1D / 2D / 3D steady pressure | Done | `tests/test_pressure_solver_*.py` | Cartesian finite-volume prototype. |
| Face flux / velocity | Darcy face flux and velocity | Done | `tests/test_velocity.py` | Uses pressure and transmissibility. |
| Relperm / fractional flow | Corey relperm and water fractional flow | Done | `tests/test_relperm.py`, `tests/test_fractional_flow.py` | Oil-water only. |
| CFL | Explicit transport CFL checks | Done | `tests/test_cfl.py` | Face-flux and pore-volume based. |
| 1D saturation | Explicit 1D water transport | Done | `tests/test_saturation_solver_1d.py` | Upwind fractional flow. |
| 3D saturation | Explicit 3D water transport | Done | `tests/test_saturation_solver_3d.py` | Cartesian x/y/z transport. |
| Field fusion | Confidence-weighted field fusion | Done | `tests/test_field_fusion.py`, `tests/test_field_mapper.py` | Same-grid fusion and point mapping. |
| Result manager | NPY / JSON / CSV outputs | Done | `tests/test_result_manager.py` | Case output validation. |
| CLI runner | YAML-driven case execution | Done | `tests/test_cli_run_case.py` | Supports dry-run and overrides. |
| Capillary pressure | Pc(Sw) models | Done | `tests/test_capillary_pressure.py` | Brooks-Corey, van Genuchten, none. |
| Capillary flux | Face capillary water flux | Done | `tests/test_capillary_flux.py` | Independent flux module. |
| Capillary transport | Optional 1D / 3D capillary coupling | Done | `tests/test_saturation_capillary_*.py`, `tests/test_capillary_pipeline.py` | Explicit transport path. |
| Gravity flux | Face gravity water flux | Done | `tests/test_gravity_flux.py` | z gravity convention with depth positive down. |
| Gravity transport | Optional 1D / 3D gravity coupling | Done | `tests/test_saturation_gravity_*.py`, `tests/test_gravity_pipeline.py` | Explicit transport path. |
| Combined flux composer | Compose advective, capillary, gravity water flux | Done | `tests/test_water_flux_composer.py` | Also builds conservative effective CFL flux. |
| Combined transport | Combined capillary + gravity transport | Done | `tests/test_saturation_combined_capillary_gravity_3d.py`, `tests/test_combined_pipeline.py` | YAML/CLI case available. |
| Validation scripts | Reproducible validation reports | Done | `tests/test_validation_harness.py`, `tests/test_combined_validation.py` | Includes combined dt sensitivity. |
| Profiling scripts | Runtime summaries | Done | `tests/test_combined_profiling.py` | C++ decision support. |
| UDP | Frontend communication | Deferred | lightweight regression only | Protocol unknown. |
| C++ | Native kernels | Planned | `tests/test_requirement_traceability.py` | Only after profiling justifies it. |
| Three-phase design | Simplified incompressible water-oil-gas design | Done | `tests/test_three_phase_design.py` | Transport design only. |
| Three-phase relperm | Corey-style three-phase relative permeability, mobility, and fractional flow | Done | `tests/test_three_phase_relperm.py` | Independent module only; no transport coupling. |
| Three-phase fractional flow | `fw`, `fo`, `fg` from phase mobility | Done | `tests/test_three_phase_relperm.py` | Enforces `fw + fo + fg = 1`. |
| Three-phase phase flux | Advective `Fw`, `Fo`, `Fg` from total Darcy face flux | Done | `tests/test_three_phase_flux.py` | Independent upwind phase-flux module; no saturation update. |
| Three-phase 1D transport | Explicit 1D water-oil-gas transport | Done | `tests/test_three_phase_transport_1d.py` | Independent module; not connected to CLI/YAML. |
| Three-phase transport | Explicit 1D/3D water-oil-gas transport | Done | `tests/test_three_phase_transport_1d.py`, `tests/test_three_phase_transport_3d.py` | Independent module only; not connected to CLI/YAML and current executable remains oil-water. |
| Three-phase pipeline | YAML/CLI simplified incompressible WOG case | Done | `tests/test_three_phase_pipeline.py` | Advective three-phase transport only; not black-oil and no PVT / Rs / Rv / bubble point. |
| Three-phase validation / profiling | Validation, dt sensitivity, and runtime profiling for three-phase case | Done | `tests/test_three_phase_validation.py`, `tests/test_three_phase_profiling.py` | Confirms closure, bounds, material balance, and small-case runtime; no C++ recommended yet. |
| Cross-scale analysis design | One-backend, two-module design for lab-field comparison | Done | `tests/test_cross_scale_design.py` | Design only; cross-scale implementation is planned and must not modify solver internals. |
| Similarity criteria | Reynolds, capillary, Peclet, mobility, gravity, dimensionless pressure/time | Planned | Not applicable | Future `042_similarity_criteria_module`. |
| Scale-effect analysis | Scale ratios and regime-shift detection | Planned | Not applicable | Future `043_scale_effect_analysis_module`. |
| Lab-field validation | Curve mismatch metrics and validation reports | Planned | Not applicable | Future `044_lab_field_validation_module`. |
| Black-oil model | Black-oil PVT / simulator behavior | Planned | Not applicable | Not equivalent to three-phase design; out of current scope. |
