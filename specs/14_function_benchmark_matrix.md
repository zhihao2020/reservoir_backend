# Function Benchmark Matrix

## 1. Core Principle

Current stage priority is requirement-level function hardening and benchmark
validation. 当前阶段功能优先，流程后置.

```text
Function hardening first.
Workflow design after contract confirmation.
```

The requirement functions are now relatively clear, while the client's final
workflow, frontend interaction, data formats, and acceptance route may still
change before contract confirmation. Therefore this stage does not over-design
a Petrel-like workflow, does not implement UDP, does not implement black-oil,
and does not add new solver algorithms.

The most important current task is to improve each functional module's
algorithm reliability, numerical accuracy, benchmark credibility, and
validation evidence.

## 2. Function Benchmark Matrix

| Function module | Requirement source | Functional objective | Input data | Output data | Current algorithm | Current implementation status | Known limitations | Candidate benchmark | Validation metric | Next hardening task | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Saturation inversion module | Saturation from multi-source signals | Convert resistivity / EM / acoustic signals to water saturation and confidence | resistivity, formation factor, porosity, cementation exponent, saturation exponent, EM signal, acoustic velocity / travel time | water saturation, confidence, warning, sensitivity report, benchmark report | Archie equation inversion; EM empirical inversion; Acoustic empirical inversion; uncertainty-weighted inversion / confidence / user-weight fusion | Done for 046 hardening; EM/acoustic remain empirical physics | EM/acoustic are empirical; no Bayesian inversion; no automatic calibration; no commercial petrophysical interpretation claim | Archie analytical formula check; synthetic noisy resistivity dataset; known Sw-resistivity pair; sensitivity test for porosity / m / n; uncertainty-weighted fusion benchmark | absolute Sw error, RMSE, confidence range, bounds, clipping count, normalized weights | deeper calibration datasets; outlier detection policy; uncertainty propagation across field fusion | High |
| Pressure field reconstruction module | 3D pressure reconstruction | Reconstruct pressure, face flux, velocity and mass balance | permeability field, boundary pressure, source/sink, fluid viscosity, grid geometry, extracted OPM/MRST reference metadata | 3D pressure field, face flux, velocity field, mass balance report, diagnostics report, benchmark report | finite-volume / TPFA pressure solve; structured Cartesian grid; Dirichlet boundary; source/sink term; face flux reconstruction | Done for prototype Cartesian cases and 047 benchmark hardening | Limited boundary/well model; no finite element solver; no corner-point / NNC; no full SPE1/SPE10 / OPM / MRST equivalence | 1D linear pressure analytical solution; 2D / 3D manufactured linear pressure field; OPM water-1ph metadata sanity; OPM SPE1CASE1 layered adapted pressure; MRST simpleIncompTPFA reference note; source/sink material balance; boundary sanity; flux conservation | L2/Linf pressure error, flux conservation error, mass balance residual, pressure range, monotonicity score, reference metadata coverage | full 051 open-source benchmark adaptation; stronger solver diagnostics; mixed-boundary diagnostics; larger heterogeneous cases | High |
| Saturation transport module | Oil-water saturation evolution | Advance water saturation from pressure-derived flux | pressure-derived flux, porosity, initial saturation, relperm parameters, time step, boundary injection saturation | 3D water saturation, production curve, material balance report, CFL report | upwind finite-volume transport; Corey relative permeability; fractional flow; explicit time stepping; CFL control; material balance check | Done for oil-water prototype | Explicit timestep restriction; simplified injection / wells | Buckley-Leverett 1D qualitative benchmark; front movement test; boundedness test; material balance test; CFL stability test | Sw bounds, front position, material balance error, max CFL | relative permeability table interpolation; shock-front benchmark; semi-implicit transport option; better well injection handling | High |
| Capillary / gravity enhancement module | Optional physics enhancement | Add capillary diffusion and gravity segregation trends | saturation field, capillary pressure parameters, density difference, gravity vector, permeability, relative permeability | capillary flux, gravity flux, updated saturation, stability report | capillary pressure model; capillary diffusion flux; gravity flux; water flux composer; combined capillary + gravity transport | Done for prototype optional paths | Explicit capillary can be restrictive; no table Pc yet | capillary smoothing benchmark; gravity segregation benchmark; combined transport stability benchmark; boundedness check; gradient reduction check; expected gravity direction check | gradient reduction, flux sign, max CFL, Sw bounds | capillary pressure table; semi-implicit capillary diffusion; vertical-equilibrium-style gravity benchmark | High |
| Simplified three-phase WOG module | Simplified WOG transport | Track Sw/Sg/So closure and phase fluxes | Sw, Sg, So, relperm parameters, viscosity, density, pressure flux, time step | Sw field, Sg field, So field, phase flux, phase production summary, closure report | three-phase Corey relative permeability; phase mobility; fractional flow; phase flux; explicit Sw / Sg transport; So = 1 - Sw - Sg closure | Done for simplified incompressible WOG | Current simplified three-phase WOG is not black-oil | Sw + So + Sg closure benchmark; phase saturation boundedness; three-phase material balance; phase flux consistency | closure error, bounds, phase balance error | three-phase relperm table; injection composition; gas mobility benchmark; black-oil design as future extension | Medium |
| Parameter field fusion module | Multi-source field integration | Fuse static/dynamic/inverted fields with confidence | porosity field, permeability field, dynamic pressure / saturation field, confidence weights | fused parameter field, confidence field, fusion report | weighted averaging; confidence-weighted fusion; NaN-aware fusion; field clipping; property consistency checks | Done for lightweight fusion | No geostatistical uncertainty / facies conditioning | known weighted average formula; synthetic property field fusion; NaN handling benchmark; confidence-weighted benchmark; field bounds check | allclose to known result, NaN count, clipping count | spatial smoothing; facies-conditioned fusion; uncertainty propagation; upscaling-aware fusion | Medium |
| Cross-scale similarity module | Lab-field similarity comparison | Compute dimensionless criteria and similarity score | lab/field descriptors | Re, Ca, Pe, mobility ratio, gravity number, dimensionless pressure/time, score | Reynolds number; Capillary number; Peclet number; Mobility ratio; Gravity number; dimensionless pressure; dimensionless time; similarity score = exp(-abs(log(field / lab))) | Done | Threshold interpretation is engineering heuristic | formula benchmark; identical descriptor score = 1; known ratio similarity score; missing optional criterion warning | formula error, score range, warning count | weighted criterion profiles; domain-specific threshold sets; report integration | Medium |
| Scale-effect analysis module | Cross-scale regime interpretation | Compare field/lab scale ratios and regime shifts | lab/field descriptors and dimensionless numbers | scale ratios, dominant force, flow regime, regime shift report | field / lab scale ratio; regime classification; dominant force detection; regime shift detection | Done | Thresholds need domain calibration | known scale ratio; capillary-dominated case; viscous-dominated case; gravity-dominated case; transport regime shift case | ratio error, classification match, shift flag | benchmark-derived thresholds; domain-specific regime map | Medium |
| Lab-field validation module | Curve comparison | Compare laboratory measured curve with field/simulation curve | time-series curves | RMSE, MAE, MAPE, R2, normalized RMSE, max absolute error, multi-curve summary | curve alignment; linear interpolation; RMSE; MAE; MAPE; R2; normalized RMSE; max absolute error; multi-curve aggregation | Done | No time-unit conversion / weighting yet | known curve metric values; partial overlap case; no-overlap error case; constant reference R2 warning; multi-curve partial failure | metric formula error, warning behavior, partial failure count | time-unit conversion; weighted curve validation; production curve templates | Medium |
| Result reporting module | Reproducible output | Export summaries, validation/profiling/accuracy reports | case reports and benchmark outputs | summary JSON, validation reports, profiling reports, test outputs | JSON / Markdown / CSV / NPY report generation | Partial / Done for current reports | No result catalog yet | report schema test; required key test; no NaN / Inf in report; result path consistency | schema key coverage, finite values | result catalog; field statistics; slice export; CSV export; VTK export; case comparison report | Medium |
| Future interface module | Frontend/backend exchange | Provide future frontend communication | command-style requests | case_id, result_dir, summary, report paths | UDP deferred; CLI / YAML / reports remain primary interface | Planned only | Contract workflow unknown | interface contract doc test | no UDP implementation, schema placeholder exists | Do not develop UDP before contract workflow is confirmed; future UDP / REST / WebSocket / frontend dashboard | Low |
| Future black-oil extension | Advanced reservoir engineering compatibility | Design future black-oil behavior | PVT, pressure, phase state, wells | black-oil state and reports | Not implemented | Planned | Requires Bo/Bw/Bg/Rs/Rv/bubble point/PVT/well controls | future black-oil design benchmarks | not applicable yet | black-oil model design; PVT table module | Low |

## 3. Benchmark Sources And Validation Metrics

The benchmark plan prioritizes small deterministic cases first:

- analytical / manufactured benchmarks for pressure and formulas
- qualitative physical benchmarks for transport trends
- open-source adapted benchmarks after core function hardening

The current benchmark suite is not a commercial simulator validation suite. It
is a development gate for numerical reliability and regression control.

## 4. Hardening Roadmap

```text
045_function_benchmark_matrix
046_saturation_inversion_hardening
047_pressure_solver_benchmark_hardening
048_saturation_transport_benchmark_hardening
049_parameter_fusion_benchmark_hardening
050_cross_scale_benchmark_hardening
051_open_source_benchmark_adaptation
052_result_catalog_and_export
053_black_oil_model_design
054_pvt_table_module
055_accuracy_acceptance_report
```

045 defines what to validate. 046-050 harden current requirement functions.
051 introduces open benchmark adaptation. 052 improves result management and
export. 053-054 are future black-oil preparation. 055 produces the final
accuracy acceptance report.

## 5. Explicit Non-Goals

This stage does not implement:

- Petrel workflow
- commercial simulator behavior
- black-oil
- UDP
- C++
- full SPE10 reproduction
- OPM Flow equivalence
- MRST integration
- history matching
- automatic calibration

Current work focuses on requirement-level function hardening and benchmark
validation.
