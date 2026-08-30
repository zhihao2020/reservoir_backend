# V1 repository audit

Date: 2026-08-29. Labels: REUSE / ADAPT / REPLACE / REMOVE / NEW.

| Capability | Path | Decision |
|------------|------|----------|
| Cartesian grid, TPFA | `grid/`, `discretization/tpfa.py` | REUSE |
| Control / Observation / Sensor | `domain/types.py` | REUSE |
| ObservationOperator | `observation/operator.py` | REUSE |
| Ports, units, YAML/CSV case IO | `ports/`, `io/` | REUSE |
| Relperm, capillary, PVT, rock | `physics/` | REUSE (fixed in V1, not inverted) |
| IMPES / implicit transport / FIM | `solver/impes.py`, `transport.py`, `fi.py` | REUSE via ForwardModel adapter |
| EOS + compositional FIM | `eos/`, `comp/`, `solver/fi_comp.py` | ADAPT as FluidModel |
| 2-region / contrast log K + LM | `inverse/parameterization.py`, `inverse/lm.py` | ADAPT (lab transition until ES-MDA) |
| Coarse-field K | `CoarseFieldParameterization` | REMOVE |
| Fracture strip \(k_m,k_f,k_{srv},x_f\) | `inverse/frac.py`, `io/shale_case.py` | REMOVE |
| Jiyang field HNP | `examples/jiyang/`, `validation/jiyang/` | REMOVE |
| Black-oil CMG rulers | `validation/black_oil/` | REMOVE |
| Shale IMEX S1–S5 | `examples/shale_oil/`, `validation/shale_oil/` | REMOVE |
| Similarity / `--auto` structure search / `cmg_out` | `twin.similarity`、`inverse.structure`、`io.cmg_out` | REMOVE |
| Scalar \(C_f\) log parameterization | `inverse/log_conductivity.py` | NEW |
| Dual continuum state | `domain/state.py` | NEW |
| FractureConductivityModel | `physics/conductivity.py` | NEW |
| Warren-Root transfer | `physics/transfer.py` | NEW |
| FluidModel | `physics/fluid_model.py` | NEW |
| ForwardModel adapter | `ports/forward.py`, `solver/forward_adapter.py` | NEW |
| ES-MDA | `inverse/esmda.py`, `twin/history_match.py` | NEW |
| Parameter EnKF (one step) | `inverse/parameter_enkf.py` | NEW |
| UDP / online checkpoint | — | NEW (later) |
