# `run` — one-command forward

Case YAML carries **grid / wells / fluid / schedule**. No extra CLI flags.

```bash
python -m reservoir_backend run examples/run/case.yaml
python -m reservoir_backend run examples/lab/lab_cf.yaml --output results/run
```

`run` is an alias of `simulate`. Invert is still `apply` / `invert`.

| CMG habit | YAML key | Loader |
|-----------|----------|--------|
| `*GRID` | `grid` | `io.grid_cfg` |
| `*WELL` / `*PERF` | `wells.file` or `wells.yaml` | `io.well_load` |
| `*EOS` / GEM card | `fluid.eos_yaml` / `fluid.gem_deck` / `fluid.file` | `io.eos_load` |
| well-control `*TIME` | `schedule` (alias of `controls`) | `io.case` |
| GEM well history | `well_history` | `io.well_history` (rates/BHP nRMSE) |

`fluid.gem_deck` is wired on main. There is no in-tree Jiyang GEM card; missing
paths refuse invented Tc/Pc. This example points at the published C1–nC10
`examples/lab_v1/pvt.yaml`.
