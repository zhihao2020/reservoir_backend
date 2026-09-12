# Case schema (CMG-habit YAML)

The user command is one short forward:

```bash
python -m reservoir_backend run <case.yaml>
```

The case file carries **grid / wells / fluid / schedule**. CLI flags stay optional
(`--output` only). `run` aliases existing `simulate`; it does not open invert.

## Field map

| CMG habit | YAML (preferred) | Also accepted | Loader |
|-----------|------------------|---------------|--------|
| `*GRID` | `grid` | `geometry` + `grid` | `io.grid_cfg` |
| `*WELL` / `*PERF` / `*GEOMETRY` | `wells.file` (`.inc` snippet) | `wells:` list, `wells: wells.yaml`, `ports:` | `io.well_load` |
| `*EOS` / published card | `fluid.eos_yaml` | `fluid.file`, `physics.fluid.file` | `io.eos_load.load_eos_card` |
| GEM deck pointer | `fluid.gem_deck` | `physics.fluid.gem_deck` | same; missing file **refuses** invented Tc/Pc |
| well-control `*TIME` | `schedule` | `schedule.file`, `experiment.controls`, `controls` | `io.case` |

`fluid` may sit at the top level or under `physics`. First present pointer wins:
`file`, then `gem_deck`, then `eos_yaml`.

`schedule` is an alias of `controls`. First present wins:
`experiment.controls`, `experiment.schedule`, top-level `controls`, top-level `schedule`.
A mapping `{file: ...}` is the same shape as `wells.file`.

## Minimal forward case

See `examples/run/case.yaml`:

```yaml
grid: {type: cartesian, nx: 3, ny: 1, nz: 1}
physics: {model: compositional_dpdp, ...}
fluid: {eos_yaml: ../lab_v1/pvt.yaml}   # or gem_deck: real.dat
wells: {file: wells.inc}                # *WELL / *PERF snippet
schedule: schedule.csv                  # or inline list
```

Product lab cases (`examples/lab_v1/case_dev.yaml`, `examples/lab/lab_cf.yaml`)
already use this contract with `wells:` + `physics.fluid.file` + `experiment.controls`.
`run` loads them without renaming.

## Not in this cut

- No Jiyang GEM card in-tree. Do not invent Tc/Pc. Point `fluid.gem_deck` at a
  real file when one exists.
- Black-oil residual / `solver/fi.py` is out of scope. Forward is compositional
  (`compositional_dpdp` or `compositional`).
- Invert stays `apply` / `invert`. `run` does not grow those flags.
