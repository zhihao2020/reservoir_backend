# Probe series to full-grid pressure

User-facing wrapper around **existing** invert + forward. Not new physics
and not a 1 Hz online invert. Invert is one LM fit over the series;
after K is known the forward reports cell pressure at requested times.

## Function

```python
from reservoir_backend.twin.field import pressure_field, step_pressure

# probes + p(t) -> batch invert K -> p on every cell at report times
out = pressure_field(
    "examples/lab/lab_30cm.yaml",
    probes=[("P1", 0.08, 0.15, 0.05), ("P2", 0.22, 0.15, 0.25)],
    series={"times_s": t, "values": p_by_probe, "sigma": 2e3},  # or a CSV path
)
# out.pressure.shape == (n_times, n_cells)

# Skip invert when K (or a calibrate posterior) is already known
out = pressure_field("examples/lab/lab_30cm.yaml", k=k_mean, report_times=t)

# One solver dt on a fixed-K twin (not an invert)
state = step_pressure(twin, k_mean, state=state, dt=twin.physics.dt_init)
```

`series` CSV uses the existing observation schema
(`time_s,sensor,kind,value,sigma`). Case YAML still supplies grid / PVT /
controls. `DigitalTwin.reconstruct` is unchanged: ensemble UQ at **one** time.

## CLI

```bash
# Batch invert then write p(t). Case YAML has sensors; CSV is p(t).
reservoir reconstruct examples/lab/lab_30cm.yaml --series observations.csv --output results/field

# K already known: forward only
reservoir reconstruct examples/lab/lab_30cm.yaml --k k.npy --report-times times.npy --output results/field

# Same multi-time dump after the existing invert command
reservoir invert examples/lab/lab_30cm.yaml --self-check --write-field --output results/inv
```

Writes `pressure.npy` / `pressure_field.npz` with shape `(n_times, n_cells)`,
plus `times_s.npy` and `k.npy`.

Existing `invert --output` / `apply --output` still write the last-time UQ
snapshot (`pressure_mean.npy`, shape `(n_cells,)`).

## What this is not

- Not one invert per sample.
- `step_pressure` only advances the existing FIM/IMPES stepper one `dt` with
  K held fixed. There is no online per-second invert.
