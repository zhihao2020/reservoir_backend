# Experimental Data Schema

TASK-008 defines the lightweight experimental data contract for the backend.
The schema maps raw experiment files into a standard internal
`ExperimentalDataset` containing named `ExperimentalField` arrays, units,
metadata, source name, input file, and input format.

## Standard Fields

| Field | Meaning | Canonical unit | Required by default | Physical rule |
| --- | --- | --- | --- | --- |
| `resistivity` | Electrical resistivity / Rt | `ohm_m` | optional | `resistivity > 0` |
| `electromagnetic_response` | Empirical EM response | `dimensionless` | optional | finite numeric |
| `acoustic_response` | Empirical acoustic response | `m_s` | optional | finite numeric |
| `pressure` | Pressure | `Pa` | optional | finite numeric |
| `saturation` | Water saturation or generic saturation | `fraction` | optional | `0 <= saturation <= 1` |
| `porosity` | Porosity | `fraction` | optional | `0 <= porosity <= 1` |
| `permeability` | Absolute permeability | `m2` | optional | `permeability > 0` |
| `temperature` | Temperature | `K` | optional | finite numeric |
| `time` | Time coordinate | `s` | optional | finite numeric |
| `x`, `y`, `z` | Cartesian coordinates | `m` | optional | finite numeric |
| `confidence` | Source confidence | `fraction` | optional | `0 <= confidence <= 1` |
| `variance` | Source variance / uncertainty | `variance` | optional | `variance >= 0` |

Required fields are selected by the caller, for example:

```python
read_experimental_data(path, required_fields=["time", "porosity", "permeability"])
```

## Units

The first implementation normalizes common project units:

- pressure: `Pa`, `kPa`, `MPa`, `bar`
- permeability: `m2`, `mD`, `D`
- fractions: `fraction`, `decimal`, `percent`
- time: `s`, `min`, `h`, `day`
- coordinates: `m`, `cm`, `mm`
- temperature: `K`, `C`

Unknown or missing units are reported in QC warnings. They are not silently
invented.

## Shape Conventions

Fields in one dataset should normally share the same shape. Supported shapes:

- 1D record arrays, such as time series;
- 2D arrays for tabular/spatial data;
- 3D arrays for structured grid fields with project convention `(nz, ny, nx)`.

Shape mismatch is reported by the QC pipeline and must be corrected before
using data as solver input.

## Metadata Fields

Dataset metadata may include:

- `source_name`
- `input_file`
- `input_format`
- source instrument or experiment ID
- raw column names
- reader warnings
- unit warnings

Metadata is preserved by the reader and included in QC reports.

## Example Dataset

CSV example:

```csv
time_s,porosity_fraction,permeability_md,pressure_mpa,resistivity_ohm_m
0,0.20,100,10.0,20
10,0.22,120,9.8,22
20,0.24,140,9.6,24
```

JSON example:

```json
{
  "source_name": "core_flood_001",
  "fields": {
    "time": {"values": [0, 10, 20], "unit": "s"},
    "porosity": {"values": [0.2, 0.22, 0.24], "unit": "fraction"},
    "permeability": {"values": [100, 120, 140], "unit": "mD"}
  },
  "metadata": {"operator": "lab"}
}
```

NPZ example:

```python
np.savez(
    "case.npz",
    time=np.array([0.0, 10.0]),
    time_unit=np.array("s"),
    porosity=np.array([0.2, 0.25]),
    porosity_unit=np.array("fraction"),
)
```

## Fixture Contract

TASK-009 adds reusable experimental-data fixtures under:

```text
tests/fixtures/experimental_data/
```

The fixture manifest records input paths, metadata paths, expected summaries,
pass/fail behavior, warnings, errors, and covered fields. See:

- `docs/data_contract.md`
- `tests/fixtures/experimental_data/manifest.json`
