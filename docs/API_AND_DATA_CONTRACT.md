# API 与数据契约

## CLI

```bash
python -m reservoir_backend.pipeline.run --config config/sensor_case.yaml --output results/sensor_run
```

输出：

- `mesh.csv` — cell_id,i,j,k,x,y,z
- `pressure.npy` — 压力场
- `saturation.npz` — sw, so, sg
- `properties.npz` — permeability_m2, porosity
- `summary.json` — 摘要与井-单元映射

## Python API

见 `reservoir_backend.pipeline`：`build_mesh`、`reconstruct_pressure`、`reconstruct_saturation`、`invert_rock_properties`、`run_time_slice`、`save_fields`。

## YAML（`config/sensor_case.yaml`）

- `bounds`: xmin/xmax/ymin/ymax/zmin/zmax
- `grid`: dx, dy, dz（标量或向量）
- `wells`: name, x, y, z
- `sensors`: time, well_pressure, well_saturation `[sw,so,sg]`, boundary_pressure
- `priors`: permeability_m2, porosity, viscosity_pa_s

## 假设

- 压力求解需要渗透率先验（不适定问题正则化）。
- 饱和度以井点传感器为硬数据，空间用反距离权重。
- 物性反演为达西尺度 MVP，非地质统计学/EnKF。
