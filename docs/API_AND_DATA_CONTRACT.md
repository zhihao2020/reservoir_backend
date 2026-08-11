# API 与数据契约

## CLI

```bash
# 单时刻
python -m reservoir_backend.pipeline.run --config config/sensor_case.yaml --output results/sensor_run

# 多时刻 CSV
python -m reservoir_backend.pipeline.run --config config/sensor_series_case.yaml --mode series --output results/series

# 形态发现
python -m reservoir_backend.pipeline.run --config config/sensor_series_case.yaml --mode discovery --output results/discovery

# ES-MDA 渗透率集成
python -m reservoir_backend.pipeline.run --config config/sensor_series_case.yaml --mode esmda --output results/esmda --ne 16 --na 3
```

### 输出

| mode | 产物 |
|------|------|
| slice | `mesh.csv`, `pressure.npy`, `saturation.npz`, `properties.npz`, `summary.json` |
| series | `t_XXXX/` 各时刻四场 + `series_summary.json` |
| discovery | `shape_indicator.npy`, `active_mask.npy`, coarse/fine 历史 |
| esmda | `k_mean.npy`, `k_std.npy`, `k_ensemble.npy`, `esmda_report.json`, `mean_history/` |

## Python API

见 `reservoir_backend.pipeline`：

- 四场：`build_mesh`, `reconstruct_pressure`, `run_time_slice`, …
- 时间序列：`load_sensor_series`, `run_time_series`, `run_shape_discovery`
- 集成：`run_esmda_permeability`, `generate_logk_ensemble`

## YAML

### 单时刻 `config/sensor_case.yaml`

- `bounds`, `grid`, `wells`, `sensors`, `priors`

### 多时刻 `config/sensor_series_case.yaml`

- `series.wells_csv` / `series.boundary_csv`
- `esmda`: ne, n_assimilations, k_mean, logk_std, corr_len_cells, obs_std_frac, seed

## CSV 契约

### wells（长表）

`time,well,pressure_pa,sw,so,sg`

### boundary（可选）

`time,side,pressure_pa` — side ∈ left/right/front/back/bottom/top

## 假设

- 压力求解需要渗透率先验；ES-MDA 在 log-k 空间更新集成。
- 饱和度以井点传感器为硬数据，空间用反距离权重。
- 物性局部反演为达西尺度 MVP；ES-MDA 仅同化井压（非全场历史拟合产品）。
