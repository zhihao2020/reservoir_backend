# GEM shailoil 导出 CSV 说明

本文件描述 `results/shailoil_10y/csv/` 里从 CMG GEM 页岩油模型抽出的数据。
源 deck 是 [`examples/shailoil.dat`](../examples/shailoil.dat)，输出 `.out` 在 `results/shailoil_10y/shailoil.out`。

重新导出：

```bash
python scripts/cmg_shailoil_compare.py --cmg-csv --case results/shailoil_10y/case --out results/shailoil_10y/csv
```

## 模型

30 cm 立方体实验室模型，GEM 2024.20，Peng–Robinson，14 组分，120 ℃。
初始压力 20 MPa，初始含水饱和度 0，注入井注纯 CO₂。

| 项 | 值 |
|---|---|
| 几何 | `origin = (0,0,0)` m，`extent = (0.30, 0.30, 0.30)` m |
| 网格 | 15×15×15，格子边长 0.02 m |
| 孔隙度 | 0.05（常值） |
| 水平渗透率 | 0.03 mD |
| 垂向渗透率 | 0.2 × 水平渗透率 |
| 时间 | 360 天年为一年，每月 30 天，共 10 年 120 步 |
| 首步 / 末步 | 30 天 / 3600 天（`time_s` 2 592 000 / 311 040 000） |

Deck 用 `*KDIR *DOWN`（K=1 在顶）。导出时已翻成软件约定：`k=0` 在底、`k=14` 在顶。
全场、测点、井轨迹用同一套 0 起始下标：

```
cell = k * 15 * 15 + j * 15 + i
x = (i + 0.5) * 0.02
y = (j + 0.5) * 0.02
z = (k + 0.5) * 0.02
```

`i,j,k` 均为 0…14。不要把 CSV 里的 `i,j,k` 当成 GEM 1 起始块号。

## 文件一览

| 文件 | 内容 | 行数（不含表头） |
|---|---|---|
| `cmg_fields.csv` | 全场压力、饱和度、孔隙度、渗透率 | 120 × 3375 = 405 000 |
| `cmg_probes.csv` | 27 个测点的 p / sg / so / sw（底层 3×3×3） | 120 × 27 × 4 = 12 960 |
| `cmg_wells.csv` | 5 口井的井底压力和注采量 | 120 × 5 = 600 |
| `cmg_well_paths.csv` | 每口井井底–井趾两端 | 5 |
| `cmg_well_trajectories.csv` | 每口井每个射孔格子 | 51 |

## `cmg_fields.csv` 全场

| 列 | 含义 | 单位 |
|---|---|---|
| `time_s` | 模拟时间 | s |
| `cell` | 软件格子编号，见上式 | — |
| `i` `j` `k` | 0 起始下标 | — |
| `p_pa` | 压力 | Pa |
| `sw` `sg` `so` | 水 / 气 / 油饱和度 | — |
| `phi` | 孔隙度 | — |
| `k_m2` | I 方向渗透率 | m²（0.03 mD = 2.96077e-17 m²） |

每个时间步按 `cell = 0 … 3374` 连续写出。孔隙度、渗透率在本模型里几乎不随时空变化。

## `cmg_probes.csv` 测点

长表：每个时刻、每个测点、每个量一行。

| 列 | 含义 |
|---|---|
| `time_s` | 模拟时间，s |
| `probe` | 测点名 `P_{I}_{J}_{K}`，名字里的 IJK 是 GEM 1 起始块号 |
| `quantity` | `pressure` / `sg` / `so` / `sw` |
| `value` | 数值 |
| `unit` | 压力为 `Pa`，饱和度空 |

27 个测点在底层软件 `k = 0, 1, 2`（z = 0.01 / 0.03 / 0.05 m），平面 3×3，软件 `i, j ∈ {2, 7, 12}`（x, y = 0.05 / 0.15 / 0.25 m）。这与五口井的 (i, j) 对齐：中心是注入井，四角靠近四口生产井。

名字 `P_{I}_{J}_{K}` 仍用 GEM 1 起始块号：`I = i+1`，`J = j+1`，`K = 15 - k`（即 15 / 14 / 13）。

| 软件 k | z / m | GEM K | 9 个平面点（软件 i,j） |
|---|---|---|---|
| 0 | 0.01 | 15 | (2,2) (7,2) (12,2) (2,7) (7,7) (12,7) (2,12) (7,12) (12,12) |
| 1 | 0.03 | 14 | 同上 |
| 2 | 0.05 | 13 | 同上 |

例如底层中心为 `P_8_8_15`（0.15, 0.15, 0.01），四角底层为 `P_3_3_15`、`P_13_3_15`、`P_3_13_15`、`P_13_13_15`。

全表 27 行见 `cmg_probes.csv` 对应的 `probes.csv`。从已有全场 `cmg_truth.npz` 重采样，没有重跑 GEM。

## `cmg_wells.csv` 注采

| 列 | 含义 | 单位 / 符号 |
|---|---|---|
| `time_s` | 模拟时间 | s |
| `well` | `INJ` / `PROD1` … `PROD4` | — |
| `pw_pa` | 井底压力 | Pa |
| `q_m3s` | 储层条件下总量 | m³/s，注入为正、采出为负 |
| `qw_m3s` `qo_m3s` `qg_m3s` | 水 / 油 / 气分相流量 | 同上 |

控制条件（deck）：注入井最大 0.0072 m³/d（= 8.333e-8 m³/s）、井底压力上限 20 MPa；生产井井底压力下限 19 MPa。

## `cmg_well_paths.csv` 井底–井趾

每口井一行：`id, kind, x_m, y_m, z_m, x2_m, y2_m, z2_m`。
`(x,y,z)` 是射孔第一格中心（井底，z 大），`(x2,y2,z2)` 是最后一格（井趾，z 小）。井都是铅直的。

## `cmg_well_trajectories.csv` 射孔轨迹

每个射孔格子一行，沿井从井底到井趾，`seq` 从 0 递增。

| 列 | 含义 |
|---|---|
| `well` `kind` | 井名、injector / producer |
| `seq` | 沿井序号，0 = 井底 |
| `i` `j` `k` | 软件 0 起始下标，与全场 CSV 一致 |
| `x_m` `y_m` `z_m` | 该格中心，m |
| `cell` | 与全场 CSV 同一套编号 |

| 井 | 类型 | 射孔数 | 软件 (i, j) | 软件 k 范围（井底→井趾） |
|---|---|---|---|---|
| INJ | injector | 11 | 7, 7 | 14 → 4 |
| PROD1 | producer | 5 | 2, 12 | 14 → 10 |
| PROD2 | producer | 15 | 12, 12 | 14 → 0 |
| PROD3 | producer | 15 | 2, 2 | 14 → 0 |
| PROD4 | producer | 5 | 12, 2 | 4 → 0 |

井半径 0.003 m，皮肤 0。

## 和 `examples/online/` 的关系

`examples/online/` 的 `observations.csv` / `series.csv` / `probes.csv` / `wells.csv` 与这里的测点、注采、井底–井趾同源，但反演网格是 60×60×60（同一 0.30 m 立方体，格子更细）。全场 `cmg_fields.csv` 是 GEM 原生 15×15×15，不要和 60³ 反演场直接按 `cell` 对齐，除非先把 15³ 每格复制成 4×4×4。
