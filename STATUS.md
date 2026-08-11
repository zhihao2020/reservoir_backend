# 项目状态

主线：**传感器四场流水线**（见 `references/软件要求.txt`）。

| 状态 | 含义 |
|------|------|
| **已验证** | 有实现与针对性测试 |
| **MVP** | 可用但含明确简化假设 |
| **排除** | 不做 |

## 能力表

| 能力 | 状态 | 入口 | 证据 | 假设/边界 |
|------|------|------|------|-----------|
| 1 网格划分 | 已验证 | `pipeline.build_mesh` | `tests/test_pipeline_mesh.py` | 边界+井+dx/dy/dz → 序号/坐标 |
| 2 压力场 | MVP | 压力硬点插值/TPFA | `tests/test_pipeline_fields.py` | 井+`observer_p` 仅测 p |
| 3 饱和度场 | MVP | 饱和度硬点插值 | `tests/test_pipeline_fields.py` | 井+`observer_s` 仅测 S |
| 4 物性 | MVP | 点 k,φ → 自动空间插值 | `point_workflow` + `spatial_interp` | LOO-CV 选 IDW/普通克里金/堆叠；log-k；无用户方法配置 |
| 空间插值自动选择 | 已验证 | `pipeline.auto_interpolate_to_grid` | `tests/test_spatial_interp.py` | 点数不足或退化几何→IDW；LOO RMSE 差距超 5% 选优，否则 1/RMSE² 堆叠 |
| 测点分工 | 已验证 | `observer_p` / `observer_s` | `tests/test_pipeline_fields.py` | **同一测点不同时测 p 与 S** |
| 4 物性场 k/φ | MVP | `pipeline.invert_rock_properties` | `tests/test_pipeline_fields.py` | 达西 k；流量场；φ 物质平衡；**非均质数组** |
| 非均质四场验收 | MVP | `validation/heterogeneous_four_field/` | run_validate.py | **禁止均质**；通道/断层孪生 |
| 饱和度输运代理 | MVP | `pipeline.transport_water_saturation` | `tests/test_pipeline_fields.py` | **f_w(S)** 迎风 + 井产注量源汇 |
| 井产注量 | MVP | `SensorSample.well_rate` | 同上 | m³/s，+注 −采 |
| CMG 差距报告 | 已跑 | `validation/cmg_gap_report/` | GAP_REPORT.md | 非均质通道/断层 .out 对照 |
| 端到端 + CLI | MVP | `python -m reservoir_backend.pipeline.run` | `tests/test_pipeline_e2e_cli.py` | slice/series/discovery/esmda |
| CSV 多时刻传感器 | 已验证 | `pipeline.load_sensor_series` | `tests/test_sensor_series_inversion.py` | 注采井+observer_p/s 时序；边界 CSV |
| 时序点优先反演 | MVP | `run_time_series` | 同上 | 每时刻点优先；k/φ 跨时刻传递 |
| ES-MDA k 反演 | MVP | `pipeline.run_esmda_permeability` | `tests/test_sensor_io_esmda.py` | α 归一化；R 预条件；可选 GC 局部化；膨胀 |
| 方法学参考库 | 只读 | `references/methods/` | methods/README.md | equinor/pyesmda/dass；**禁止 import** |
| 多时刻形态发现 | MVP | `pipeline.run_shape_discovery` | `tests/test_shape_discovery.py` | 指标=ΔSw+k+Δp；跨时刻 k/φ 数组传递 |
| k–p 固定点迭代 | MVP | `pipeline.run_time_slice` | `tests/test_pipeline_fields.py` | 默认 2 次；加密后映射 k 场 |
| 正交指示加密 | MVP | `pipeline.refine_mesh_by_indicator` | 同上 | 高指示区 bbox 全局加密 |
| 合成通道孪生 | 已验证 | `pipeline.build_channel_twin` | `tests/test_shape_discovery.py` | 已知通道 mask；Dice 软阈值 |
| CMG 三维通道验证 | MVP | `validation/cmg_channel_3d/` | IMEX Normal Termination + report | `*VARI`+`*DTOP` 起伏山脊；非水平层 |
| CMG 断层通道验证 | MVP | `validation/cmg_fault_3d/` | IMEX Normal Termination + report | `*FAULT` throw + `*TRANSI` 封闭/窗；狗腿通道 |
| 合成断层孪生 | MVP | `pipeline.build_faulted_channel_twin` | `tests/test_shape_discovery.py` | 低渗断层带 + 偏移通道 |
| 测点推荐 (DOE) | MVP | `pipeline.recommend_probes` / `place_uniform_probes` | `tests/test_probe_design.py` | maximin / variance / hybrid；exclusive p/S；无业务 YAML 开关 |
| CMG 虚拟测点扫 N | MVP | `validation/cmg_probe_study/` | PROBE_STUDY.md | 从 .out 全场 p/S 虚拟抽样；不改 CMG 井网 |

## 排除

- 角点网格 / LGR / NNC
- 黑油 PVT、工业井网
- OPM/MRST 运行时与等价声明
- REST/前端/UDP 产品
- 跨尺度、历史拟合产品套件
