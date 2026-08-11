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
| 网格划分 | 已验证 | `pipeline.build_mesh` | `tests/test_pipeline_mesh.py` | 轴对齐包围盒；正交间距 |
| 压力场重建 | MVP | `pipeline.reconstruct_pressure` | `tests/test_pipeline_fields.py` / `test_pressure_solver_3d` | **矩阵井点 Dirichlet**；k 标量/数组先验 |
| 饱和度场重建 | MVP | `pipeline.reconstruct_saturation` | `tests/test_pipeline_fields.py` | 井点 IDW；sw+so+sg=1 |
| 物性反演 k、φ | MVP | `pipeline.invert_rock_properties` | `tests/test_pipeline_fields.py` | k 数组先验+迭代；φ 连续/物质平衡代理 |
| 端到端 + CLI | MVP | `python -m reservoir_backend.pipeline.run` | `tests/test_pipeline_e2e_cli.py` | slice/series/discovery/esmda |
| CSV 多时刻传感器 | 已验证 | `pipeline.load_sensor_series` | `tests/test_sensor_io_esmda.py` | 长表 wells + boundary CSV |
| ES-MDA k 反演 | MVP | `pipeline.run_esmda_permeability` | `tests/test_sensor_io_esmda.py` | α 归一化；R 预条件；可选 GC 局部化；膨胀 |
| 方法学参考库 | 只读 | `references/methods/` | methods/README.md | equinor/pyesmda/dass；**禁止 import** |
| 多时刻形态发现 | MVP | `pipeline.run_shape_discovery` | `tests/test_shape_discovery.py` | 指标=ΔSw+k+Δp；跨时刻 k/φ 数组传递 |
| k–p 固定点迭代 | MVP | `pipeline.run_time_slice` | `tests/test_pipeline_fields.py` | 默认 2 次；加密后映射 k 场 |
| 正交指示加密 | MVP | `pipeline.refine_mesh_by_indicator` | 同上 | 高指示区 bbox 全局加密 |
| 合成通道孪生 | 已验证 | `pipeline.build_channel_twin` | `tests/test_shape_discovery.py` | 已知通道 mask；Dice 软阈值 |
| CMG 三维通道验证 | MVP | `validation/cmg_channel_3d/` | IMEX Normal Termination + report | `*VARI`+`*DTOP` 起伏山脊；非水平层 |
| CMG 断层通道验证 | MVP | `validation/cmg_fault_3d/` | IMEX Normal Termination + report | `*FAULT` throw + `*TRANSI` 封闭/窗；狗腿通道 |
| 合成断层孪生 | MVP | `pipeline.build_faulted_channel_twin` | `tests/test_shape_discovery.py` | 低渗断层带 + 偏移通道 |

## 排除

- 角点网格 / LGR / NNC
- 黑油 PVT、工业井网
- OPM/MRST 运行时与等价声明
- REST/前端/UDP 产品
- 跨尺度、历史拟合产品套件
