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
| 压力场重建 | MVP | `pipeline.reconstruct_pressure` | `tests/test_pipeline_fields.py` | 需 k 先验；井点硬钉扎 |
| 饱和度场重建 | MVP | `pipeline.reconstruct_saturation` | `tests/test_pipeline_fields.py` | 井点 IDW；sw+so+sg=1 |
| 物性反演 k、φ | MVP | `pipeline.invert_rock_properties` | `tests/test_pipeline_fields.py` | 达西尺度估 k；φ 先验/弱更新 |
| 端到端 + CLI | MVP | `python -m reservoir_backend.pipeline.run` | `tests/test_pipeline_e2e_cli.py` | 合成传感器案例 |
| 多时刻形态发现 | MVP | `pipeline.run_shape_discovery` | `tests/test_shape_discovery.py` | 指标=ΔSw+k+Δp；非几何先验山体 |
| 正交指示加密 | MVP | `pipeline.refine_mesh_by_indicator` | 同上 | 高指示区 bbox 全局加密 |
| 合成通道孪生 | 已验证 | `pipeline.build_channel_twin` | `tests/test_shape_discovery.py` | 已知通道 mask；Dice 软阈值 |
| CMG 三维通道验证 | MVP | `validation/cmg_channel_3d/` | IMEX Normal Termination + report | `*VARI`+`*DTOP` 起伏山脊；非水平层 |

## 排除

- 角点网格 / LGR / NNC
- 黑油 PVT、工业井网
- OPM/MRST 运行时与等价声明
- REST/前端/UDP 产品
- 跨尺度、历史拟合产品套件
