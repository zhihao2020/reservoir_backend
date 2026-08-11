# 架构

## 产品主线

```text
软件要求 1–4（每时刻 t）：

```text
1 build_mesh(边界, 井, dx/dy/dz)
2 reconstruct_pressure(井压, 边界P/流量, k先验)     → 全网格 p
3 reconstruct_saturation(井饱和度, 边界流量, p)      → 全网格 sw,so,sg
4 invert_rock_properties(p, S, 流量)                 → 全网格 k, φ（可非均质）
```

端到端：`run_time_slice` / `run_time_series`（k–p 固定点迭代）。

CMG 仅作**非均质**正演对照（起伏通道 / 断层狗腿），不进产品内核。
```

## 保留的包

| 路径 | 职责 |
|------|------|
| `reservoir_backend/core/` | Grid3D、Field3D、井、单位 |
| `reservoir_backend/solver/` | TPFA 压力、传导率、速度、输运辅助 |
| `reservoir_backend/pipeline/` | 四场产品 API |
| `reservoir_backend/io/` | 配置加载、结构化 deck 子集、场 NPZ |
| `config/sensor_case.yaml` | 传感器案例 |
| `references/` | 需求与只读上游 submodule |
| `tests/` | pipeline + 核心求解器测试 |
| `scripts/check_doc_code_consistency.py` | 文档路径检查 |

## 已删除（无用）

`harness`、`benchmarks`、`examples`、`accuracy_reports`、`validation_reports`、`profiling_reports`、`requirements`、  
`data`、`field_data`、`results`（包）、`utils`、跨尺度/融合/反演/工作流等旧包。

运行时输出目录：`results/`（仅产物）。
