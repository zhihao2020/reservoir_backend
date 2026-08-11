# 架构

## 产品主线

```text
build_mesh → reconstruct_pressure → reconstruct_saturation → invert_rock_properties
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
